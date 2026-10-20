import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


#1. parse weather data file 
def parse_custom_weather(file_path):
    weather_df = pd.read_csv(file_path)
    weather_df.columns = weather_df.columns.str.strip()
    weather_df['time_stamp'] = pd.to_datetime(weather_df['time_stamp'])
    return weather_df



#2. simulated route generation from malta to gozo only
#cirkewwa(35.9904, 14.33084)to mgarr (36.02482, 14.30271)
#each route has 25 waypoints (25 min crossing at 1-min intervals)
#with realistic lateral variation to simulate real navigational paths
MALTA_PORT  = (35.9904,  14.33084)#Cirkewwa
GOZO_PORT   = (36.02482, 14.30271)#Mgarr

def generate_synthetic_routes(n_routes=200, n_waypoints=25, seed=42):
    """generate n_routes realistic Malta to Gozo ferry paths
    each route interpolates from Cirkewwa to Mġarr across 25 waypoints
    (25 minutes at 1-minute intervals, matching real crossing duration)
    with small random lateral offsets to simulate real navigational variation.
    returns a list of DataFrames, one per route.
    """
    random.seed(seed)
    np.random.seed(seed)

    lat_start, lon_start = MALTA_PORT
    lat_end,   lon_end   = GOZO_PORT

    routes = []

    for i in range(n_routes):
        lats = np.linspace(lat_start, lat_end, n_waypoints)
        lons = np.linspace(lon_start, lon_end, n_waypoints)

        #add smooth lateral deviation — sine curve with random amplitude
        amplitude  = random.uniform(0.0005, 0.003)   #55m – 330m lateral swing
        phase      = random.uniform(0, math.pi)
        curve      = amplitude * np.sin(np.linspace(0, math.pi, n_waypoints) + phase)

        #perpendicular offset (west–east along the channel)
        lats += curve * random.uniform(-1, 1)
        lons += curve * random.uniform(-0.5, 0.5)

        #pin start and end exactly to the terminals
        lats[0], lons[0] = lat_start, lon_start
        lats[-1], lons[-1] = lat_end, lon_end

        df = pd.DataFrame({"latitude": lats, "longitude": lons})
        df["route_id"] = i + 1
        routes.append(df)

    return routes


#3. time stamp generation 
def generate_timestamps(df, start_time, interval_seconds=60):
    timestamps = [
        start_time + timedelta(seconds=j * interval_seconds)
        for j in range(len(df))
    ]
    df["timestamp"] = timestamps
    return df


#4. navigation physics, speed over ground and course over ground 
def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers


def calculate_cog(lat1, lon1, lat2, lon2):
    lat1, lat2  = math.radians(lat1), math.radians(lat2)
    diff_lon    = math.radians(lon2 - lon1)
    x = math.sin(diff_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def process_route_physics(df, time_interval=60):
    distances, sogs, cogs = [0], [0], [0]

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        dist = calculate_distance(
            (prev["latitude"], prev["longitude"]),
            (curr["latitude"], curr["longitude"])
        )
        sog  = (dist / time_interval) * 3600   # km/h
        cog  = calculate_cog(prev["latitude"], prev["longitude"],
                              curr["latitude"], curr["longitude"])

        distances.append(dist)
        sogs.append(sog)
        cogs.append(cog)

    df["distance_km"] = distances
    df["sog_kmh"]     = sogs
    df["cog_deg"]     = cogs

    #rolling smoothing + cap at Gozo ferry operational max 13.5
    df["sog_kmh"] = df["sog_kmh"].rolling(5, min_periods=1).mean()
    df["sog_kmh"] = np.clip(df["sog_kmh"], 0, 25)
    return df


#5. data fusion and feature engineering
def fuse_and_engineer(route_df, weather_df):
    route_df   = route_df.sort_values('timestamp')
    weather_df = weather_df.sort_values('time_stamp')

    merged_df = pd.merge_asof(
        route_df,
        weather_df,
        left_on='timestamp',
        right_on='time_stamp',
        direction='nearest'
    ).drop(columns=['time_stamp'])

    #cap simulated speed at 25 km/h — real Gozo ferry operational max
    merged_df["sog_kmh"] = np.clip(merged_df["sog_kmh"], 0, 25)

    #relative environmental angles
    merged_df["rel_wind_angle"]    = (merged_df["wind_direction_180m (°)"]    - merged_df["cog_deg"]) % 360
    merged_df["rel_wave_angle"]    = (merged_df["wave_direction (°)"]         - merged_df["cog_deg"]) % 360
    merged_df["rel_current_angle"] = (merged_df["ocean_current_direction (°)"] - merged_df["cog_deg"]) % 360

    #head/tail alignment components
    merged_df["headwind_factor"]   = np.cos(np.radians(merged_df["rel_wind_angle"]))
    merged_df["headwave_factor"]   = np.cos(np.radians(merged_df["rel_wave_angle"]))
    merged_df["headcurrent_factor"]= np.cos(np.radians(merged_df["rel_current_angle"]))

    #base power curve (SOG^3) scaled for a heavy double-ended diesel ferry
    #coefficient tuned so cruise speed (20 km/h) produces 280 L/h base burn
    base_power = np.power(merged_df["sog_kmh"].rolling(3, min_periods=1).mean(), 3)

    #scaled weather penalties — tuned to produce 20–120 L/h of additional burn
    #under the Mediterranean conditions in the weather dataset
    wind_res    = merged_df["wind_speed_180m (km/h)"]        * merged_df["headwind_factor"]    * 2.5
    wave_res    = merged_df["wave_height (m)"]               * merged_df["headwave_factor"]    * 80.0
    current_res = merged_df["ocean_current_velocity (km/h)"] * merged_df["headcurrent_factor"] * 50.0

    #fuel rate (L/h):
    #base_power * 0.38 → ~280 L/h at 20 km/h cruise (realistic for Malita-class ferry)
    #weather penalties add 20–120 L/h depending on conditions
    #300 L/h floor covers engine idle + hotel load when speed is very low
    merged_df["fuel_rate_l_h"] = np.maximum(
        300.0,
        (base_power * 0.38) + wind_res + wave_res + current_res
    )

    #true fuel volume consumed per 1-minute tracking interval
    merged_df["fuel_consumed_liters"] = merged_df["fuel_rate_l_h"] * (60 / 3600)

    return merged_df



#6. feature engineering for regression 
#route-level summary features fed to the ML model
def build_route_features(trip_final):
    """
    Collapse a per-waypoint trip DataFrame into a single-row
    feature vector that describes the full route.
    """
    return {
        "total_distance_km"        : trip_final["distance_km"].sum(),
        "mean_sog_kmh"             : trip_final["sog_kmh"].mean(),
        "mean_cog_deg"             : trip_final["cog_deg"].mean(),
        "std_cog_deg"              : trip_final["cog_deg"].std(),          # heading variation
        "mean_wind_speed"          : trip_final["wind_speed_180m (km/h)"].mean(),
        "mean_wave_height"         : trip_final["wave_height (m)"].mean(),
        "mean_current_velocity"    : trip_final["ocean_current_velocity (km/h)"].mean(),
        "mean_headwind_factor"     : trip_final["headwind_factor"].mean(),
        "mean_headwave_factor"     : trip_final["headwave_factor"].mean(),
        "mean_headcurrent_factor"  : trip_final["headcurrent_factor"].mean(),
        "max_wave_height"          : trip_final["wave_height (m)"].max(),
        "max_wind_speed"           : trip_final["wind_speed_180m (km/h)"].max(),
        "lateral_deviation_std"    : trip_final["latitude"].std(),         # path curvature proxy
    }


#7. regression model- gradient boosting 
def train_regression_model(features_df, target_series):
    """
    Train a Gradient Boosting Regressor to predict total fuel
    consumed per route from route + weather features.
    Returns the fitted model, scaler, and test-set metrics.
    """
    X = features_df.values
    y = target_series.values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    return model, scaler, mae, r2



#8. optimized route selection
#use the trained model to score all 200 routes and pick
#the one predicted to burn the least fuel
def select_optimised_route(model, scaler, features_df, all_routes_final):
    X_all       = scaler.transform(features_df.values)
    predictions = model.predict(X_all)

    best_idx    = int(np.argmin(predictions))
    best_route  = all_routes_final[best_idx]
    best_fuel   = predictions[best_idx]

    return best_route, best_idx + 1, best_fuel, predictions


#9. execution piepeline 
if __name__ == "__main__":

    weather_path = "weatherDATA/weather.csv"

    #load weather data 
    print("Parsing custom weather tables...")
    weather_df = parse_custom_weather(weather_path)

    #generate 200 synthetic malta to gozo routes with 25 waypoints each
    print("Generating 200 synthetic Malta→Gozo routes...")
    raw_routes = generate_synthetic_routes(n_routes=200, n_waypoints=25)
    print(f"  {len(raw_routes)} routes created.")

    #process each route through physics + weather fusion ---
    print("Running physics and weather fusion on all routes...")
    all_routes_final = []
    features_list    = []
    fuel_totals      = []

    #spread routes across the weather dataset timespan
    weather_start = weather_df['time_stamp'].min()
    weather_end   = weather_df['time_stamp'].max()
    time_span_hrs = (weather_end - weather_start).total_seconds() / 3600

    for i, route in enumerate(raw_routes):
        print(f"  Processing route {i+1}/200...", end="\r")

        #assign a start time spread across the weather window
        offset_hrs  = (i / 200) * time_span_hrs
        start_time  = weather_start + timedelta(hours=offset_hrs)

        route = generate_timestamps(route, start_time, interval_seconds=60)
        route = process_route_physics(route, time_interval=60)
        route = route.iloc[1:].reset_index(drop=True)   # drop first zero-speed row

        trip_final = fuse_and_engineer(route, weather_df)
        trip_final["route_id"] = i + 1

        all_routes_final.append(trip_final)
        features_list.append(build_route_features(trip_final))
        fuel_totals.append(trip_final["fuel_consumed_liters"].sum())

    print(f"\nAll 200 routes processed.")

    #assemble feature matrix and target vector
    features_df    = pd.DataFrame(features_list)
    target_series  = pd.Series(fuel_totals, name="total_fuel_liters")

    print(f"\nFuel stats across all routes:")
    print(f"  Min  : {target_series.min():.2f} L")
    print(f"  Max  : {target_series.max():.2f} L")
    print(f"  Mean : {target_series.mean():.2f} L")

    #train regression model
    print("\nTraining Gradient Boosting regression model...")
    model, scaler, mae, r2 = train_regression_model(features_df, target_series)
    print(f"  MAE : {mae:.3f} L")
    print(f"  R²  : {r2:.4f}")

    #select optimised route
    print("\nSelecting optimised route...")
    best_route, best_id, predicted_fuel, all_predictions = select_optimised_route(
        model, scaler, features_df, all_routes_final
    )
    actual_fuel = fuel_totals[best_id - 1]
    print(f"  Optimised Route ID  : {best_id}")
    print(f"  Predicted fuel      : {predicted_fuel:.2f} L")
    print(f"  Actual modelled fuel: {actual_fuel:.2f} L")


    #PLOT 1 — all routes + optimised route highlighted
    plt.figure(figsize=(9, 7))

    for route in all_routes_final:
        plt.plot(route["longitude"], route["latitude"],
                 color="steelblue", alpha=0.12, linewidth=0.8)

    plt.plot(best_route["longitude"], best_route["latitude"],
             color="crimson", linewidth=3,
             label=f"Optimised Route (ID {best_id})")

    plt.scatter(*MALTA_PORT[::-1], color="navy",  zorder=5, s=80, label="Cirkewwa (Malta)")
    plt.scatter(*GOZO_PORT[::-1],  color="green", zorder=5, s=80, label="Mġarr (Gozo)")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Gozo Ferry — All 200 Routes with Optimised Route")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optimised_route_map.png", dpi=300, bbox_inches="tight")
    plt.savefig("output.png")
    plt.close()


    #PLOT 2 — Predicted vs Actual fuel scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(target_series, all_predictions, alpha=0.5, color="steelblue")
    min_val = min(target_series.min(), all_predictions.min())
    max_val = max(target_series.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction")
    plt.xlabel("Actual Fuel (L)")
    plt.ylabel("Predicted Fuel (L)")
    plt.title(f"Regression Model — Predicted vs Actual Fuel\nMAE={mae:.3f} L   R²={r2:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("regression_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.savefig("output.png")
    plt.close()


    #PLOT 3 — Feature importance
    importances = model.feature_importances_
    feat_names  = features_df.columns.tolist()
    sorted_idx  = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feat_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue"
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance — Fuel Prediction Model")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig("output.png")
    plt.close()


    #PLOT 4 — Distribution of fuel across all routes
    plt.figure(figsize=(8, 5))
    plt.hist(target_series, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(actual_fuel, color="crimson", linewidth=2,
                label=f"Optimised route ({actual_fuel:.1f} L)")
    plt.xlabel("Total Fuel Consumed (L)")
    plt.ylabel("Number of Routes")
    plt.title("Fuel Distribution Across 200 Routes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fuel_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig("output.png")
    plt.close()


    #PLOT 5 — Wind speed vs fuel coloured by wave height
    combined = pd.concat(all_routes_final, ignore_index=True)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        combined["wind_speed_180m (km/h)"],
        combined["fuel_consumed_liters"],
        c=combined["wave_height (m)"],
        cmap="plasma", alpha=0.4, s=10
    )
    plt.colorbar(sc, label="Wave Height (m)")
    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Fuel Consumed (L per interval)")
    plt.title("Fuel Consumption vs Wind Speed (coloured by Wave Height)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("wind_vs_fuel.png", dpi=300, bbox_inches="tight")
    plt.savefig("output.png")
    plt.close()


    #OUTPUT SUMMARY
    print("\nOPTIMISED ROUTE PIPELINE COMPLETE:")
    print("------------------------------------------------------------------------------------")
    print(f"  Routes generated      : 200")
    print(f"  Model MAE             : {mae:.3f} L")
    print(f"  Model R²              : {r2:.4f}")
    print(f"  Optimised Route ID    : {best_id}")
    print(f"  Predicted fuel        : {predicted_fuel:.2f} L")
    print(f"  Actual modelled fuel  : {actual_fuel:.2f} L")
    print(f"  Worst route fuel      : {target_series.max():.2f} L")
    print(f"  Fuel saving vs worst  : {target_series.max() - actual_fuel:.2f} L")
    print("------------------------------------------------------------------------------------")
    print("\nOptimised route waypoints:")
    print(best_route[["latitude", "longitude", "sog_kmh", "cog_deg"]].to_string(index=False))
    print("------------------------------------------------------------------------------------")
