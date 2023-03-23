import pandas as pd

def gps_vehicle_overlap():
    stats = pd.read_csv("stats/feature_statistics.csv")
    all_features = stats.columns[1:]
    gps_vehicle = []
    vehicle = []
    overlap = []
    for feature in all_features:
        feature_split = feature.split("/")
        if feature_split[1] == "vehicle":
            vehicle.append(feature_split[1:])
        if len(feature_split) > 3:
            if feature_split[1] == "gps_m2" and feature_split[2] == "vehicle":
                gps_vehicle.append(feature_split[2:])
    for feature in gps_vehicle:
        if feature in vehicle:
            overlap.append(feature)
    print(overlap)
    print(len(overlap), len(vehicle), len(gps_vehicle))

if __name__ == "__main__":
    gps_vehicle_overlap()