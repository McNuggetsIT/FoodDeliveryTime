from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[["Age_norm", "Rating_norm"]] = scaler.fit_transform(df[["Delivery_person_Age", "Delivery_person_Ratings"]])
df["Age_Rating_score"] = (df["Age_norm"] + df["Rating_norm"]) / 2
df["Rider_Experience"] = df["Delivery_person_Age"] * df["Delivery_person_Ratings"]

from haversine import haversine

df["distance_real_km"] = df.apply(lambda row: haversine(
    (row["Restaurant_latitude"], row["Restaurant_longitude"]),
    (row["Delivery_location_latitude"], row["Delivery_location_longitude"])
), axis=1)


traffic_mapping = {"Low": 1, "Very low": 2, "Moderate": 3, "High": 4, "Very high":4}
df["Traffic_Level_num"] = df["Traffic_Level"].map(traffic_mapping)
df["Distance_Traffic"] = df["Distance (km)"] * df["Traffic_Level_num"]
df["Traffic_Weather_Risk"] = df["Traffic_Level_num"] * df["precipitation"]