
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")

# Convert date_time into datetime format
df["date_time"] = pd.to_datetime(df["date_time"])

# Create useful time features
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.day_of_week

# Create a folder to save graphs
Path("charts").mkdir(exist_ok=True)

# GRAPH 1: Average traffic by hour
hourly_traffic = df.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
hourly_traffic.plot(kind="line", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/traffic_by_hour.png")
plt.close()

# GRAPH 2: Average traffic by day of the week
days = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

daily_traffic = df.groupby("day_of_week")["traffic_volume"].mean()
daily_traffic.index = days

plt.figure(figsize=(10, 5))
daily_traffic.plot(kind="bar")
plt.title("Average Traffic Volume by Day")
plt.xlabel("Day of the Week")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/traffic_by_day.png")
plt.close()

# GRAPH 3: Average traffic by weather
weather_traffic = (
    df.groupby("weather_main")["traffic_volume"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
weather_traffic.plot(kind="bar")
plt.title("Average Traffic Volume by Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("charts/traffic_by_weather.png")
plt.close()

print("Traffic analysis completed successfully!")
print("Three graphs saved in the charts folder:")
print("1. traffic_by_hour.png")
print("2. traffic_by_day.png")
print("3. traffic_by_weather.png")
