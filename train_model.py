
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

# 2. Create time features
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.day_of_week
df["month"] = df["date_time"].dt.month

# 3. Select input features and target
features = [
    "hour", "day_of_week", "month",
    "temp", "rain_1h", "snow_1h",
    "clouds_all", "weather_main"
]

target = "traffic_volume"

X = df[features]
y = df[target]

# 4. Split chronologically: earlier records train, later records test
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 5. Prepare numeric and categorical features
numeric_features = [
    "hour", "day_of_week", "month",
    "temp", "rain_1h", "snow_1h", "clouds_all"
]
categorical_features = ["weather_main"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "weather",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 6. Build a smaller Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=30,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training the machine learning model...")
model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} vehicles")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# 8. Save the compressed model
Path("models").mkdir(exist_ok=True)

model_path = "models/traffic_model.pkl"
joblib.dump(model, model_path, compress=3)

# 9. Display saved model size
size_mb = Path(model_path).stat().st_size / (1024 * 1024)

print(f"\nModel saved to {model_path}")
print(f"Model size: {size_mb:.2f} MB")
print("Training completed successfully!")
