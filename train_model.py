import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

data = pd.read_csv("data/student_data.csv")
X = data.drop("final_score", axis=1)
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R² Score:", round(r2_score(y_test, predictions), 2))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/student_performance_model.pkl")
print("Model saved to model/student_performance_model.pkl")
