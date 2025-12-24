# Ridge-Lasso-Regression-Analysis

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import joblib


# ================= STEP 3 — Load Dataset =================
df = pd.read_csv("csv_result-dataset (1).csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# City mapping — (Brazil → India)
city_mapping = {
    "Sao Paulo": "Mumbai",
    "Rio de Janeiro": "Delhi",
    "Porto Alegre": "Bengaluru",
    "Campinas": "Hyderabad",
    "Belo Horizonte": "Chennai"
}
df["City"] = df["City"].replace(city_mapping)

# Save after mapping
df.to_csv("csv_result-dataset (1).csv", index=False)

# Remove missing / duplicate values
df = df.dropna(subset=["rent_amount"])
df = df.drop_duplicates().reset_index(drop=True)



# ================= STEP 4 — EDA (optional) =================
# (You can comment these 4 graphs after viva)
sns.histplot(df["rent_amount"], kde=True)
plt.title("Rent Distribution")
plt.show()

sns.scatterplot(data=df, x="area", y="rent_amount", hue="City")
plt.title("Area vs Rent by City")
plt.show()

sns.boxplot(data=df, x="furnished", y="rent_amount")
plt.title("Furnished vs Unfurnished Rent")
plt.show()

numeric_cols = df.select_dtypes(include=np.number).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================= STEP 5 — Feature Selection =================
target = "rent_amount"
features = [col for col in df.columns if col not in [target, "total_amount"]]

X = df[features]
y = df[target]

categorical = ["City", "pets", "furnished"]
numeric = [col for col in features if col not in categorical]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= STEP 6 — Train & Compare ML Models =================
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append([name, rmse, mae, r2])
    print(f"{name} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
print("\nModel Comparison:\n", results_df)

# ================= STEP 7 — Select Best Model & Save =================
best_model_name = results_df.sort_values(by="R2", ascending=False).iloc[0]["Model"]
print("\nBest Model Selected:", best_model_name)

best_model = RandomForestRegressor(n_estimators=200, random_state=42)

final_pipeline = Pipeline(steps=[("prep", preprocessor),
                                 ("model", best_model)])

final_pipeline.fit(X, y)
joblib.dump(final_pipeline, "rent_model.pkl")
print("\n🎉 Final model saved successfully as rent_model.pkl")

