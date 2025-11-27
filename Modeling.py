# ------------------------------------------------------------
# COVID-19 Case Modeling in Homeless Populations
# Author: Safaa Hassan
# Description:
#   End-to-end exploratory data analysis and ML modeling pipeline.
#   Demonstrates data cleaning, visualization, feature selection,
#   and regression modeling using Decision Trees and Random Forests.
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
table1 = pd.read_csv("Table1_Full_Covid_Homeless.csv")
table2 = pd.read_csv("Table2_Full_Covid_Homeless.csv")

# ------------------------------------------------------------
# Basic Inspection
# ------------------------------------------------------------
print("Table 1 Summary:")
print(table1.info())
print(table1.describe(include='all'))

print("\nTable 2 Summary:")
print(table2.info())
print(table2.describe(include='all'))

# ------------------------------------------------------------
# Data Cleaning: Fill missing numeric values with column means
# ------------------------------------------------------------
for df in [table1, table2]:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------

# Histograms of numerical variables
table1.select_dtypes(include='number').hist(bins=15, figsize=(14, 10))
plt.suptitle("Distribution of Numerical Features — Table 1")
plt.tight_layout()
plt.show()

# Correlation heatmaps
def plot_heatmap(df, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()

plot_heatmap(table1.select_dtypes(include='number'), "Correlation Heatmap — Table 1")
plot_heatmap(table2.select_dtypes(include='number'), "Correlation Heatmap — Table 2")

# ------------------------------------------------------------
# Feature Reduction
# ------------------------------------------------------------
def reduce_features(df, target_col, threshold=0.8):
    corr = df.corr()
    high_corr = corr[target_col][corr[target_col] > threshold].index.tolist()
    high_corr.remove(target_col)  # remove self
    reduced_df = df.drop(columns=high_corr)
    return reduced_df

reduced_t1 = reduce_features(table1.select_dtypes(include='number'), "Total PEH Cases (%)")
reduced_t2 = reduce_features(table2.select_dtypes(include='number'), "Total Patients")

print("Remaining Table 1 Features:", reduced_t1.columns)
print("Remaining Table 2 Features:", reduced_t2.columns)

# ------------------------------------------------------------
# Model Preparation
# ------------------------------------------------------------

# Table 1 setup
X1 = reduced_t1.drop(columns=["Total PEH Cases (%)"])
y1 = reduced_t1["Total PEH Cases (%)"]

# Table 2 setup (simple)
if "Total Patients" in reduced_t2.columns:
    X2 = reduced_t2.drop(columns=["Total Patients"])
    y2 = reduced_t2["Total Patients"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# Baseline Models (Decision Tree)
# ------------------------------------------------------------
dt1 = DecisionTreeRegressor(random_state=42).fit(X1_train, y1_train)
y1_pred = dt1.predict(X1_test)

dt2 = DecisionTreeRegressor(random_state=42).fit(X2_train, y2_train)
y2_pred = dt2.predict(X2_test)

print("\nModel 1 (Table 1) — Decision Tree")
print("MSE:", mean_squared_error(y1_test, y1_pred))
print("R²:", r2_score(y1_test, y1_pred))

print("\nModel 2 (Table 2) — Decision Tree")
print("MSE:", mean_squared_error(y2_test, y2_pred))
print("R²:", r2_score(y2_test, y2_pred))

# ------------------------------------------------------------
# Feature Importance + Improved Model (Random Forest)
# ------------------------------------------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X1_train, y1_train)

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_n = 3

top_features = X1.columns[sorted_idx][:top_n]
print("\nTop Features (Table 1):", top_features)

# Train model with top features only
X1_train_sel = X1_train[top_features]
X1_test_sel = X1_test[top_features]

rf_top = RandomForestRegressor(random_state=42)
rf_top.fit(X1_train_sel, y1_train)
y1_pred_rf = rf_top.predict(X1_test_sel)

print("\nRandom Forest (Top Features)")
print("MSE:", mean_squared_error(y1_test, y1_pred_rf))
print("R²:", r2_score(y1_test, y1_pred_rf))

# ------------------------------------------------------------
# Plot Feature Importance
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(top_features, importances[sorted_idx][:top_n])
plt.title("Feature Importance — Random Forest (Top 3 Features)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y1_test, y1_pred_rf, alpha=0.7)
plt.plot([y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted — Random Forest (Table 1)")
plt.tight_layout()
plt.show()

