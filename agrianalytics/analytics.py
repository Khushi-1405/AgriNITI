import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# --- CONFIGURATION ---
CSV_FILE = "/content/Agri_data.csv"  # replace with your actual CSV file path
OUTPUT_HTML = "mandi_analysis_dashboard.html"

# --- INSPECT INITIAL DATA ---
df_initial = pd.read_csv(CSV_FILE)
print("Initial data summary:")
display(df_initial.head())
print("\nInitial data info:")
df_initial.info()

# --- LOAD AND PREPROCESS DATA ---
df = pd.read_csv(CSV_FILE)
# df.dropna(inplace=True) # This was removing all rows, we'll handle NaNs later if needed.

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}


print("\n--- Model Comparison (MSE Scores) ---")
mse_scores = {}

df_encoded = pd.get_dummies(df[['State', 'Commodity', 'Grade']])
X = df_encoded
Y = df['Modal_x0020_Price'] # Use Modal_x0020_Price as the target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
for name, model in models.items():
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(Y_test, preds)
    mse_scores[name] = mse
    print(f"{name}: MSE = {mse:.2f}")


#Visualization of this prediction

plt.figure(figsize=(8, 5))
sns.barplot(x=list(mse_scores.keys()), y=list(mse_scores.values()), palette="viridis")
plt.ylabel("Mean Squared Error")
plt.title("Comparison of Regression Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png")

# --- ENCODING FOR ML ---
df_encoded = pd.get_dummies(df[['State', 'Commodity', 'Grade']])
X = df_encoded
Y = df['Modal_x0020_Price'] # Use Modal_x0020_Price as the target variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("MSE (Mean Squared Error):", mean_squared_error(Y_test, Y_pred))

#This means, Lower MSE (Mean Squared Error) = Better predictions

#Lower MSE (Mean Squared Error) = Better predictions

#A very high MSE may signal the need for:

#More features (e.g., weather, time, price fluctuations)
#Data normalization or scaling
#A different ML model or more data

df['Predicted Trade Value'] = model.predict(df_encoded)

# --- VISUALIZE ---
sns.set(style="whitegrid")
plt.figure(figsize=(18, 15))
sns.barplot(data=df, x="State", y="Grade", hue="Commodity")
plt.xticks(rotation=45)
plt.title("State-wise Trade Value by Commodity")
plt.tight_layout()
plt.savefig("statewise_commodity_analysis.png")
plt.show()

# Subset of relevant columns for pairplot
pairplot_data = df[['Min_x0020_Price', 'Max_x0020_Price', 'Modal_x0020_Price', 'Predicted Trade Value', 'State', 'Commodity']]

# Create the pairplot
sns.set(style="ticks")
pair_plot = sns.pairplot(pairplot_data, hue='Commodity', diag_kind='kde', corner=True)
pair_plot.fig.suptitle("State-wise & Commodity-wise Mandi Data Relationships", y=1.02)

# Save it
plt.savefig("mandi_pairplot.png")
plt.show()

# --- EXPORT HTML DASHBOARD ---
html = f"""
<html>
<head>
    <title>Mandi Data Analytics</title>
    <style>
        body {{ font-family: Arial; margin: 30px; background-color: #f9f9f9; }}
        h1 {{ color: darkgreen; }}
        .img-container {{ margin-top: 30px; }}
        select {{ padding: 5px; margin-top: 10px; }}
    </style>
    <script>
        function filterTable() {{
            var inputState = document.getElementById("stateSelect").value;
            var inputCommodity = document.getElementById("commoditySelect").value;
            var rows = document.querySelectorAll("#data-table tbody tr");
            rows.forEach(row => {{
                var state = row.getAttribute("data-state");
                var commodity = row.getAttribute("data-commodity");
                row.style.display = (inputState === "All" || state === inputState) &&
                                   (inputCommodity === "All" || commodity === inputCommodity) ? "" : "none";
            }});
        }}
    </script>
</head>
<body>
    <h1>Mandi Trade Analytics Dashboard</h1>
    <p>Select filters to interact with data:</p>
    <label>State:
        <select id="stateSelect" onchange="filterTable()">
            <option>All</option>
            {''.join([f'<option>{s}</option>' for s in sorted(df['State'].unique())])}
        </select>
    </label>
    <label>Commodity:
        <select id="commoditySelect" onchange="filterTable()">
            <option>All</option>
            {''.join([f'<option>{c}</option>' for c in sorted(df['Commodity'].unique())])}
        </select>
    </label>
    <div class="img-container">
        <img src="statewise_commodity_analysis.png" width="100%" />
    </div>



    <div class="img-container">
    <h2>Model Comparison (MSE)</h2>
    <img src="model_comparison.png" width="80%" />
    </div>


    <h2>Detailed Data Table</h2>
    <table border="1" id="data-table" cellpadding="5" cellspacing="0">
        <thead>
            <tr><th>State</th><th>Commodity</th><th>Mandi</th><th>Trade Value</th><th>Predicted</th></tr>
        </thead>
        <tbody>
            {''.join([f'<tr data-state="{row.State}" data-commodity="{row.Commodity}"><td>{row.State}</td><td>{row.Commodity}</td><td>{row.Variety}</td><td>{row["Modal_x0020_Price"]}</td><td>{row["Predicted Trade Value"]:.2f}</td></tr>' for _, row in df.iterrows()])}
        </tbody>
    </table>
</body>
</html>
"""

with open(OUTPUT_HTML, "w") as f:
    f.write(html)



with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML dashboard generated at: {OUTPUT_HTML}")
