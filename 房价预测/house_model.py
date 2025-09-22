import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

CSV_PATH = r"F:\机器学习\房价预测\house_prices.csv"
SAVE_MODEL_PATH = r"F:\机器学习\房价预测\house_price_pipeline.pkl"
TOP_K_LOCATIONS = 50
N_ESTIMATORS = 500
RANDOM_STATE = 42

print("读取 CSV：", CSV_PATH)
df = pd.read_csv(CSV_PATH)

features = [
    'location','Carpet Area','Status','Floor',
    'Transaction','Furnishing','Bathroom','Balcony',
    'Car Parking','Ownership'
]
features = [c for c in features if c in df.columns]
target = 'Price (in rupees)'

data = df[features + [target]].copy()

def extract_num_generic(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(',', '')
    m = re.search(r'(\d+\.?\d*)', s)
    return float(m.group(1)) if m else np.nan

def parse_floor(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if 'ground' in s:
        return 0.0
    if 'basement' in s or 'lower basement' in s:
        return -1.0
    m = re.search(r'(\d+)', s)
    return float(m.group(1)) if m else np.nan

if 'Carpet Area' in data.columns:
    data['Carpet Area'] = data['Carpet Area'].apply(extract_num_generic)
if 'Floor' in data.columns:
    data['Floor'] = data['Floor'].apply(parse_floor)
if 'Car Parking' in data.columns:
    data['Car Parking'] = data['Car Parking'].apply(extract_num_generic)

for col in ['Bathroom','Balcony']:
    if col in data.columns:
        coerced = pd.to_numeric(data[col], errors='coerce')
        mask_na = coerced.isna() & data[col].notna()
        if mask_na.any():
            extracted = data.loc[mask_na, col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
            coerced.loc[mask_na] = pd.to_numeric(extracted, errors='coerce')
        data[col] = coerced

num_cols = [c for c in ['Carpet Area','Floor','Bathroom','Balcony','Car Parking'] if c in data.columns]
for c in num_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')

q_low = data[target].quantile(0.01)
q_high = data[target].quantile(0.99)
data = data[(data[target] >= q_low) & (data[target] <= q_high)].copy()
print(f"去掉异常值后样本数：{len(data)}")

if 'location' in data.columns:
    top_locs = data['location'].value_counts().nlargest(TOP_K_LOCATIONS).index.tolist()
    data['location'] = data['location'].where(data['location'].isin(top_locs), other='Other')

cat_cols = [c for c in ['location','Status','Transaction','Furnishing','Ownership'] if c in data.columns]
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
pipe = Pipeline([('preprocess', preprocessor), ('model', model)])

X = data[features]
y = np.log1p(data[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print("训练样本：", X_train.shape, "测试样本：", X_test.shape)
pipe.fit(X_train, y_train)

y_pred_log = pipe.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

print("MAE:", mean_absolute_error(y_test_original, y_pred))
print("R² :", r2_score(y_test_original, y_pred))

joblib.dump(pipe, SAVE_MODEL_PATH)

plt.figure(figsize=(6,6))
plt.scatter(y_test_original, y_pred, alpha=0.3)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("真实价格")
plt.ylabel("预测价格")
plt.title("预测 vs 真实（对数坐标）")
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         "r--", lw=2)
plt.tight_layout()
plt.savefig(r'F:\机器学习\房价预测\预测_vs_真实_改进版.png')
plt.show()
