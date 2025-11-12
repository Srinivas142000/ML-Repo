# -*- coding: utf-8 -*-
"""Time Series Forecasting - Kaggle Sticker Sales"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================================
# PART 1: SINGLE TIME SERIES TRAINING (Learning/Validation)
# ============================================================================

# Load data
df = pd.read_csv("ts_train.csv")
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded {len(df)} training rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Filter for single time series (Kaggle, Canada, Discount Stickers)
single_series = df[
    (df['country'] == 'Canada') & 
    (df['store'] == 'Discount Stickers') & 
    (df['product'] == 'Kaggle')
].copy().sort_values('date')

# Create features
single_series['year'] = single_series['date'].dt.year
single_series['month'] = single_series['date'].dt.month
single_series['day'] = single_series['date'].dt.day
single_series['day_of_week'] = single_series['date'].dt.dayofweek
single_series['quarter'] = single_series['date'].dt.quarter

# Lag features
single_series['lag_1'] = single_series['num_sold'].shift(1)
single_series['lag_7'] = single_series['num_sold'].shift(7)
single_series['lag_14'] = single_series['num_sold'].shift(14)
single_series['lag_30'] = single_series['num_sold'].shift(30)

# Rolling features
single_series['rolling_mean_7'] = single_series['num_sold'].rolling(7).mean()
single_series['rolling_mean_30'] = single_series['num_sold'].rolling(30).mean()
single_series['rolling_std_7'] = single_series['num_sold'].rolling(7).std()

# Drop NaN
single_series = single_series.dropna()

# Split chronologically
train = single_series[single_series['year'] < 2016]
test = single_series[single_series['year'] == 2016]

# Define features
feature_cols = ['year', 'month', 'day', 'day_of_week', 'quarter',
                'lag_1', 'lag_7', 'lag_14', 'lag_30',
                'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']

X_train = train[feature_cols]
y_train = train['num_sold']
X_test = test[feature_cols]
y_test = test['num_sold']

# Train model
print("\nTraining single-series model...")
model_single = RandomForestRegressor(n_estimators=100, random_state=42)
model_single.fit(X_train, y_train)

# Evaluate
y_pred = model_single.predict(X_test)
rmse_single = np.sqrt(mean_squared_error(y_test, y_pred))
mae_single = mean_absolute_error(y_test, y_pred)

print(f"Single Series Performance:")
print(f"  RMSE: {rmse_single:.2f}")
print(f"  MAE: {mae_single:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(test['date'], y_test, label='Actual', alpha=0.7)
plt.plot(test['date'], y_pred, label='Predicted', alpha=0.7)
plt.title('Single Series: Actual vs Predicted (2016)')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# PART 2: GLOBAL MODEL TRAINING (All 90 Time Series)
# ============================================================================

# Reload and prepare full dataset
df_full = pd.read_csv("ts_train.csv")
df_full['date'] = pd.to_datetime(df_full['date'])

# Fill Holographic Goose NaNs with 0 (product didn't exist yet)
df_full['num_sold'] = df_full['num_sold'].fillna(0)

# Sort data
df_full = df_full.sort_values(['country', 'store', 'product', 'date'])

# Create features for all series
df_full['year'] = df_full['date'].dt.year
df_full['month'] = df_full['date'].dt.month
df_full['day'] = df_full['date'].dt.day
df_full['day_of_week'] = df_full['date'].dt.dayofweek
df_full['quarter'] = df_full['date'].dt.quarter

# Lag features (grouped by each time series)
df_full['lag_1'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].shift(1)
df_full['lag_7'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].shift(7)
df_full['lag_14'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].shift(14)
df_full['lag_30'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].shift(30)

# Rolling features (grouped by each time series)
df_full['rolling_mean_7'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df_full['rolling_mean_30'] = df_full.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

# Drop NaN
df_full = df_full.dropna()

# One-hot encode
df_encoded = pd.get_dummies(df_full, columns=['country', 'store', 'product'], drop_first=True)

# Split train/validation
train_global = df_encoded[df_encoded['year'] < 2016]
test_global = df_encoded[df_encoded['year'] == 2016]

# Define features
feature_cols_global = ['year', 'month', 'day', 'day_of_week', 'quarter',
                       'lag_1', 'lag_7', 'lag_14', 'lag_30',
                       'rolling_mean_7', 'rolling_mean_30']
feature_cols_global += [col for col in df_encoded.columns if col.startswith(('country_', 'store_', 'product_'))]

# Train global model
print(f"\nTraining global model on {len(train_global)} samples...")
model_global = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_global.fit(train_global[feature_cols_global], train_global['num_sold'])

# Evaluate
y_pred_global = model_global.predict(test_global[feature_cols_global])
rmse_global = np.sqrt(mean_squared_error(test_global['num_sold'], y_pred_global))
mae_global = mean_absolute_error(test_global['num_sold'], y_pred_global))

print(f"\nGlobal Model Performance:")
print(f"  RMSE: {rmse_global:.2f}")
print(f"  MAE: {mae_global:.2f}")

# Scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(test_global['num_sold'], y_pred_global, alpha=0.1, s=1)
plt.plot([0, test_global['num_sold'].max()], [0, test_global['num_sold'].max()], 'r--', label='Perfect')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Global Model: All 90 Series')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# PART 3: FINAL MODEL & KAGGLE SUBMISSION (2017-2019 Predictions)
# ============================================================================

# Load test data
train_raw = pd.read_csv("ts_train.csv")
test_raw = pd.read_csv("ts_test.csv")

train_raw['date'] = pd.to_datetime(train_raw['date'])
test_raw['date'] = pd.to_datetime(test_raw['date'])

# Mark train vs test
train_raw['is_test'] = False
test_raw['is_test'] = True
test_raw['num_sold'] = np.nan

# Combine
df_all = pd.concat([train_raw, test_raw], ignore_index=True)
df_all = df_all.sort_values(['country', 'store', 'product', 'date'])

# Fill training NaNs
df_all.loc[~df_all['is_test'], 'num_sold'] = df_all.loc[~df_all['is_test'], 'num_sold'].fillna(0)

print(f"\nCombined dataset: {len(df_all)} rows")
print(f"Train: {(~df_all['is_test']).sum()} rows")
print(f"Test: {df_all['is_test'].sum()} rows")

# Create features
df_all['year'] = df_all['date'].dt.year
df_all['month'] = df_all['date'].dt.month
df_all['day'] = df_all['date'].dt.day
df_all['day_of_week'] = df_all['date'].dt.dayofweek
df_all['quarter'] = df_all['date'].dt.quarter

# Initial lag features
df_all['lag_1'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].shift(1)
df_all['lag_7'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].shift(7)
df_all['lag_14'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].shift(14)
df_all['lag_30'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].shift(30)

df_all['rolling_mean_7'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df_all['rolling_mean_30'] = df_all.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

# Encode
df_all_encoded = pd.get_dummies(df_all, columns=['country', 'store', 'product'], drop_first=True)

# Train final model on ALL 2010-2016 data
train_final = df_all_encoded[~df_all_encoded['is_test']].dropna()

feature_cols_final = ['year', 'month', 'day', 'day_of_week', 'quarter',
                      'lag_1', 'lag_7', 'lag_14', 'lag_30',
                      'rolling_mean_7', 'rolling_mean_30']
feature_cols_final += [col for col in df_all_encoded.columns if col.startswith(('country_', 'store_', 'product_'))]

print(f"\nTraining final model on {len(train_final)} samples...")
model_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_final.fit(train_final[feature_cols_final], train_final['num_sold'])

# ============================================================================
# DAY-BY-DAY RECURSIVE PREDICTION (Efficient Method)
# ============================================================================

test_dates = sorted(df_all[df_all['is_test']]['date'].unique())
print(f"\nPredicting {len(test_dates)} days for 90 time series...")

for i, current_date in enumerate(test_dates):
    if i % 100 == 0:
        print(f"Progress: Day {i}/{len(test_dates)}")
    
    # Get all rows for this date
    mask_date = (df_all['date'] == current_date) & (df_all['is_test'] == True)
    test_rows = df_all[mask_date]
    
    if len(test_rows) == 0:
        continue
    
    # Get features
    X_pred = df_all_encoded.loc[test_rows.index, feature_cols_final].fillna(0)
    
    # Predict for all 90 series at once
    predictions = model_final.predict(X_pred)
    
    # Store predictions in original dataframe
    df_all.loc[test_rows.index, 'num_sold'] = predictions
    
    # Update lag features for ALL series (more efficient than per-series)
    for (country, store, product), group in df_all.groupby(['country', 'store', 'product']):
        mask_series = (
            (df_all['country'] == country) &
            (df_all['store'] == store) &
            (df_all['product'] == product)
        )
        
        series_data = df_all[mask_series].sort_values('date')
        
        # Recalculate lag features
        df_all.loc[mask_series, 'lag_1'] = series_data['num_sold'].shift(1).values
        df_all.loc[mask_series, 'lag_7'] = series_data['num_sold'].shift(7).values
        df_all.loc[mask_series, 'lag_14'] = series_data['num_sold'].shift(14).values
        df_all.loc[mask_series, 'lag_30'] = series_data['num_sold'].shift(30).values
        
        # Update rolling features
        df_all.loc[mask_series, 'rolling_mean_7'] = series_data['num_sold'].rolling(7, min_periods=1).mean().values
        df_all.loc[mask_series, 'rolling_mean_30'] = series_data['num_sold'].rolling(30, min_periods=1).mean().values
        
        # Update encoded dataframe
        df_all_encoded.loc[mask_series, 'lag_1'] = df_all.loc[mask_series, 'lag_1']
        df_all_encoded.loc[mask_series, 'lag_7'] = df_all.loc[mask_series, 'lag_7']
        df_all_encoded.loc[mask_series, 'lag_14'] = df_all.loc[mask_series, 'lag_14']
        df_all_encoded.loc[mask_series, 'lag_30'] = df_all.loc[mask_series, 'lag_30']
        df_all_encoded.loc[mask_series, 'rolling_mean_7'] = df_all.loc[mask_series, 'rolling_mean_7']
        df_all_encoded.loc[mask_series, 'rolling_mean_30'] = df_all.loc[mask_series, 'rolling_mean_30']

print("\n✅ Predictions complete!")

# ============================================================================
# CREATE KAGGLE SUBMISSION FILE
# ============================================================================

# Get test predictions
test_predictions = df_all[df_all['is_test']].copy()

# Create submission
submission = pd.DataFrame({
    'id': test_predictions['id'],
    'num_sold': test_predictions['num_sold']
})

# Save
submission.to_csv('submission.csv', index=False)

print(f"\n✅ Submission file created!")
print(f"Rows: {len(submission)}")
print(f"Prediction range: {submission['num_sold'].min():.2f} to {submission['num_sold'].max():.2f}")
print(f"Mean prediction: {submission['num_sold'].mean():.2f}")

# Sanity check - show Kaggle predictions
kaggle_test = test_predictions[
    (test_predictions['country'] == 'Canada') &
    (test_predictions['store'] == 'Discount Stickers') &
    (test_predictions['product'] == 'Kaggle')
].head(10)

print("\nSample predictions (Kaggle, Canada):")
for _, row in kaggle_test.iterrows():
    print(f"{row['date'].date()}: {row['num_sold']:.2f}")

print("\n✅ Done! Check submission.csv")