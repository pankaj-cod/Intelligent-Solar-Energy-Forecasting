# Intelligent Solar Energy Forecasting

A machine-learning pipeline to **forecast solar PV AC power output** using **plant generation data + weather sensor data**. The current implementation is notebook-based and trains **plant-specific** forecasting models (Plant 1 and Plant 2).

## What this project does

This project builds a supervised regression model to predict:

- **Target:** `AC_POWER`

using a combination of:

- **Weather features** (`AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`)
- **Time features** derived from `DATE_TIME` (hour/day/month)
- **Autoregressive features** from past `AC_POWER` values (lags + rolling mean)

The workflow is implemented in:
- `notebooks/GenAI_Project (2).ipynb`

## Data used

The notebook uses four CSV files:

### Generation data
- `Plant_1_Generation_Data.csv`
- `Plant_2_Generation_Data.csv`

Observed columns for generation (`p1_gen.info()` in the notebook):
- `DATE_TIME` (string initially, parsed to datetime)
- `PLANT_ID` (int)
- `SOURCE_KEY` (string/categorical)
- `DC_POWER` (float)
- `AC_POWER` (float) ← **prediction target**
- `DAILY_YIELD` (float)
- `TOTAL_YIELD` (float)

### Weather sensor data
- `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

Observed columns for weather (`p1_weather.info()` in the notebook):
- `DATE_TIME` (string initially, parsed to datetime)
- `PLANT_ID` (int)
- `SOURCE_KEY` (string/categorical)
- `AMBIENT_TEMPERATURE` (float)
- `MODULE_TEMPERATURE` (float)
- `IRRADIATION` (float)

## Methodology (as implemented in the notebook)

### 1) Parsing timestamps
Both generation and weather data parse `DATE_TIME` with:

- `pd.to_datetime(..., errors="coerce")`

### 2) Merge generation + weather
The datasets are merged using an inner join on the timestamp:

- `pd.merge(gen_df, weather_df, on="DATE_TIME", how="inner")`

This produces a combined table with generation fields + weather sensor fields aligned by time.

### 3) Sort by time
The merged dataset is sorted by `DATE_TIME` (time order matters for lag features and forecasting):

- `df = df.sort_values("DATE_TIME").reset_index(drop=True)`

### 4) Feature engineering

#### Time-based features
From `DATE_TIME`:
- `hour = DATE_TIME.dt.hour`
- `day = DATE_TIME.dt.day`
- `month = DATE_TIME.dt.month`

#### Lag features (past AC power)
The forecasting pipeline creates:
- `ac_power_prev_1 = AC_POWER.shift(1)`
- `ac_power_prev_2 = AC_POWER.shift(2)`
- `ac_power_prev_24 = AC_POWER.shift(24)`

These capture short-term momentum and daily seasonality.

#### Rolling statistics
- `ac_power_roll_3 = AC_POWER.rolling(3).mean()`

This smooths very short-term noise and provides a local trend signal.

#### Handling missing values created by lags/rolling
Rows with missing lag values are dropped:
- `df = df.dropna().reset_index(drop=True)`

### 5) Avoiding leakage / dropping unused columns
To prevent leakage and avoid identifiers, the notebook removes fields that would make the prediction too direct or are not intended as model inputs.

Dropped from features (with `errors="ignore"`):
- Target and timestamp: `DATE_TIME`, `AC_POWER`
- Leakage-prone generation outputs: `DC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`
- Identifiers: `PLANT_ID`, `SOURCE_KEY` (including `_x` and `_y` variants after merge)

### 6) Train/test split (time-aware)
The notebook uses an 80/20 split without shuffling:
- train: earliest 80% of rows
- test: latest 20% of rows

### 7) Model
Model used:

- `RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)`

### 8) Metrics
The notebook evaluates:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Results (from the notebook output)

The notebook trains a separate model for each plant via `run_forecast_pipeline(...)` and prints:

- **Plant 1**
  - MAE: **8.440196041108372**
  - RMSE: **26.341842799727328**
  - R²: **0.994819099004392**

- **Plant 2**
  - MAE: **10.240416179314803**
  - RMSE: **24.514383098095372**
  - R²: **0.9925096270691259**

Notebook note:
- The forecasting model is plant-sensitive and does not fully generalize across locations.

## Saved models

The notebook exports trained models using joblib:

- `model_plant1.pkl`
- `model_plant2.pkl`

## Dependencies

From `requirements.txt`:
- streamlit
- pandas
- scikit-learn
- joblib

## License

Apache License 2.0
