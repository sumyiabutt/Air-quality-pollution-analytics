"""
Air Quality & Pollution Analytics (Lahore) - Data preprocessing script
Big Data Analytics - University of Central Punjab

Run:  python preprocess.py
Needs: pandas, numpy   (pip install pandas numpy)
Input : ./Original_Data/*.csv
Output: ./Preprocessed_Data/*.csv  and  cleaning_log.json
"""
import json, os
import numpy as np
import pandas as pd

IN, OUT = "Original_Data", "Preprocessed_Data"
os.makedirs(OUT, exist_ok=True)
log = {}


def interp(df, cols, limit):
    """Linear interpolation, only for short gaps (<= limit rows)."""
    before = df[cols].isna().sum().sum()
    df[cols] = df[cols].interpolate(method="linear", limit=limit, limit_area="inside")
    return int(before - df[cols].isna().sum().sum())


def aqi_category(v):
    """US EPA AQI category label (0-500 scale)."""
    if pd.isna(v): return np.nan
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive Groups"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"


# ---------------------------------------------------------------- File 1
f1 = pd.read_csv(f"{IN}/1_daily_air_quality_2022_2026.csv", parse_dates=["date"])
L = {"rows_before": len(f1), "cols": f1.shape[1], "missing_before": f1.isna().sum().to_dict()}
empty = f1.drop(columns="date").isna().all(axis=1)
L["fully_empty_rows_removed"] = int(empty.sum())
f1 = f1[~empty].reset_index(drop=True)
L["duplicates"] = int(f1.duplicated().sum())
L["us_aqi_over_300"] = int((f1.us_aqi > 300).sum())
L["pm2_5_over_250"] = int((f1.pm2_5 > 250).sum())
L["missing_after"] = f1.isna().sum().to_dict()
L["rows_after"] = len(f1)
f1["us_aqi_category"] = f1.us_aqi.map(aqi_category)
f1.to_csv(f"{OUT}/1_daily_air_quality_clean.csv", index=False)
log["file1"] = L

# ---------------------------------------------------------------- File 2
f2 = pd.read_csv(f"{IN}/2_lahore_city_info.csv", dtype={"admin1": str, "admin2": str})
f2.to_csv(f"{OUT}/2_lahore_city_info_clean.csv", index=False)
log["file2"] = {"rows_before": len(f2), "rows_after": len(f2), "cols": f2.shape[1]}

# ---------------------------------------------------------------- File 3
f3 = pd.read_csv(f"{IN}/3_lahore_population_vehicles_2019_2023.csv", encoding="utf-8-sig", dtype=str)
for c in ["population", "vehicles"]:
    f3[c] = f3[c].str.replace(",", "", regex=False).astype("int64")
f3["year"] = f3["year"].astype("int64")
f3["vehicles_per_100_people"] = (f3.vehicles / f3.population * 100).round(2)
f3.to_csv(f"{OUT}/3_lahore_population_vehicles_clean.csv", index=False)
log["file3"] = {"rows_before": len(f3), "rows_after": len(f3), "cols_before": 3, "cols_after": f3.shape[1]}

# ---------------------------------------------------------------- File 4
f4 = pd.read_csv(f"{IN}/4_daily_weather_pm25_2019_2023.csv", encoding="utf-8-sig")
L = {"rows_before": len(f4), "cols_before": f4.shape[1], "missing_before": {k: v for k, v in f4.isna().sum().items() if v}}
f4["date"] = pd.to_datetime(f4["date"], format="%d-%m-%y")
f4 = f4.rename(columns={"aqi_pm2.5": "aqi_pm2_5"})
L["duplicates"] = int(f4.duplicated().sum())
inv = {}
def bad(mask, cols, name):
    inv[name] = int(mask.sum())
    f4.loc[mask, cols] = np.nan
# zeros used as placeholders in "min" columns / impossible extremes -> NaN
bad(f4.min_temp_f == 0, "min_temp_f", "min_temp_zero")
bad(f4.min_dew_point_f == 0, "min_dew_point_f", "min_dew_point_zero")
bad(f4.min_humidity_percent == 0, "min_humidity_percent", "min_humidity_zero")
bad(f4.min_pressure_in == 0, "min_pressure_in", "min_pressure_zero")
bad(f4.max_temp_f > 125, "max_temp_f", "max_temp_over_125F")
bad(f4.max_dew_point_f > 100, "max_dew_point_f", "max_dew_point_over_100F")
bad(f4.max_wind_speed_mph > 90, "max_wind_speed_mph", "max_wind_over_90mph")
bad((f4.max_wind_speed_mph == 0) & (f4.avg_wind_speed_mph == 0), ["max_wind_speed_mph", "avg_wind_speed_mph"], "wind_all_zero_days")
bad(f4.max_pressure_in > 31, "max_pressure_in", "max_pressure_over_31in")
bad(f4.avg_pressure_in < 27, "avg_pressure_in", "avg_pressure_below_27in")
L["invalid_values_set_to_missing"] = inv
# unit conversion to metric (same units as the hourly weather file)
for k in ("max", "avg", "min"):
    f4[f"{k}_temp_c"] = ((f4[f"{k}_temp_f"] - 32) * 5 / 9).round(1)
    f4[f"{k}_dew_point_c"] = ((f4[f"{k}_dew_point_f"] - 32) * 5 / 9).round(1)
    f4[f"{k}_wind_speed_kmh"] = (f4[f"{k}_wind_speed_mph"] * 1.609344).round(1)
    f4[f"{k}_pressure_hpa"] = (f4[f"{k}_pressure_in"] * 33.8639).round(1)
    f4 = f4.rename(columns={f"{k}_humidity_percent": f"{k}_humidity_pct"})
f4 = f4.drop(columns=[c for c in f4.columns if c.endswith(("_f", "_mph", "_in"))])
num = [c for c in f4.columns if c != "date"]
L["short_gaps_filled_by_interpolation"] = interp(f4, num, 3)
L["missing_after"] = {k: v for k, v in f4.isna().sum().items() if v}
L["cols_after"] = f4.shape[1]; L["rows_after"] = len(f4)
f4["aqi_category"] = f4.aqi_pm2_5.map(aqi_category)
f4.to_csv(f"{OUT}/4_daily_weather_pm25_clean.csv", index=False)
log["file4"] = L

# ---------------------------------------------------------------- File 5
raw5 = pd.read_csv(f"{IN}/5_hourly_weather_2023_2025.csv", nrows=1)
f5 = pd.read_csv(f"{IN}/5_hourly_weather_2023_2025.csv", skiprows=3)
L = {"rows_before": len(f5), "cols_before": f5.shape[1], "metadata": raw5.iloc[0].to_dict()}
f5 = f5.dropna(axis=1, how="all")
f5.columns = ["timestamp", "temperature_c", "humidity_pct", "wind_speed_kmh"]
f5["timestamp"] = pd.to_datetime(f5["timestamp"])
L["duplicates"] = int(f5.timestamp.duplicated().sum())
L["missing_before"] = int(f5.isna().sum().sum())
L["expected_hours"] = int(len(pd.date_range(f5.timestamp.min(), f5.timestamp.max(), freq="h")))
L["rows_after"] = len(f5); L["cols_after"] = f5.shape[1]
f5.to_csv(f"{OUT}/5_hourly_weather_clean.csv", index=False)
log["file5"] = L

# ---------------------------------------------------------------- File 6
f6 = pd.read_csv(f"{IN}/6_hourly_pollutants_2022_2026.csv", parse_dates=["Timestamp"])
f6 = f6.rename(columns={"Timestamp": "timestamp", "AQI": "aqi_index_1_5", "PM2.5": "pm2_5", "PM10": "pm10",
                        "CO": "co", "NO": "no", "NO2": "no2", "O3": "o3", "SO2": "so2", "NH3": "nh3"})
pol = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
L = {"rows_before": len(f6), "cols_before": f6.shape[1]}
L["exact_duplicate_rows"] = int(f6.duplicated().sum())
L["duplicate_timestamps"] = int(f6.timestamp.duplicated().sum())
L["sentinel_-9999"] = {c: int((f6[c] == -9999).sum()) for c in pol if (f6[c] == -9999).any()}
L["non_integer_aqi_rows"] = int((f6.aqi_index_1_5 % 1 != 0).sum())
L["aqi_share_level_5_pct"] = round(float((f6.aqi_index_1_5 == 5).mean() * 100), 1)
L["pm2_5_over_1000"] = int((f6.pm2_5 > 1000).sum())
L["pm10_over_1000"] = int((f6.pm10 > 1000).sum())
L["co_over_20000"] = int((f6.co > 20000).sum())
L["pm2_5_greater_than_pm10"] = int((f6.pm2_5 > f6.pm10).sum())
L["o3_zero_values"] = int((f6.o3 == 0).sum())
f6[pol] = f6[pol].replace(-9999, np.nan)
f6 = f6.drop_duplicates()
same_ts = int(f6.timestamp.duplicated().sum())
L["duplicate_timestamps_with_different_values"] = same_ts
f6 = f6.groupby("timestamp", as_index=False).mean(numeric_only=True)          # average conflicting readings
f6["aqi_index_1_5"] = f6["aqi_index_1_5"].round().astype("float")
f6 = f6.sort_values("timestamp").set_index("timestamp")
full = pd.date_range(f6.index.min(), f6.index.max(), freq="h")
L["expected_hours"] = len(full)
L["missing_hours"] = int(len(full.difference(f6.index)))
f6 = f6.reindex(full); f6.index.name = "timestamp"
miss = pd.Series(full.difference(f6.index[f6.pm2_5.notna()]))
gaps = (pd.Series(f6.index[f6.pm2_5.isna()]))
runs = (gaps.diff() != pd.Timedelta("1h")).cumsum()
L["gap_runs"] = int(runs.nunique()); L["longest_gap_hours"] = int(gaps.groupby(runs).size().max())
L["short_gaps_filled_by_interpolation"] = interp(f6, pol, 3)
L["missing_after"] = {c: int(v) for c, v in f6[pol].isna().sum().items()}
def us_aqi_pm25(x):
    bp = [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
          (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    if pd.isna(x): return np.nan
    x = np.floor(x * 10) / 10
    if x > 500.4: return 500
    for lo, hi, ilo, ihi in bp:
        if lo <= x <= hi: return round((ihi - ilo) / (hi - lo) * (x - lo) + ilo)
    return np.nan
f6["aqi_us_from_pm2_5"] = f6.pm2_5.map(us_aqi_pm25)
f6["aqi_category_us"] = f6.aqi_us_from_pm2_5.map(aqi_category)
f6 = f6.reset_index()
L["rows_after"] = len(f6); L["cols_after"] = f6.shape[1]
f6.to_csv(f"{OUT}/6_hourly_pollutants_clean.csv", index=False)
log["file6"] = L

# ---------------------------------------------------------------- Merged hourly (pollution + weather)
m = f6.merge(f5, on="timestamp", how="inner")
m["year"] = m.timestamp.dt.year; m["month"] = m.timestamp.dt.month
m["hour"] = m.timestamp.dt.hour; m["day_of_week"] = m.timestamp.dt.day_name()
m.to_csv(f"{OUT}/7_merged_hourly_pollution_weather.csv", index=False)
log["merged"] = {"rows": len(m), "cols": m.shape[1], "from": str(m.timestamp.min()), "to": str(m.timestamp.max()),
                 "rows_with_pm2_5": int(m.pm2_5.notna().sum())}

# ---------------------------------------------------------------- MASTER DAILY FILE (all sources, key = date)
# Step 1: select only the relevant columns from every file
h6 = f6.copy(); h6["date"] = h6.timestamp.dt.normalize()
d6 = h6.groupby("date").agg(
    hours_available=("pm2_5", "count"), pm2_5_mean=("pm2_5", "mean"), pm2_5_max=("pm2_5", "max"),
    pm10_mean=("pm10", "mean"), co_mean=("co", "mean"), no2_mean=("no2", "mean"),
    so2_mean=("so2", "mean"), o3_mean=("o3", "mean"), nh3_mean=("nh3", "mean"),
    aqi_us_mean=("aqi_us_from_pm2_5", "mean")).reset_index()
d6.loc[d6.hours_available == 0, [c for c in d6.columns if c not in ("date", "hours_available")]] = np.nan
d6 = d6.add_prefix("h6_").rename(columns={"h6_date": "date"})
w5 = f5.copy(); w5["date"] = w5.timestamp.dt.normalize()
d5 = w5.groupby("date").agg(temp_mean_c=("temperature_c", "mean"), temp_max_c=("temperature_c", "max"),
    temp_min_c=("temperature_c", "min"), humidity_mean_pct=("humidity_pct", "mean"),
    wind_mean_kmh=("wind_speed_kmh", "mean")).reset_index()
d5 = d5.round(2).add_prefix("h5_").rename(columns={"h5_date": "date"})
d1 = f1[["date", "pm2_5", "pm10", "us_aqi"]].rename(columns={"pm2_5": "f1_pm2_5", "pm10": "f1_pm10", "us_aqi": "f1_us_aqi"})
d4 = f4[["date", "aqi_pm2_5", "avg_temp_c", "avg_humidity_pct", "avg_wind_speed_kmh", "avg_pressure_hpa"]]
d4 = d4.rename(columns={c: "f4_" + c for c in d4.columns if c != "date"})
# Step 2: merge on the common key (date); yearly population/vehicles joined on year
master = d4.merge(d1, on="date", how="outer").merge(d6, on="date", how="outer").merge(d5, on="date", how="outer")
master = master.sort_values("date").reset_index(drop=True)
master["year"] = master.date.dt.year
master = master.merge(f3[["year", "population", "vehicles", "vehicles_per_100_people"]], on="year", how="left")
master = master.round(2)
master.to_csv(f"{OUT}/8_master_daily_all_sources.csv", index=False)
# Step 3: verification
ov = master.dropna(subset=["f1_pm2_5", "h6_pm2_5_mean"])
ov2 = master.dropna(subset=["f4_aqi_pm2_5", "h6_aqi_us_mean"])
ov3 = master.dropna(subset=["f4_avg_temp_c", "h5_temp_mean_c"])
log["master"] = {
    "rows": len(master), "cols": master.shape[1], "from": str(master.date.min().date()), "to": str(master.date.max().date()),
    "duplicate_dates": int(master.date.duplicated().sum()),
    "days_expected": int(len(pd.date_range(master.date.min(), master.date.max()))),
    "coverage_days": {"file1_daily_air_quality": int(master.f1_pm2_5.notna().sum()),
                      "file4_daily_weather_pm25": int(master.f4_aqi_pm2_5.notna().sum()),
                      "file6_hourly_pollutants": int(master.h6_pm2_5_mean.notna().sum()),
                      "file5_hourly_weather": int(master.h5_temp_mean_c.notna().sum())},
    "check_f1_vs_f6_pm2_5": {"days": len(ov), "correlation": round(float(ov.f1_pm2_5.corr(ov.h6_pm2_5_mean)), 3)},
    "check_f4_aqi_vs_f6_aqi": {"days": len(ov2), "correlation": round(float(ov2.f4_aqi_pm2_5.corr(ov2.h6_aqi_us_mean)), 3)},
    "check_f4_vs_f5_temperature": {"days": len(ov3), "correlation": round(float(ov3.f4_avg_temp_c.corr(ov3.h5_temp_mean_c)), 3)},
    "columns": list(master.columns)}

# ---------------------------------------------------------------- BIAS AND LABEL CHECKS
mn = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
winter = [11, 12, 1, 2]
bias = {}
# (a) month coverage = how often each calendar month is represented
for key, df_, col in [("file1", f1, "date"), ("file4", f4, "date")]:
    c = df_[col].dt.month.value_counts().sort_index()
    bias[f"{key}_rows_by_month"] = {mn[k]: int(v) for k, v in c.items()}
    bias[f"{key}_winter_share_pct"] = round(float(c[c.index.isin(winter)].sum() / c.sum() * 100), 1)
c = f6.dropna(subset=["pm2_5"]).timestamp.dt.month.value_counts().sort_index()
bias["file6_valid_hours_by_month"] = {mn[k]: int(v) for k, v in c.items()}
bias["file6_winter_share_pct"] = round(float(c[c.index.isin(winter)].sum() / c.sum() * 100), 1)
# (b) where are the missing values? (original files)
raw6 = pd.read_csv(f"{IN}/6_hourly_pollutants_2022_2026.csv", parse_dates=["Timestamp"]).drop_duplicates("Timestamp")
miss6 = pd.date_range(raw6.Timestamp.min(), raw6.Timestamp.max(), freq="h").difference(pd.DatetimeIndex(raw6.Timestamp))
bias["file6_missing_hours_by_year"] = {int(k): int(v) for k, v in pd.Series(miss6.year).value_counts().sort_index().items()}
bias["file6_missing_hours_by_month"] = {mn[k]: int(v) for k, v in pd.Series(miss6.month).value_counts().sort_index().items()}
r4 = pd.read_csv(f"{IN}/4_daily_weather_pm25_2019_2023.csv", encoding="utf-8-sig")
r4["date"] = pd.to_datetime(r4["date"], format="%d-%m-%y")
m4 = r4[r4["aqi_pm2.5"].isna()].date
bias["file4_missing_pm25_by_year"] = {int(k): int(v) for k, v in m4.dt.year.value_counts().sort_index().items()}
bias["file4_missing_pm25_by_month"] = {mn[k]: int(v) for k, v in m4.dt.month.value_counts().sort_index().items()}
# (c) seasonal difference of PM2.5 (shows why seasonal balance matters)
v = f6.dropna(subset=["pm2_5"]).copy(); v["w"] = v.timestamp.dt.month.isin(winter)
bias["pm2_5_mean_winter"] = round(float(v[v.w].pm2_5.mean()), 1)
bias["pm2_5_mean_other_months"] = round(float(v[~v.w].pm2_5.mean()), 1)
# (d) label balance
raw6a = pd.read_csv(f"{IN}/6_hourly_pollutants_2022_2026.csv")
bias["file6_original_aqi_share_pct"] = {str(int(k)): round(float(x), 1) for k, x in (raw6a.AQI.round().value_counts(normalize=True).sort_index() * 100).items()}
cat = f6.aqi_category_us.value_counts(normalize=True) * 100
bias["file6_us_category_share_pct"] = {k: round(float(x), 1) for k, x in cat.items()}
bias["file1_us_category_share_pct"] = {k: round(float(x), 1) for k, x in (f1.us_aqi_category.value_counts(normalize=True) * 100).items()}
# (e) label / validity check: physical ranges after cleaning
rng = {
    "file5_humidity_outside_0_100": int(((f5.humidity_pct < 0) | (f5.humidity_pct > 100)).sum()),
    "file5_temperature_outside_-10_55C": int(((f5.temperature_c < -10) | (f5.temperature_c > 55)).sum()),
    "file5_wind_outside_0_150kmh": int(((f5.wind_speed_kmh < 0) | (f5.wind_speed_kmh > 150)).sum()),
    "file4_temp_outside_-10_55C": int(((f4[["max_temp_c", "avg_temp_c", "min_temp_c"]] < -10) | (f4[["max_temp_c", "avg_temp_c", "min_temp_c"]] > 55)).sum().sum()),
    "file4_humidity_outside_0_100": int(((f4[["max_humidity_pct", "avg_humidity_pct", "min_humidity_pct"]] < 0) | (f4[["max_humidity_pct", "avg_humidity_pct", "min_humidity_pct"]] > 100)).sum().sum()),
    "file4_pressure_outside_900_1100hpa": int(((f4[["max_pressure_hpa", "avg_pressure_hpa", "min_pressure_hpa"]] < 900) | (f4[["max_pressure_hpa", "avg_pressure_hpa", "min_pressure_hpa"]] > 1100)).sum().sum()),
    "file4_wind_above_150kmh": int((f4[["max_wind_speed_kmh", "avg_wind_speed_kmh", "min_wind_speed_kmh"]] > 150).sum().sum()),
    "file6_negative_concentrations": int((f6[pol] < 0).sum().sum()),
    "file1_negative_values": int((f1.drop(columns=["date", "us_aqi_category"]) < 0).sum().sum()),
}
bias["range_checks_after_cleaning"] = rng
bias["distinct_locations_in_data"] = 1
log["bias"] = bias

json.dump(log, open("cleaning_log.json", "w"), indent=2, default=str)
print(json.dumps(log, indent=1, default=str))
