import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_campus_energy_dataset(output_path="campus_energy_data.csv", n_days=730):
    """
    Generates 2 years of realistic hourly campus load (kWh) with:
    - Diurnal work/night cycle
    - Weekly weekend vs weekday load differences
    - Weather impact (temperature-driven HVAC load)
    - Academic calendar (Semester, Exam weeks, Summer/Winter breaks)
    - Stochastic noise & occasional anomalous peaks
    """
    np.random.seed(42)
    start_date = datetime(2024, 1, 1, 0, 0)
    total_hours = n_days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    df = pd.DataFrame({"timestamp": timestamps})
    
    # 1. Temporal signals
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    # 2. Campus calendar simulation
    # Exam weeks: Late April (days 110-125) and Late November (days 325-340)
    df["is_exam_week"] = (
        df["dayofyear"].between(110, 125) | df["dayofyear"].between(325, 340)
    ).astype(int)
    
    # Vacation periods: June-July (Summer) & late Dec (Winter)
    df["is_vacation"] = (
        df["month"].isin([6, 7]) | ((df["month"] == 12) & (df["timestamp"].dt.day > 20))
    ).astype(int)
    
    # 3. Weather simulation (Northern/Eastern India subtropical profile)
    # Seasonal temperature curve + daily diurnal heat swing
    seasonal_temp = 25 - 12 * np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    daily_temp_swing = 6 * np.sin(2 * np.pi * (df["hour"] - 9) / 24)
    temp_noise = np.random.normal(0, 1.5, total_hours)
    df["temperature_c"] = seasonal_temp + daily_temp_swing + temp_noise
    
    # Humidity (inversely correlated with heat + monsoon spikes in July-August)
    humidity_base = 60 - 0.8 * (df["temperature_c"] - 25)
    monsoon_boost = np.where(df["month"].isin([7, 8, 9]), 25, 0)
    df["humidity_pct"] = np.clip(humidity_base + monsoon_boost + np.random.normal(0, 5, total_hours), 20, 98)

    # 4. Base load computation (kWh)
    # Baseline campus baseload (hostels, servers, security lights): 120 kWh
    base_load = 120.0
    
    # Academic diurnal cycle: peaks between 09:00 and 17:00
    diurnal_profile = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 14)) ** 2
    academic_demand = 350.0 * diurnal_profile * (1 - 0.75 * df["is_weekend"]) * (1 - 0.65 * df["is_vacation"])
    
    # Exam surge: night library & computer lab study load (20:00 to 03:00)
    night_exam_surge = 80.0 * df["is_exam_week"] * df["hour"].isin([20, 21, 22, 23, 0, 1, 2, 3]).astype(int)
    
    # HVAC cooling load: heavy surge when temperature > 28°C
    cooling_load = np.maximum(0, df["temperature_c"] - 28) * 18.5 * (1 - 0.5 * df["is_vacation"])
    
    # Sensor noise
    random_noise = np.random.normal(0, 12.0, total_hours)
    
    # Anomalous power spikes (simulating equipment left on or compressor faults)
    anomalies = (np.random.rand(total_hours) > 0.995).astype(int) * np.random.uniform(90, 180, total_hours)

    df["energy_consumption_kwh"] = np.round(
        base_load + academic_demand + night_exam_surge + cooling_load + random_noise + anomalies, 2
    )
    df["energy_consumption_kwh"] = np.maximum(45.0, df["energy_consumption_kwh"]) # Min physical safety floor

    # Retain primary columns
    export_cols = [
        "timestamp", "temperature_c", "humidity_pct", "is_weekend",
        "is_exam_week", "is_vacation", "energy_consumption_kwh"
    ]
    df[export_cols].to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' ({len(df):,} hourly records).")

if __name__ == "__main__":
    generate_campus_energy_dataset()