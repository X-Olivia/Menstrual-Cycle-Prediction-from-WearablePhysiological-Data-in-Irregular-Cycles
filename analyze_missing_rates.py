#!/usr/bin/env python3
"""
分析mcPHASES数据集中sleep、自我报告症状、WST和HR的缺失率
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 数据文件路径
data_dir = Path("dataset_physio/mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0")

print("=" * 80)
print("mcPHASES 数据集缺失率分析")
print("=" * 80)

# 1. 自我报告症状 (hormones_and_selfreport.csv)
print("\n【1. 自我报告症状 - hormones_and_selfreport.csv】")
print("-" * 80)
try:
    df_selfreport = pd.read_csv(data_dir / "hormones_and_selfreport.csv")
    print(f"总行数: {len(df_selfreport):,}")
    
    # 自我报告症状字段
    symptom_cols = [
        'flow_volume', 'flow_color', 'appetite', 'exerciselevel',
        'headaches', 'cramps', 'sorebreasts', 'fatigue', 'sleepissue',
        'moodswing', 'stress', 'foodcravings', 'indigestion', 'bloating'
    ]
    
    print("\n自我报告症状字段缺失率:")
    print(f"{'字段名':<20} {'总行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 60)
    
    total_rows = len(df_selfreport)
    for col in symptom_cols:
        if col in df_selfreport.columns:
            missing_count = df_selfreport[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<20} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
        else:
            print(f"{col:<20} {'列不存在':<12}")
    
    # 计算所有症状字段的平均缺失率
    symptom_missing_rates = []
    for col in symptom_cols:
        if col in df_selfreport.columns:
            missing_rate = (df_selfreport[col].isna().sum() / total_rows) * 100
            symptom_missing_rates.append(missing_rate)
    
    if symptom_missing_rates:
        avg_missing_rate = np.mean(symptom_missing_rates)
        print(f"\n自我报告症状平均缺失率: {avg_missing_rate:.2f}%")
        
except Exception as e:
    print(f"错误: {e}")

# 2. 睡眠数据
print("\n【2. 睡眠数据】")
print("-" * 80)

# sleep.csv
print("\n(1) sleep.csv - 睡眠会话记录")
try:
    df_sleep = pd.read_csv(data_dir / "sleep.csv")
    print(f"总行数: {len(df_sleep):,}")
    
    sleep_cols = ['duration', 'minutestofallasleep', 'minutesasleep', 
                  'minutesawake', 'efficiency', 'levels']
    
    print(f"\n{'字段名':<25} {'总行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 65)
    
    total_rows = len(df_sleep)
    for col in sleep_cols:
        if col in df_sleep.columns:
            missing_count = df_sleep[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<25} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
except Exception as e:
    print(f"错误: {e}")

# sleep_score.csv
print("\n(2) sleep_score.csv - 每日睡眠评分")
try:
    df_sleep_score = pd.read_csv(data_dir / "sleep_score.csv")
    print(f"总行数: {len(df_sleep_score):,}")
    
    sleep_score_cols = ['overall_score', 'composition_score', 'revitalization_score',
                        'duration_score', 'deep_sleep_in_minutes', 'restlessness']
    
    print(f"\n{'字段名':<25} {'总行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 65)
    
    total_rows = len(df_sleep_score)
    for col in sleep_score_cols:
        if col in df_sleep_score.columns:
            missing_count = df_sleep_score[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<25} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
except Exception as e:
    print(f"错误: {e}")

# 3. 腕部温度 (WST)
print("\n【3. 腕部温度 (WST)】")
print("-" * 80)

# computed_temperature.csv
print("\n(1) computed_temperature.csv - 计算的夜间温度")
try:
    df_temp = pd.read_csv(data_dir / "computed_temperature.csv")
    print(f"总行数: {len(df_temp):,}")
    
    temp_cols = ['nightly_temperature', 'baseline_relative_sample_sum',
                 'baseline_relative_nightly_standard_deviation']
    
    print(f"\n{'字段名':<40} {'总行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 80)
    
    total_rows = len(df_temp)
    for col in temp_cols:
        if col in df_temp.columns:
            missing_count = df_temp[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<40} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
except Exception as e:
    print(f"错误: {e}")

# wrist_temperature.csv (采样分析，因为文件很大)
print("\n(2) wrist_temperature.csv - 原始温度数据（采样分析）")
try:
    # 只读取前10000行进行快速分析
    df_wrist_temp = pd.read_csv(data_dir / "wrist_temperature.csv", nrows=10000)
    print(f"采样行数: {len(df_wrist_temp):,} (总文件可能有数百万行)")
    
    if 'temperature_diff_from_baseline' in df_wrist_temp.columns:
        missing_count = df_wrist_temp['temperature_diff_from_baseline'].isna().sum()
        missing_rate = (missing_count / len(df_wrist_temp)) * 100
        print(f"temperature_diff_from_baseline 缺失率 (基于采样): {missing_rate:.2f}%")
        print(f"注意: 这是基于前10,000行的采样分析，实际缺失率可能不同")
except Exception as e:
    print(f"错误: {e}")

# 4. 心率 (HR)
print("\n【4. 心率 (HR)】")
print("-" * 80)

# resting_heart_rate.csv
print("\n(1) resting_heart_rate.csv - 每日静息心率")
try:
    df_rhr = pd.read_csv(data_dir / "resting_heart_rate.csv")
    print(f"总行数: {len(df_rhr):,}")
    
    rhr_cols = ['value', 'error']
    
    print(f"\n{'字段名':<20} {'总行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 60)
    
    total_rows = len(df_rhr)
    for col in rhr_cols:
        if col in df_rhr.columns:
            missing_count = df_rhr[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<20} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
except Exception as e:
    print(f"错误: {e}")

# heart_rate.csv (采样分析，因为文件很大)
print("\n(2) heart_rate.csv - 连续心率测量（采样分析）")
try:
    # 只读取前10000行进行快速分析
    df_hr = pd.read_csv(data_dir / "heart_rate.csv", nrows=10000)
    print(f"采样行数: {len(df_hr):,} (总文件有63,100,277行)")
    
    hr_cols = ['bpm', 'confidence']
    
    print(f"\n{'字段名':<20} {'采样行数':<12} {'缺失数':<12} {'缺失率(%)':<12}")
    print("-" * 60)
    
    total_rows = len(df_hr)
    for col in hr_cols:
        if col in df_hr.columns:
            missing_count = df_hr[col].isna().sum()
            missing_rate = (missing_count / total_rows) * 100
            print(f"{col:<20} {total_rows:<12,} {missing_count:<12,} {missing_rate:<12.2f}")
    print("注意: 这是基于前10,000行的采样分析，实际缺失率可能不同")
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

