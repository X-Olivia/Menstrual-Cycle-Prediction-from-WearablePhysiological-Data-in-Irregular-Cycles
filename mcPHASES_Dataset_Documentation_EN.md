# mcPHASES Dataset Documentation

## Dataset Overview

**mcPHASES** is a comprehensive physiological health tracking dataset specifically designed for menstrual health research and wearable device data analysis. This dataset integrates data from various personal health tracking devices and survey questionnaires, providing researchers with rich physiological, hormonal, and self-reported event and symptom data.

### Basic Information
- **Dataset Version**: 1.0.0
- **License**: PhysioNet Restricted Health Data License Version 1.5.0
- **Data Collection Period**: 
  - Phase 1: January-April 2022
  - Phase 2: July-October 2024
- **Total Data Volume**: Approximately 110 million records
- **Number of Participants**: 42 female participants

## Data Structure and Organization

### Core Identifiers
All data tables can be linked through the following key fields:
- `id`: Unique participant identifier
- `day_in_study`: Standardized day index from each participant's study start (beginning from day 1)
- `study_interval`: Data collection phase identifier (2022 or 2024)
- `is_weekend`: Boolean value indicating whether the record date is a weekend

### Time Span Measurements
For measurements spanning time (such as sleep), tables may contain the following columns:
- `sleep_start_day_in_study` / `sleep_end_day_in_study`
- `start_day_in_study` / `end_day_in_study`

## Detailed Data File Descriptions

### 1. Participant Basic Information

#### subject-info.csv (42 rows)
Contains demographic and background survey information for participants:
- `birth_year`: Year of birth
- `gender`: Self-identified gender
- `ethnicity`: Self-identified race/ethnicity
- `education`: Highest education level
- `sexually_active`: Whether sexually active (Yes/No)
- `self_report_menstrual_health_literacy`: Participant's self-rated menstrual health literacy level
- `age_of_first_menarche`: Age at first menstruation

#### height_and_weight.csv (43 rows)
Participant self-reported height and weight data:
- `height_2022/2024`: Height (centimeters)
- `weight_2022/2024`: Weight (kilograms)

### 2. Hormone and Self-Report Data

#### hormones_and_selfreport.csv (5,660 rows)
Combines hormone data from Mira fertility device with daily self-reported symptom surveys:

**Menstrual Cycle Phase Labels**:
- `phase`: Menstrual cycle phase label - **100% coverage** (5,658/5,659 records labeled)
  - **Follicular**: 1,386 records (24.5%)
    - From end of menstruation to before ovulation
    - Characteristics: Baseline LH levels (1-5 mIU/mL), gradually increasing E3G
  - **Fertility (Ovulation/Fertile Window)**: 1,281 records (22.6%)
    - Fertile window around ovulation
    - Characteristics: LH surge (can reach 40+ mIU/mL), high E3G levels (200-400 ng/mL)
  - **Luteal**: 1,912 records (33.8%)
    - From after ovulation to before menstruation
    - Characteristics: LH returns to baseline, E3G decreases, PDG increases
  - **Menstrual**: 1,079 records (19.1%)
    - During menstrual bleeding
    - Characteristics: Flow volume reported, relatively low hormone levels

**Phase Labeling Basis**:
According to README, phase labels are "available or derived". Based on data analysis, labeling criteria include:
1. **LH Surge Detection**: LH surge (>10 mIU/mL) indicates ovulation, used to determine Fertility period
2. **Menstrual Flow Reports**: Self-reported menstrual flow used to determine Menstrual period
3. **Hormone Patterns**: Combined variation patterns of E3G and LH
4. **Time Calculation**: Based on typical menstrual cycle temporal patterns

**Hormone Indicators** (via urine testing):
- `lh`: **Luteinizing Hormone** level (mIU/mL) - 5,339 valid records
  - Normal range: 1-5 mIU/mL (baseline), can peak before ovulation
  - Used to predict ovulation timing
- `estrogen`: **Estrone-3-Glucuronide** (E3G) level (ng/mL) - 5,338 valid records
  - Estrogen metabolite
  - Reflects body estrogen levels
  - Gradually rises during follicular phase, peaks before ovulation
- `pdg`: **Pregnanediol Glucuronide** (PdG) level (mcg/mL) - 1,864 valid records
  - Progesterone metabolite
  - Mainly secreted during luteal phase
  - Used to confirm ovulation occurrence

**Data Characteristics**:
- Data source: Mira Fertility Plus smart fertility monitor
- Detection method: Urine immunochromatography
- Sampling frequency: **Near-daily measurements**
  - LH and E3G: 94.3% of days have data (5,339/5,659 days)
  - Most participants have >90% coverage, some reaching 100%
  - Data continuity as high as 99.8% (very few skipped days)
- Less PDG data (1,864 records) because mainly measured during luteal phase

**Self-Reported Symptoms** (0-5 Likert scale, 0="not at all", 5="very high"):
- `flow_volume`: Menstrual flow volume
- `flow_color`: Menstrual flow color classification
- `appetite`: Appetite level
- `exerciselevel`: Exercise level
- `headaches`: Headache severity
- `cramps`: Cramp severity
- `sorebreasts`: Breast tenderness
- `fatigue`: Fatigue level
- `sleepissue`: Sleep issues
- `moodswing`: Mood swings
- `stress`: Stress level
- `foodcravings`: Food cravings
- `indigestion`: Indigestion
- `bloating`: Bloating level

### 3. Activity and Exercise Data

#### active_minutes.csv (5,553 rows)
Physical activity duration categorized by intensity:
- `sedentary`: Sedentary activity minutes
- `lightly`: Light activity minutes
- `moderately`: Moderate activity minutes
- `very`: High-intensity activity minutes

#### active_zone_minutes.csv (154,483 rows)
Activity time in different heart rate zones:
- `timestamp`: Record time
- `heart_zone_id`: Heart rate zone (fat burn, cardio, peak)
- `total_minutes`: Total minutes in corresponding heart rate zone

#### exercise.csv (7,283 rows)
Detailed exercise session records:
- `activityname`: Activity name (e.g., walking, cycling)
- `activitytypeid`: Fitbit internal activity type ID
- `averageheartrate`: Average heart rate (bpm)
- `calories`: Calories burned
- `duration`: Duration (milliseconds)
- `steps`: Step count
- `elevationgain`: Elevation gain (meters)
- `hasgps`: Whether GPS data is available

#### steps.csv (7,666,950 rows)
All-day step tracking:
- `timestamp`: Record time
- `steps`: Step count recorded

#### distance.csv (7,666,950 rows)
Distance tracking data:
- `timestamp`: Record time
- `distance`: Distance covered (meters)

#### calories.csv (20,166,976 rows)
Calorie expenditure records:
- `timestamp`: Record time
- `calories`: Calories burned

#### altitude.csv (90,879 rows)
Relative altitude changes:
- `timestamp`: Record time
- `altitude`: Elevation gain (meters, non-GPS data)

### 4. Heart Rate Related Data

#### heart_rate.csv (63,100,277 rows) - **Largest data file**
Continuous heart rate measurements:
- `timestamp`: Record time
- `bpm`: Beats per minute
- `confidence`: Fitbit-generated confidence level of heart rate reading

#### resting_heart_rate.csv (13,738 rows)
Daily resting heart rate:
- `value`: Estimated resting heart rate (bpm)
- `error`: Estimated error range

#### heart_rate_variability_details.csv (436,263 rows)
Heart rate variability detailed data (5-minute intervals during sleep):
- `rmssd`: Root mean square of successive differences between heartbeats (time-domain HRV metric)
- `coverage`: Data point coverage (data quality indicator)
- `low_frequency`: **LF-HRV** - Low-frequency heart rate variability, measures long-term heart rate changes, reflects sympathetic and parasympathetic nervous activity
- `high_frequency`: **HF-HRV** - High-frequency heart rate variability, measures short-term heart rate changes, reflects parasympathetic nervous activity and respiratory sinus arrhythmia

**Note**: HF-HRV data is primarily collected during sleep at 5-minute intervals, serving as an important indicator for studying autonomic nervous system activity.

#### time_in_heart_rate_zones.csv (5,554 rows)
Time distribution in different heart rate zones:
- `in_default_zone_3`: Time in peak zone
- `in_default_zone_2`: Time in cardio zone
- `in_default_zone_1`: Time in fat burn zone
- `below_default_zone_1`: Time below fat burn zone

### 5. Sleep Data

#### sleep.csv (14,766 rows)
Detailed sleep session records:
- `duration`: Total sleep duration (milliseconds)
- `minutestofallasleep`: Time to fall asleep (minutes)
- `minutesasleep`: Actual sleep time (minutes)
- `minutesawake`: Time awake (minutes)
- `efficiency`: Sleep efficiency (percentage of time asleep in time in bed)
- `levels`: Sleep stage details (JSON format, including deep sleep, light sleep, REM, etc.)

#### sleep_score.csv (5,309 rows)
Daily sleep quality score:
- `overall_score`: Overall sleep score (out of 100)
- `composition_score`: Sleep composition score
- `revitalization_score`: Revitalization score
- `duration_score`: Sleep duration score
- `deep_sleep_in_minutes`: Deep sleep minutes
- `restlessness`: Sleep restlessness based on movement

#### respiratory_rate_summary.csv (6,302 rows)
Respiratory rate summary (per night of sleep):
- `full_sleep_breathing_rate`: Average breathing rate for entire night
- `deep_sleep_breathing_rate`: Breathing rate during deep sleep
- `light_sleep_breathing_rate`: Breathing rate during light sleep
- `rem_sleep_breathing_rate`: Breathing rate during REM sleep

### 6. Physiological Indicators

#### glucose.csv (837,131 rows)
Continuous glucose monitoring data (Dexcom device):
- `timestamp`: Record time
- `glucose_value`: Glucose concentration (mmol/L)

#### wrist_temperature.csv (6,856,020 rows)
Wrist skin temperature variations:
- `timestamp`: Record time
- `temperature_diff_from_baseline`: Temperature difference from personal baseline (Celsius)

#### computed_temperature.csv (5,576 rows)
Computed temperature readings (primarily during sleep):
- `nightly_temperature`: Computed nightly average skin temperature
- `baseline_relative_sample_sum`: Sum of deviations from baseline temperature
- `baseline_relative_nightly_standard_deviation`: Standard deviation of nightly temperature relative to baseline

#### estimated_oxygen_variation.csv (3,070,313 rows)
Estimated blood oxygen variation (during sleep):
- `infrared_to_red_signal_ratio`: Infrared to red light absorption ratio

### 7. Health Assessment

#### demographic_vo2_max.csv (11,483 rows)
Maximum oxygen uptake estimate:
- `demographic_vo2_max`: VO2 Max estimate based on demographics and heart rate data
- `demographic_vo2_max_error`: Estimation error
- `filtered_demographic_vo2_max`: Filtered VO2 Max value

#### stress_score.csv (7,933 rows)
Stress management score:
- `stress_score`: Stress score
- `sleep_points`: Sleep component score
- `responsiveness_points`: Responsiveness component score
- `exertion_points`: Exertion component score

## Data Quality and Completeness

### Data Volume Statistics
- **Largest file**: heart_rate.csv (63,100,277 rows)
- **Smallest file**: subject-info.csv (42 rows)
- **High-frequency data**: Heart rate, steps, distance, calories (minute-level records)
- **Medium-frequency data**: Glucose (5-minute intervals), temperature variations
- **Low-frequency data**: Sleep, hormones, self-reports (day-level records)

### Data Completeness
- Some participants may have missing measurement data
- Height and weight data have considerable missing values
- Hormone data only collected during specific periods
- All timestamps are in local time

## Research Application Scenarios

### 1. Menstrual Cycle Research
- Hormone level variation analysis
- Symptom pattern recognition
- Cycle prediction model development

### 2. Sleep Quality Analysis
- Relationship between sleep stages and menstrual cycle
- Factors affecting sleep quality
- Personalized sleep recommendations

### 3. Physiological Indicator Monitoring
- Heart rate variability analysis
- Blood glucose fluctuation patterns
- Body temperature change trends

### 4. Behavioral Pattern Research
- Exercise habits and menstrual cycle
- Stress levels and physiological indicators
- Self-reported symptom validation

### 5. Wearable Device Data Mining
- Multimodal data fusion
- Anomaly detection algorithms
- Personalized health recommendations

## Data Usage Precautions

### Privacy Protection
- All data has been de-identified
- Complies with PhysioNet Restricted Health Data License
- Prohibited to attempt personal identification
- Must not share data access with others

### Technical Requirements
- Large file processing capability (heart rate data >63 million rows)
- JSON data parsing (sleep details, exercise data)
- Time series data processing
- Multi-table relational queries

### Data Quality Considerations
- Device accuracy limitations
- User wearing compliance
- Self-report subjectivity
- Missing data handling

## File Integrity Verification

The dataset includes a SHA256 checksum file (SHA256SUMS.txt) that can be used to verify file integrity:
```bash
sha256sum -c SHA256SUMS.txt
```



