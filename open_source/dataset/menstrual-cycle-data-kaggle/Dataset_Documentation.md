# Menstrual Cycle Dataset Documentation
## FedCycleData071012 (2).csv

**Created:** November 2025  
**Dataset Source:** Kaggle - Menstrual Cycle Data  
**Original File:** `FedCycleData071012 (2).csv`  
**Cleaned File:** `cleaned_menstrual_data.csv`  
**Analysis Notebook:** `P1.ipynb`

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Dictionary](#data-dictionary)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Performance](#model-performance)


---

## Dataset Overview

### Basic Statistics
- **Total Records:** 1,665 cycles
- **Total Columns:** 80 variables
- **Unique Participants:** ~129 individuals
- **Data Type:** Longitudinal menstrual cycle tracking data
- **Study Design:** Natural Family Planning (NFP) fertility awareness method tracking

### Key Features
This dataset contains comprehensive menstrual cycle information including:
- **Cycle Characteristics:** Length, ovulation timing, luteal phase
- **Fertility Indicators:** Peak days, fertile window, cervical mucus patterns
- **Menstrual Patterns:** Duration, intensity scores by day
- **Demographic Information:** Age, BMI, reproductive history
- **Lifestyle Factors:** Medications, surgeries, family planning methods
- **Relationship Data:** Partner information and family planning intentions

---

## Data Dictionary

### Core Cycle Variables
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **ClientID** | Unique participant identifier | String | 0% | Format: nfp#### |
| **CycleNumber** | Sequential cycle number per participant | Integer | 0% | 1-50+ cycles tracked |
| **Group** | Study group assignment | Integer | 0% | 0=Control, 1=Treatment |
| **CycleWithPeakorNot** | Whether cycle had identifiable peak day | Integer | 0% | 0=No peak, 1=Peak detected |
| **ReproductiveCategory** | Reproductive status classification | Integer | 0% | 0=Normal, 1=Irregular, 2=Anovulatory |
| **LengthofCycle** | Total cycle length in days | Integer | 0% | **Target Variable** (21-45 days typical) |

### Fertility & Ovulation Indicators
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **MeanCycleLength** | Average cycle length for participant | Float | 91.5% | Calculated across all cycles |
| **EstimatedDayofOvulation** | Predicted ovulation day | Integer | 9.0% | Day of cycle (typically 12-16) |
| **LengthofLutealPhase** | Days from ovulation to menstruation | Integer | 9.1% | Normal: 10-16 days |
| **FirstDayofHigh** | First day of high fertility signs | Integer | 15.5% | Cervical mucus indicator |
| **TotalNumberofHighDays** | Total high fertility days | Integer | 0.7% | Pre-peak fertility window |
| **TotalHighPostPeak** | High fertility days after peak | Integer | 0% | Post-ovulation fertility |
| **TotalNumberofPeakDays** | Days with peak fertility signs | Integer | 0% | Usually 1-3 days |
| **TotalDaysofFertility** | Total fertile window length | Integer | 0% | Combined fertile days |
| **TotalFertilityFormula** | Calculated fertility window | Integer | 0% | Formula-based estimate |

### Menstrual Flow Characteristics
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **LengthofMenses** | Duration of menstrual flow | Integer | 0.2% | Typically 3-7 days |
| **MeanMensesLength** | Average menstrual duration | Float | 91.5% | Participant average |
| **MensesScoreDayOne** through **MensesScoreDay15** | Daily bleeding intensity scores | Integer | Variable | 1=Light, 2=Medium, 3=Heavy |
| **TotalMensesScore** | Sum of all daily bleeding scores | Integer | 0% | Cumulative intensity |
| **MeanBleedingIntensity** | Average daily bleeding intensity | Float | 91.5% | Normalized score |

### Sexual Activity & Fertility Awareness
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **NumberofDaysofIntercourse** | Days with intercourse in cycle | Integer | 0% | Fertility awareness tracking |
| **IntercourseInFertileWindow** | Intercourse during fertile days | Integer | 0% | 0=No, 1=Yes |
| **UnusualBleeding** | Irregular bleeding episodes | Integer | 0% | 0=Normal, 1=Unusual |
| **PhasesBleeding** | Bleeding outside menstrual phase | String | 99.9% | Mostly missing |
| **IntercourseDuringUnusBleed** | Intercourse during unusual bleeding | Integer | 99.9% | Mostly missing |

### Demographic Information
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Age** | Participant age in years | Integer | 91.5% | Reproductive age women |
| **AgeM** | Male partner age | Integer | 91.5% | Partner demographics |
| **Height** | Height in inches | Integer | 91.5% | Physical characteristics |
| **Weight** | Weight in pounds | Integer | 91.5% | For BMI calculation |
| **BMI** | Body Mass Index | Float | 91.5% | Calculated field |

### Relationship & Marital Status
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Maristatus** | Marital status | Integer | 91.5% | 0=Single, 1=Married |
| **MaristatusM** | Partner marital status | Integer | 91.5% | Partner status |
| **Yearsmarried** | Years married | Integer | 91.5% | Relationship duration |
| **Wedding** | Wedding planning status | String | 99.9% | If engaged |

### Cultural & Educational Background
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Religion** | Religious affiliation | Integer | 91.5% | Coded categories |
| **ReligionM** | Partner religious affiliation | Integer | 91.5% | Partner religion |
| **Ethnicity** | Ethnic background | Integer | 91.5% | Coded categories |
| **EthnicityM** | Partner ethnic background | Integer | 91.5% | Partner ethnicity |
| **Schoolyears** | Years of education | Integer | 91.5% | Educational attainment |
| **SchoolyearsM** | Partner years of education | Integer | 91.5% | Partner education |

### Socioeconomic Factors
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **OccupationM** | Partner occupation category | Integer | 91.5% | Coded job categories |
| **IncomeM** | Partner income level | Integer | 91.5% | Income brackets |

### Reproductive History
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Reprocate** | Reproductive category | Integer | 91.5% | Clinical classification |
| **Numberpreg** | Number of pregnancies | Integer | 91.5% | **Impute with 0** |
| **Livingkids** | Number of living children | Integer | 91.5% | **Impute with 0** |
| **Miscarriages** | Number of miscarriages | Integer | 91.5% | **Impute with 0** |
| **Abortions** | Number of abortions | Integer | 91.5% | **Impute with 0** |
| **LivingkidsM** | Partner's living children | Integer | 91.5% | From previous relationships |
| **Boys** | Number of male children | Integer | 91.5% | Gender breakdown |
| **Girls** | Number of female children | Integer | 91.5% | Gender breakdown |

### Medical History & Medications
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Medvits** | Taking medications/vitamins | Integer | 91.5% | 0=No, 1=Yes |
| **Medvitexplain** | Medication details | String | 93.7% | Free text descriptions |
| **Med_Class** | **Processed medication category** | Integer | **0%** | **0=None, 1=Supplements, 2=Moderate, 3=Hormonal** |
| **Gynosurgeries** | Gynecological surgeries | String | 99.9% | **Drop - too sparse** |
| **MedvitsM** | Partner medications | Integer | 91.5% | Partner medical status |
| **MedvitexplainM** | Partner medication details | String | 99.9% | Partner medications |
| **Urosurgeries** | Urological surgeries | String | 99.9% | **Drop - too sparse** |
| **Breastfeeding** | Currently breastfeeding | Integer | 91.5% | **Impute with 0** |

### Family Planning
| Column Name | Description | Data Type | Missing Rate | Notes |
|-------------|-------------|-----------|--------------|-------|
| **Method** | Current contraceptive method | String | 91.5% | Family planning method |
| **Prevmethod** | Previous contraceptive method | String | 99.9% | Historical method |
| **Methoddate** | Date stopped previous method | String | 99.9% | Timing information |
| **Whychart** | Reason for fertility tracking | Integer | 91.5% | Motivation for NFP |
| **Nextpreg** | Desired timing of next pregnancy | Integer | 91.5% | Family planning goals |
| **NextpregM** | Partner's pregnancy timing preference | Integer | 91.5% | Partner preferences |
| **Spousesame** | Agreement on pregnancy timing | Integer | 91.5% | Couple concordance |
| **SpousesameM** | Partner agreement confirmation | Integer | 91.5% | Partner perspective |
| **Timeattemptpreg** | Months trying to conceive | Integer | 91.5% | For those seeking pregnancy |

---

## Data Preprocessing

### Missing Data Handling Strategy

Based on the analysis in `P1.ipynb`, the following preprocessing steps were implemented:

#### 1. High Missingness Variables (>90% missing)
**Action: Drop or Impute with Domain Knowledge**

- **Surgery History** (`Gynosurgeries`, `Urosurgeries`): **DROPPED** - 99.9% missing, no learnable signal
- **Demographics** (`Age`, `BMI`, `Height`, `Weight`): **DROPPED** - 91.5% missing, imputation would create artificial correlations
- **Reproductive History** (`Numberpreg`, `Livingkids`, `Miscarriages`, `Abortions`, `Breastfeeding`): **IMPUTED with 0** - Missing indicates "no such event"

#### 2. Medication Processing
**Custom Classification System**

The `Medvitexplain` text field was processed into a structured `Med_Class` variable:

```python
# Medication Classification
Med_Class = {
    0: "No medications/supplements",
    1: "Nutritional supplements (vitamins, fish oil, probiotics)",
    2: "Moderate systemic medications (antidepressants, ADHD meds, allergy meds)",
    3: "Strong hormonal medications (thyroid, progesterone, fertility drugs)"
}
```

**Keywords for Classification:**
- **Class 3 (Hormonal):** progesterone, levothyroxine, synthroid, clomid, thyroid medications
- **Class 2 (Systemic):** antidepressants, ADHD medications, metformin, allergy medications
- **Class 1 (Supplements):** multivitamins, fish oil, probiotics, prenatal vitamins

#### 3. Client-Based Imputation
**For Core Cycle Variables**

Missing values in `EstimatedDayofOvulation` and `LengthofMenses` were filled using the mean of other cycles from the same participant:

```python
def fill_by_client_mean(df, col, id_col="ClientID"):
    # Fill missing values using participant's other cycles
    # Remove rows where no other cycles available for that participant
```

**Results:**
- `EstimatedDayofOvulation`: 147 filled, 2 removed
- `LengthofMenses`: 4 filled, 0 removed

#### 4. Final Dataset Characteristics
- **Original:** 1,665 cycles × 80 variables
- **Processed:** 1,660 cycles × 13 variables
- **Zero Missing Values** in final modeling dataset

---

## Model Performance

### Evaluation Methodology
**GroupKFold Cross-Validation** was used to prevent data leakage, ensuring all cycles from the same participant stay in the same fold.

### Target Variable: Cycle Length Prediction

#### Decision Tree Regressor
```
Configuration: max_depth=5, random_state=42
Cross-Validation: 10-fold GroupKFold

Results:
- Mean MAE: 1.871 ± 0.258 days
- Mean R²: 0.457 ± 0.137
- Precision (≤1 day): 39.3% of predictions
- Large errors (>5 days): 6.1% of predictions
- Maximum error: 25.2 days
```

#### Random Forest Regressor
```
Configuration: n_estimators=300, max_depth=None
Cross-Validation: 10-fold GroupKFold

Results:
- Mean MAE: 1.896 ± 0.238 days
- Mean R²: 0.451 ± 0.130
```

#### LSTM Neural Network
```
Configuration: 64 LSTM units, 3-cycle sequence window
Train/Validation/Test Split: Grouped by participant

Results:
- Test MAE: 1.972 days
- Test R²: 0.342
```

### Feature Importance Analysis

**Most Predictive Features (Decision Tree):**
1. **EstimatedDayofOvulation** (92.2%) - Dominant predictor
2. **LengthofMenses** (3.3%) - Secondary importance
3. **CycleWithPeakorNot** (2.6%) - Ovulation detection
4. **ReproductiveCategory** (1.7%) - Cycle regularity status
5. **Numberpreg** (0.2%) - Reproductive history

**Key Insights:**
- Ovulation timing is the strongest predictor of cycle length
- Menstrual duration provides additional predictive power
- Demographic and lifestyle factors have minimal impact
- Medication classification shows no significant predictive value

