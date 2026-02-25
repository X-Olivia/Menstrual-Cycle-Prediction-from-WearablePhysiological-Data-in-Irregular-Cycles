# cycle_lengths.npz Dataset Documentation

## 1. Dataset Overview

### 1.1 File Information
- **File Path**: `/Users/xujing/FYP/reproduction/menstrual_cycle_analysis/data/cycle_length_data/cycle_lengths.npz`
- **File Size**: 175,030 bytes (approximately 171 KB)
- **File Format**: NumPy compressed array format (.npz)

### 1.2 Dataset Structure
The dataset contains 7 primary key-value pairs:

| Key | Data Type | Shape | Dtype | Description |
|-----|-----------|-------|-------|-------------|
| `data_model` | numpy.ndarray | () | <U19 | Data model type identifier |
| `I` | numpy.ndarray | () | int64 | Number of users (5000) |
| `C` | numpy.ndarray | () | int64 | Number of cycles per user (11) |
| `hyperparameters` | numpy.ndarray | (8,) | float64 | Model hyperparameter array |
| `cycle_lengths` | numpy.ndarray | (5000, 11) | int64 | Primary data: user cycle length matrix |
| `cycle_skipped` | numpy.ndarray | (5000, 11) | int64 | Skipped cycle indicator matrix |
| `true_params` | numpy.ndarray | () | object | True parameter object |

### 1.3 Primary Data Details

#### cycle_lengths (Core Data)
- **Dimensions**: 5000 users × 11 cycles
- **Value Range**: 10-142 days
- **Mean Value**: 24.49 days
- **Data Completeness**: No missing values
- **Sample Data**: [19, 23, 24, 26, 21, ...]

#### cycle_skipped (Auxiliary Data)
- **Dimensions**: 5000 users × 11 cycles
- **Value Range**: 0-6
- **Mean Value**: 0.11
- **Meaning**: 0 indicates normal cycle, >0 indicates number of skipped cycles

#### hyperparameters
- **Length**: 8 parameters
- **Values**: [160.0, 4.0, 2.0, 20.0, 1.0, ...]
- **Purpose**: Generative model hyperparameter configuration

## 2. Dataset Usage in the Project

### 2.1 Modules Directly Using This Dataset

#### 2.1.1 Prediction Module (src/prediction/)
1. **data_functions.py**
   - `get_data()` function: Core data loading function
   - Purpose: Loads cycle_lengths.npz file, extracts key data including I, C, cycle_lengths
   - Supports both real data loading and simulated data generation

2. **All Predictive Models**
   - **Poisson Model** (`poisson_with_skipped_cycles_models.py`)
   - **Generalised Poisson Model** (`generalized_poisson_with_skipped_cycles_models.py`)
   - **Neural Network Models** (`neural_network_models.py`)
   - Purpose: Indirectly uses data through data_functions.py for model training and prediction

#### 2.1.2 Script Module (scripts/)
1. **poisson_model_fit_predict.py**
   - Directly loads cycle_lengths.npz file
   - Purpose: Basic Poisson model fitting and prediction script

2. **evaluate_predictive_models.py**
   - Uses data through evaluation_utils.py
   - Purpose: Comprehensive evaluation of multiple predictive model performance

3. **evaluation_utils.py**
   - `get_full_I_size()` function processes data
   - Purpose: Utility functions for model evaluation

### 2.2 Data Usage Flow
```
cycle_lengths.npz → data_functions.get_data() → Predictive Models → Evaluation Scripts
```

## 3. Modules Not Using This Dataset

### 3.1 Characterisation Module (src/characterization/)
The following modules **do not use** the cycle_lengths.npz dataset, but rather utilise other data sources:

1. **cohort_summary_statistics.py**
   - Data Used: `cohort_cycles_flagged.pickle`, `tracking_enriched.pickle`
   - Function: Computes cohort summary statistics

2. **cycle_period_length_analysis.py**
   - Data Used: `cohort_cycle_stats.pickle`, `cohort_cycles_flagged.pickle`, `tracking_enriched.pickle`
   - Function: Cycle and period length analysis

3. **symptom_tracking_analysis_bootstrapping.py**
   - Data Used: `cohort_clean_cycle_stats.pickle`, `tracking_enriched.pickle`
   - Function: Symptom tracking analysis (bootstrap-based)

4. **compute_cohort_clean_cycle_stats.py**
   - Function: Computes cleaned cohort cycle statistics

5. **compute_cohort_cycles_flagged.py**
   - Function: Computes flagged cohort cycles

6. **compute_cohort_clean_symptom_tracking_stats.py**
   - Function: Computes cleaned symptom tracking statistics

### 3.2 Other Unused Prediction Module Components
1. **baseline.py** - Baseline models (may use different data interfaces)
2. **plotting_functions.py** - Plotting functions (process model outputs)
3. **evaluation_functions.py** - Evaluation functions (process model results)
4. **model_evaluation_functions.py** - Model evaluation functions
5. **aux_functions.py** - Auxiliary functions

## 4. Dataset Purpose Summary

### 4.1 Primary Uses
- **Menstrual Cycle Length Prediction Modelling**: Serves as training data for machine learning and statistical models
- **Model Performance Evaluation**: Used to assess the accuracy of different predictive models
- **Algorithm Research**: Supports research into various algorithms including Poisson models, generalised Poisson models, and neural networks

### 4.2 Data Characteristics
- **Simulated Data**: This is a generated synthetic dataset, not real user data
- **Standardised Format**: Standard matrix format of 5000 users × 11 cycles
- **High Completeness**: No missing values, high data quality
- **Research-Suitable**: Appropriately sized dataset suitable for algorithm development and validation

### 4.3 Limitations
- **Prediction Tasks Only**: Primarily used for cycle length prediction, does not support symptom analysis or other research areas
- **Simulated Data**: May not fully reflect real-world complexity
- **Fixed Format**: Relatively fixed data structure with limited extensibility

## 5. Technical Specifications

### 5.1 Data Loading Method
```python
import numpy as np
data = np.load('cycle_lengths.npz', allow_pickle=True)
I = data['I']  # Number of users
C = data['C']  # Number of cycles
cycle_lengths = data['cycle_lengths']  # Primary data
```

### 5.2 Environment Requirements
- Requires use of the project's conda environment (`menstrual`) to avoid numpy architecture compatibility issues
- Python 3.x + NumPy + PyTorch (for neural network models)

---
