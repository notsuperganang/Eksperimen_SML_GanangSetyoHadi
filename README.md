# Eksperimen Supervised Machine Learning - Ganang Setyo Hadi

Repository ini berisi eksperimen machine learning untuk prediksi Heart Failure menggunakan dataset dari Kaggle. Project ini merupakan bagian dari submission Dicoding - Machine Learning Operations (MLOps) class.

## 📊 Dataset

**Heart Failure Prediction Dataset**
- **Source**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size**: 918 observations
- **Features**: 11 clinical features + 1 target variable
- **Target**: HeartDisease (Binary: 0=Normal, 1=Heart Disease)

### Dataset Features:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Age of the patient (years) |
| Sex | Categorical | Sex of the patient (M/F) |
| ChestPainType | Categorical | Type of chest pain (TA/ATA/NAP/ASY) |
| RestingBP | Numerical | Resting blood pressure (mm Hg) |
| Cholesterol | Numerical | Serum cholesterol (mm/dl) |
| FastingBS | Numerical | Fasting blood sugar (1 if >120 mg/dl, 0 otherwise) |
| RestingECG | Categorical | Resting electrocardiogram results |
| MaxHR | Numerical | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Exercise-induced angina (Y/N) |
| Oldpeak | Numerical | ST depression induced by exercise |
| ST_Slope | Categorical | Slope of peak exercise ST segment |
| HeartDisease | Binary | Target variable (1=disease, 0=normal) |

## 🎯 Project Objectives

1. **Exploratory Data Analysis (EDA)** - Comprehensive data exploration and insights extraction
2. **Data Preprocessing** - Clean and prepare data with proper train-test isolation
3. **Automated Pipeline** - Create reusable preprocessing functions
4. **CI/CD Integration** - Implement GitHub Actions for automated preprocessing
5. **Model Training** - Build and evaluate machine learning models (future work)

## 📁 Project Structure

```
Eksperimen_SML_GanangSetyoHadi/
├── .github/
│   └── workflows/              # GitHub Actions workflows (to be added)
├── preprocessing/
│   ├── Eksperimen_GanangSetyoHadi.ipynb    # Experimentation notebook
│   ├── automate_GanangSetyoHadi.py         # Automated preprocessing script
│   └── heart_preprocessing/                 # Preprocessed data output
│       ├── X_train.csv                      # Training features
│       ├── X_test.csv                       # Test features
│       ├── y_train.csv                      # Training labels
│       └── y_test.csv                       # Test labels
├── heart_raw.csv                            # Raw dataset
└── README.md                                # Project documentation
```

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

1. **Missing Values**:
   - 172 Cholesterol values = 0 (18.74%) - Implicit missing values
   - 1 RestingBP value = 0 (0.11%) - Implicit missing values
   - These are medically impossible values treated as missing

2. **Target Distribution**:
   - Heart Disease: 508 cases (55.34%)
   - Normal: 410 cases (44.66%)
   - Relatively balanced dataset

3. **Top Predictive Features**:
   - ST_Slope: correlation 0.51 with target
   - ExerciseAngina: correlation 0.49 with target
   - MaxHR: correlation -0.42 with target (negative)
   - Oldpeak: correlation 0.43 with target

4. **Data Quality**:
   - No duplicate rows
   - Outliers are legitimate medical cases (kept in dataset)
   - No multicollinearity detected (all |r| < 0.7)

## 🛠️ Data Preprocessing Pipeline

### Critical Principle: **No Data Leakage**

All preprocessing steps are designed to prevent data leakage by following these principles:
- Train-test split is performed **FIRST** before any transformations
- All statistics (imputation, encoding, scaling) are computed from **training set only**
- Same transformations are applied to both train and test sets

### Preprocessing Steps:

#### 1. Train-Test Split (80:20)
- Stratified split to maintain class balance
- Random seed: 42 for reproducibility
- **Performed BEFORE any preprocessing**

#### 2. Handling Implicit Missing Values
- Convert Cholesterol = 0 → NaN
- Convert RestingBP = 0 → NaN
- Applied to both train and test sets

#### 3. Grouped Imputation
- Group by: `[Sex, AgeGroup]` (NOT including target variable)
- Imputation method: Median
- **Statistics computed from training set only**
- Applied to both sets using train statistics
- Fallback to overall median if group not found in train

#### 4. Categorical Encoding
- **Binary Features** (Label Encoding):
  - Sex: F=0, M=1
  - ExerciseAngina: N=0, Y=1
  
- **Ordinal Features** (Ordered Label Encoding):
  - ST_Slope: Up=0, Flat=1, Down=2
  
- **Nominal Features** (One-Hot Encoding):
  - ChestPainType (4 categories)
  - RestingECG (3 categories)
  - Columns aligned between train and test (dtype=int)

#### 5. Feature Scaling
- Method: RobustScaler (robust to outliers)
- Features scaled: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- **Scaler fitted on training set only**
- Transform applied to both sets

### Validation Checks:

✅ No missing values in train and test  
✅ All features are numeric (int/float)  
✅ Train and test columns aligned  
✅ No data leakage detected  
✅ Dataset ready for model training  

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Dependencies

```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
imbalanced-learn==0.11.0
```

### Running the Preprocessing Pipeline

#### Option 1: Using the Automated Script

```bash
# Run the automated preprocessing pipeline
python preprocessing/automate_GanangSetyoHadi.py
```

This will:
1. Load the raw dataset
2. Perform all preprocessing steps
3. Save preprocessed data to `preprocessing/heart_preprocessing/`
4. Display validation results

#### Option 2: Using the Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook preprocessing/Eksperimen_GanangSetyoHadi.ipynb
```

Run all cells to see the complete experimentation process including:
- Detailed EDA with visualizations
- Step-by-step preprocessing with explanations
- Validation and quality checks

#### Option 3: Import as Module

```python
from preprocessing.automate_GanangSetyoHadi import preprocess_pipeline

# Run preprocessing
X_train, X_test, y_train, y_test, scaler, results = preprocess_pipeline(
    filepath='heart_raw.csv',
    save_output=True,
    output_dir='./preprocessing/heart_preprocessing/'
)

# Use the data for model training
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
```

## 📈 Results

### Preprocessing Output:

| File | Description | Shape |
|------|-------------|-------|
| `X_train.csv` | Training features | (734, 18) |
| `X_test.csv` | Test features | (184, 18) |
| `y_train.csv` | Training labels | (734,) |
| `y_test.csv` | Test labels | (184,) |

### Data Quality Metrics:

- **Missing Values**: 0 (all handled)
- **Feature Count**: 18 (after encoding)
- **Data Types**: All numeric (int64/float64)
- **Train-Test Alignment**: Perfect match
- **Data Leakage**: None detected

## 🔄 Future Work

- [ ] Implement GitHub Actions for automated preprocessing
- [ ] Model training and hyperparameter tuning
- [ ] Model evaluation and comparison
- [ ] MLflow integration for experiment tracking
- [ ] Model deployment pipeline
- [ ] Monitoring and logging setup

## 📚 References

1. **Dataset**: [Heart Failure Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
2. **Original Study**: Fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved from https://www.kaggle.com/fedesoriano/heart-failure-prediction
3. **MLOps Course**: Dicoding - Machine Learning Operations

## 👤 Author

**Ganang Setyo Hadi**

Submission for Dicoding - Machine Learning Operations Class

---

## 📝 Notes

### Data Preprocessing Best Practices Applied:

1. ✅ **Train-Test Split First**: Prevents data snooping
2. ✅ **Fit on Train Only**: All transformations learn from training data
3. ✅ **No Target Leakage**: Target variable not used in feature engineering
4. ✅ **Proper Validation**: Comprehensive checks for data quality
5. ✅ **Reproducibility**: Fixed random seeds and documented process
6. ✅ **Modular Code**: Reusable functions for production use

### Key Considerations:

- **Implicit Missing Values**: Medical domain knowledge used to identify impossible values
- **Outlier Handling**: Outliers are kept as they represent legitimate medical cases
- **Feature Engineering**: Minimal engineering to preserve interpretability
- **Scaling**: RobustScaler chosen for resilience to outliers
- **Documentation**: Comprehensive inline comments and docstrings

## 📄 License

This project is created for educational purposes as part of Dicoding's MLOps class submission.

Dataset is publicly available on Kaggle under its respective license.

---

**Last Updated**: October 23, 2025
