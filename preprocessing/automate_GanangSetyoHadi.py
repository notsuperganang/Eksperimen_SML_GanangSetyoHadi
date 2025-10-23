"""
Heart Failure Prediction - Automated Preprocessing Pipeline
Author: Ganang Setyo Hadi
Description: Automated preprocessing functions to prepare data for model training
             with proper train-test isolation to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load raw dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the raw dataset CSV file
        
    Returns:
    --------
    df : pd.DataFrame
        Raw dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def split_data(df, target_column='HeartDisease', test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    IMPORTANT: This should be called FIRST before any preprocessing!
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    target_column : str
        Name of target column
    test_size : float
        Proportion of test set (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        Train and test splits
    """
    X = df.drop(target_column, axis=1).copy()
    y = df[target_column].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def handle_implicit_missing(X_train, X_test):
    """
    Convert impossible values (0) to NaN for medical features.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets
        
    Returns:
    --------
    X_train, X_test : pd.DataFrame
        Feature sets with implicit missing values converted to NaN
    """
    # Convert Cholesterol = 0 to NaN
    cholesterol_zero_train = (X_train['Cholesterol'] == 0).sum()
    cholesterol_zero_test = (X_test['Cholesterol'] == 0).sum()
    X_train['Cholesterol'] = X_train['Cholesterol'].replace(0, np.nan)
    X_test['Cholesterol'] = X_test['Cholesterol'].replace(0, np.nan)
    
    # Convert RestingBP = 0 to NaN
    bp_zero_train = (X_train['RestingBP'] == 0).sum()
    bp_zero_test = (X_test['RestingBP'] == 0).sum()
    X_train['RestingBP'] = X_train['RestingBP'].replace(0, np.nan)
    X_test['RestingBP'] = X_test['RestingBP'].replace(0, np.nan)
    
    print(f"✓ Implicit missing handled: Cholesterol({cholesterol_zero_train}+{cholesterol_zero_test}), RestingBP({bp_zero_train}+{bp_zero_test})")
    
    return X_train, X_test


def grouped_imputation(X_train, X_test):
    """
    Impute missing values using grouped median (by Sex and AgeGroup).
    Statistics are computed from TRAIN set only to prevent data leakage.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets with missing values
        
    Returns:
    --------
    X_train, X_test : pd.DataFrame
        Feature sets with imputed values
    """
    # Create age groups
    X_train['AgeGroup'] = pd.cut(X_train['Age'], 
                                  bins=[0, 45, 60, 100], 
                                  labels=['Young', 'Middle', 'Old'])
    X_test['AgeGroup'] = pd.cut(X_test['Age'], 
                                 bins=[0, 45, 60, 100], 
                                 labels=['Young', 'Middle', 'Old'])
    
    # Calculate imputation statistics from TRAIN only
    imputation_stats = {}
    
    # Cholesterol
    if X_train['Cholesterol'].isnull().sum() > 0:
        chol_medians = X_train.groupby(['Sex', 'AgeGroup'])['Cholesterol'].median()
        chol_fallback = X_train['Cholesterol'].median()
        imputation_stats['Cholesterol'] = {'medians': chol_medians, 'fallback': chol_fallback}
    
    # RestingBP
    if X_train['RestingBP'].isnull().sum() > 0:
        bp_medians = X_train.groupby(['Sex', 'AgeGroup'])['RestingBP'].median()
        bp_fallback = X_train['RestingBP'].median()
        imputation_stats['RestingBP'] = {'medians': bp_medians, 'fallback': bp_fallback}
    
    # Apply imputation function
    def apply_imputation(df, column, stats):
        """Apply pre-computed imputation statistics"""
        for index, row in df[df[column].isnull()].iterrows():
            group_key = (row['Sex'], row['AgeGroup'])
            if group_key in stats['medians'].index:
                df.at[index, column] = stats['medians'][group_key]
            else:
                df.at[index, column] = stats['fallback']
        return df
    
    # Impute both sets using train statistics
    if 'Cholesterol' in imputation_stats:
        X_train = apply_imputation(X_train, 'Cholesterol', imputation_stats['Cholesterol'])
        X_test = apply_imputation(X_test, 'Cholesterol', imputation_stats['Cholesterol'])
    
    if 'RestingBP' in imputation_stats:
        X_train = apply_imputation(X_train, 'RestingBP', imputation_stats['RestingBP'])
        X_test = apply_imputation(X_test, 'RestingBP', imputation_stats['RestingBP'])
    
    # Drop temporary AgeGroup column
    X_train = X_train.drop('AgeGroup', axis=1)
    X_test = X_test.drop('AgeGroup', axis=1)
    
    # Final fallback for any remaining missing values
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            X_test[col] = X_test[col].fillna(X_train[col].median())
    
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"✓ Grouped imputation complete: Train missing={train_missing}, Test missing={test_missing}")
    
    return X_train, X_test


def encode_features(X_train, X_test):
    """
    Encode categorical features with proper alignment between train and test.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets
        
    Returns:
    --------
    X_train, X_test : pd.DataFrame
        Encoded feature sets with aligned columns
    """
    # Binary features - Label Encoding
    binary_mappings = {
        'Sex': {'F': 0, 'M': 1},
        'ExerciseAngina': {'N': 0, 'Y': 1}
    }
    
    for feature, mapping in binary_mappings.items():
        X_train[feature] = X_train[feature].map(mapping)
        X_test[feature] = X_test[feature].map(mapping)
    
    # Ordinal features - Ordered Label Encoding
    ordinal_mappings = {
        'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
    }
    
    for feature, mapping in ordinal_mappings.items():
        X_train[feature] = X_train[feature].map(mapping)
        X_test[feature] = X_test[feature].map(mapping)
    
    # Nominal features - One-Hot Encoding with alignment
    nominal_features = ['ChestPainType', 'RestingECG']
    
    X_train = pd.get_dummies(X_train, columns=nominal_features, 
                             prefix=nominal_features, drop_first=False, dtype=int)
    X_test = pd.get_dummies(X_test, columns=nominal_features, 
                            prefix=nominal_features, drop_first=False, dtype=int)
    
    # Align test columns with train
    # Add missing columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    
    # Remove extra columns
    extra_cols = set(X_test.columns) - set(X_train.columns)
    for col in extra_cols:
        X_test = X_test.drop(col, axis=1)
    
    # Reorder columns
    X_test = X_test[X_train.columns]
    
    print(f"✓ Features encoded: {X_train.shape[1]} features, columns aligned")
    
    return X_train, X_test


def scale_features(X_train, X_test):
    """
    Scale numerical features using RobustScaler.
    Scaler is fitted on TRAIN set only to prevent data leakage.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets
        
    Returns:
    --------
    X_train_scaled, X_test_scaled : pd.DataFrame
        Scaled feature sets
    scaler : RobustScaler
        Fitted scaler object for future use
    """
    # Features to scale (those with outliers)
    robust_features = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Age']
    
    # Initialize scaler
    scaler = RobustScaler()
    
    # Create copies
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit on train, transform both
    scaler.fit(X_train[robust_features])
    X_train_scaled[robust_features] = scaler.transform(X_train[robust_features])
    X_test_scaled[robust_features] = scaler.transform(X_test[robust_features])
    
    print(f"✓ Features scaled: {len(robust_features)} features using RobustScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def validate_preprocessing(X_train, X_test, y_train, y_test):
    """
    Validate preprocessing results to ensure data quality.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets
    y_train, y_test : pd.Series
        Train and test target sets
        
    Returns:
    --------
    validation_results : dict
        Dictionary containing validation check results
    """
    results = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'train_missing': X_train.isnull().sum().sum(),
        'test_missing': X_test.isnull().sum().sum(),
        'columns_match': list(X_train.columns) == list(X_test.columns),
        'all_numeric_train': X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1],
        'all_numeric_test': X_test.select_dtypes(include=[np.number]).shape[1] == X_test.shape[1],
        'target_balance_train': y_train.value_counts().to_dict(),
        'target_balance_test': y_test.value_counts().to_dict()
    }
    
    # Print validation summary
    print("\n" + "="*60)
    print("PREPROCESSING VALIDATION")
    print("="*60)
    print(f"Train shape: {results['train_shape']}")
    print(f"Test shape: {results['test_shape']}")
    print(f"Missing values - Train: {results['train_missing']}, Test: {results['test_missing']}")
    print(f"Columns match: {results['columns_match']}")
    print(f"All numeric - Train: {results['all_numeric_train']}, Test: {results['all_numeric_test']}")
    print(f"Target balance (train): {results['target_balance_train']}")
    print(f"Target balance (test): {results['target_balance_test']}")
    
    all_pass = (
        results['train_missing'] == 0 and 
        results['test_missing'] == 0 and
        results['columns_match'] and
        results['all_numeric_train'] and
        results['all_numeric_test']
    )
    
    if all_pass:
        print("\n✓ All validation checks PASSED!")
        print("✓ Dataset ready for model training!")
    else:
        print("\n⚠️  Some validation checks FAILED!")
    
    print("="*60)
    
    return results


def preprocess_pipeline(filepath, target_column='HeartDisease', test_size=0.2, random_state=42, save_output=False, output_dir='./'):
    """
    Complete preprocessing pipeline from raw data to model-ready data.
    Applies all preprocessing steps in correct order to prevent data leakage.
    
    Parameters:
    -----------
    filepath : str
        Path to raw dataset CSV file
    target_column : str
        Name of target column (default: 'HeartDisease')
    test_size : float
        Proportion of test set (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    save_output : bool
        Whether to save preprocessed data to CSV (default: False)
    output_dir : str
        Directory to save output files (default: './')
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        Preprocessed train and test sets ready for model training
    scaler : RobustScaler
        Fitted scaler object for future use
    validation_results : dict
        Validation check results
        
    Example:
    --------
    >>> X_train, X_test, y_train, y_test, scaler, results = preprocess_pipeline(
    ...     filepath='heart_raw.csv',
    ...     save_output=True,
    ...     output_dir='./preprocessing/'
    ... )
    """
    print("\n" + "="*60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("="*60)
    print("Author: Ganang Setyo Hadi")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("[Step 1/6] Loading data...")
    df = load_data(filepath)
    
    # Step 2: Split data (FIRST!)
    print("\n[Step 2/6] Splitting data (train-test)...")
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)
    
    # Step 3: Handle implicit missing values
    print("\n[Step 3/6] Handling implicit missing values...")
    X_train, X_test = handle_implicit_missing(X_train, X_test)
    
    # Step 4: Grouped imputation
    print("\n[Step 4/6] Applying grouped imputation...")
    X_train, X_test = grouped_imputation(X_train, X_test)
    
    # Step 5: Encode features
    print("\n[Step 5/6] Encoding categorical features...")
    X_train, X_test = encode_features(X_train, X_test)
    
    # Step 6: Scale features
    print("\n[Step 6/6] Scaling numerical features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    # Validate preprocessing
    validation_results = validate_preprocessing(X_train, X_test, y_train, y_test)
    
    # Save preprocessed data if requested
    if save_output:
        print(f"\nSaving preprocessed data to {output_dir}...")
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        print("✓ Preprocessed data saved successfully!")
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, scaler, validation_results


# Main execution
if __name__ == "__main__":
    """
    Example usage of the automated preprocessing pipeline.
    """
    import os
    
    # Configuration - handle path based on where script is run from
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(script_dir)
    
    RAW_DATA_PATH = os.path.join(project_root, "heart_raw.csv")
    OUTPUT_DIR = os.path.join(script_dir, "heart_preprocessing")  

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("RUNNING AUTOMATED PREPROCESSING")
    print("="*60)
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, results = preprocess_pipeline(
        filepath=RAW_DATA_PATH,
        target_column='HeartDisease',
        test_size=0.2,
        random_state=42,
        save_output=True,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✓ Preprocessing completed successfully!")
    print(f"✓ Training data ready: {X_train.shape}")
    print(f"✓ Test data ready: {X_test.shape}")
    print(f"\nPreprocessed data saved to: {OUTPUT_DIR}")
    print("\nYou can now use this data for model training!")
