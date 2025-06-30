# 1.Install the necessary packages
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# 2.Import you data and perform basic data exploration phase
# 2.1 General information about the dataset
@st.cache_data
def load_data():
    st.write("Loading app...")
    df = pd.read_csv('data/Financial_inclusion_dataset.csv')

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns) # Column names
    print(df.dtypes)  # Data types
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df

def preprocess_input(df):
    # Make a copy of the dataframe to avoid modifying original data
    df = df.copy()

    # 2.2 Create a Profiling Report
    profile = ProfileReport(df, title="Financial inclusion Profiling Report", explorative=True)
    profile.to_file("profiling_report.html")

    # 2.3 Handle Missing and corrupted values
    # Check Missing values
    df.isnull().sum()  # Count missing values

    ''' No missing values'''

    def detect_corrupted_values_all_columns(df):
        print("🔍 Scanning all columns for potential corrupted values...\n")

        for col in df.columns:
            print(f"--- Checking column: {col} ---")
            series = df[col]
            dtype = series.dtype

            # 1. Check for numeric conversion if object
            if dtype == 'object':
                # Try to convert to numeric
                coerced = pd.to_numeric(series, errors='coerce')
                if coerced.isnull().sum() > 0 and coerced.notnull().sum() > 0:
                    print(f"⚠️ Column '{col}' contains non-numeric values. Example problematic values:")
                    print(series[coerced.isnull()].unique()[:5])

            # 2. Check for invalid datetime conversion
            try:
                parsed = pd.to_datetime(series, errors='coerce')
                if parsed.notnull().sum() > 0 and parsed.isnull().sum() > 0:
                    print(f"⚠️ Column '{col}' has {parsed.isnull().sum()} unparseable date(s).")
            except Exception:
                pass  # not a date column

            # 3. Check for extreme values in numeric columns
            if pd.api.types.is_numeric_dtype(series):
                col_min = series.min()
                col_max = series.max()

                if col_min < -1e6 or col_max > 1e6:
                    print(f"⚠️ Column '{col}' has extreme values (min: {col_min}, max: {col_max}).")

                if (series < 0).sum() > 0 and 'age' in col.lower():
                    print(f"⚠️ Column '{col}' has negative values (potential issue for 'age').")

            # 4. Categorical check
            if dtype == 'object' or dtype.name == 'category':
                unique_vals = series.nunique()
                if unique_vals > df.shape[0] * 0.5:
                    print(f"⚠️ Column '{col}' has high cardinality ({unique_vals} unique values).")
                elif unique_vals <= 10:
                    print(f"✅ Unique values in '{col}': {series.unique()}")

            print("")  # spacing

        print("✅ Scan completed.\n")
    detect_corrupted_values_all_columns(df)

    # Handle corrupted values
    replace_dict = {
        'marital_status': {
            'Divorced/Seperated': 'Divorced/Separated',
            'Dont know': "Don't know"
        },
        'education_level': {
            'Other/Dont know/RTA': 'Other/Don\'t know/RTA'
        },
        'job_type': {
            'Dont Know/Refuse to answer': "Don't know/Refuse to answer"
        }
    }

    for col, mapping in replace_dict.items():
        df[col] = df[col].replace(mapping)



    # 2.4 Remove Duplicates
    df.duplicated().sum()     # Check duplicates
    df.drop_duplicates(inplace=True)

    # 2.5 Handle outliers, if they exist
    # First step : outliers Detection
    cols = ['year','household_size','age_of_respondent']

    df_subset = df[cols]

    # 1. Visualize data distribution with boxplots
    plt.figure(figsize=(15, 6))
    df_subset.boxplot()
    plt.title('Boxplots of Selected Numeric Features')
    plt.xticks(rotation=45)
    plt.show()

    # 2. Detect outliers with Z-score (threshold=3)
    z_scores = np.abs(stats.zscore(df_subset))
    outliers_z = (z_scores > 3)

    print("Outliers detected by Z-score (per column):")
    print(pd.DataFrame(outliers_z, columns=cols).sum())

    # 3. Detect outliers with IQR method
    Q1 = df_subset.quantile(0.25)
    Q3 = df_subset.quantile(0.75)
    IQR = Q3 - Q1

    outliers_iqr = ((df_subset < (Q1 - 1.5 * IQR)) | (df_subset > (Q3 + 1.5 * IQR)))

    print("\nOutliers detected by IQR method (per column):")
    print(outliers_iqr.sum())


    # Household size
    sns.boxplot(x=df['household_size'])
    plt.title('Boxplot of Household Size')
    plt.show()

    # Age
    sns.boxplot(x=df['age_of_respondent'])
    plt.title('Boxplot of Age of Respondent')
    plt.show()


    # Handle outliers
    df_filtered = df[df['household_size'] <= 20].copy()
    print(f"Data shape before removal: {df.shape}")
    print(f"Data shape after removal: {df_filtered.shape}")


    # Handling of Outliers

    '''1.   Age of Respondent
    The outliers detected in the age_of_respondent column were retained because they fall within 
    a realistic and plausible range. 
    For example, respondents aged between the minimum and maximum observed 
    ages (e.g., 16 to 100) are possible in the dataset context, 
    reflecting actual survey participants of various ages. 
    Removing these could risk losing valid data points and bias the analysis.
    
    2.   Household Size
    In contrast, outliers in the household_size column were removed because 
    they represented extremely rare or potentially erroneous values (e.g., household size of 21).
     Such extreme values could disproportionately influence statistical analysis and model training.
      Given there was only a single or very few such outliers, removing them was considered 
      the best approach to maintain data integrity without significant data loss.
    '''
    # 2.6 Encode categorical features

    # Label encode education_level once
    le = LabelEncoder()
    df_filtered['education_level_encoded'] = le.fit_transform(df_filtered['education_level'])
    joblib.dump(le, 'model/le_education_level.pkl')
    # Convert target column to 0/1 BEFORE one-hot encoding
    df_filtered['bank_account'] = df_filtered['bank_account'].map({'No': 0, 'Yes': 1})

    # Then one-hot encode other categorical columns EXCEPT education_level and bank_account
    df_encoded = pd.get_dummies(df_filtered, columns=[
        'country', 'location_type', 'cellphone_access',
        'gender_of_respondent', 'relationship_with_head', 'marital_status',
        'job_type'], drop_first=True)
    df_encoded.drop(columns=['education_level'], inplace=True)
    return df_encoded

# Preprocess the data
df_raw = load_data()
df = preprocess_input(df_raw.copy())

# 3. Based on the previous data exploration train and test a machine learning classifier
# Train and evaluate the model
def train_and_evaluate(df_encoded):
    # Set target and features
    target = 'bank_account'
    # Exclude 'uniqueid' column along with the target and original education_level
    X = df_encoded.drop(['bank_account', 'uniqueid'], axis=1)
    y = df_encoded['bank_account']

    # Step 5: Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Train a Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    clf = RandomForestClassifier(class_weight='balanced',random_state=42)
    clf.fit(X_train, y_train)

    # Step 7: Predict on test data
    y_pred = clf.predict(X_test)

    # Step 8: Evaluate the model
    print("🔍 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Account', 'Has Account']))

    print("📊 Metrics Summary:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    # Sauvegarder le modèle entraîné
    joblib.dump(clf, "model/financial_model.pkl", compress=3)
    # Save columns used
    joblib.dump(X.columns.tolist(), 'model/model_columns.pkl')
    print("✅ Model saved in model/financial_model.pkl")