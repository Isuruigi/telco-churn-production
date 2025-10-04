import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE, REPORTS_PATH
from src.data_loader import TelcoDataLoader
from src.utils import create_directories

FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'advanced_eda')
create_directories([FIGURES_PATH])

def perform_univariate_analysis(df: pd.DataFrame, column: str):
    """Performs univariate analysis on a single column."""
    print(f"--- Univariate Analysis of {column} ---")
    if column in NUMERICAL_FEATURES:
        print(df[column].describe())
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(os.path.join(FIGURES_PATH, f'{column}_univariate_dist.png'))
        plt.close()
    elif column in CATEGORICAL_FEATURES:
        print(df[column].value_counts(normalize=True))
        plt.figure(figsize=(10, 6))
        sns.countplot(y=column, data=df, order=df[column].value_counts().index)
        plt.title(f'Distribution of {column}')
        plt.savefig(os.path.join(FIGURES_PATH, f'{column}_univariate_dist.png'))
        plt.close()
    else:
        print(f"Column {column} not found in numerical or categorical feature lists.")
    print(f"--- End Univariate Analysis of {column} ---")

def perform_bivariate_analysis(df: pd.DataFrame, col1: str, col2: str):
    """Performs bivariate analysis on two columns."""
    print(f"--- Bivariate Analysis of {col1} and {col2} ---")
    if col1 in NUMERICAL_FEATURES and col2 in NUMERICAL_FEATURES:
        sns.scatterplot(x=col1, y=col2, data=df)
        plt.title(f'{col1} vs. {col2}')
        plt.savefig(os.path.join(FIGURES_PATH, f'{col1}_vs_{col2}_scatter.png'))
        plt.close()
    elif col1 in CATEGORICAL_FEATURES and col2 in NUMERICAL_FEATURES:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col1, y=col2, data=df)
        plt.title(f'{col2} by {col1}')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(FIGURES_PATH, f'{col2}_by_{col1}_boxplot.png'))
        plt.close()
    elif col1 in NUMERICAL_FEATURES and col2 in CATEGORICAL_FEATURES:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col2, y=col1, data=df)
        plt.title(f'{col1} by {col2}')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(FIGURES_PATH, f'{col1}_by_{col2}_boxplot.png'))
        plt.close()
    elif col1 in CATEGORICAL_FEATURES and col2 in CATEGORICAL_FEATURES:
        pd.crosstab(df[col1], df[col2]).plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'{col1} vs. {col2}')
        plt.savefig(os.path.join(FIGURES_PATH, f'{col1}_vs_{col2}_stackedbar.png'))
        plt.close()
    print(f"--- End Bivariate Analysis of {col1} and {col2} ---")

def perform_multivariate_analysis(df: pd.DataFrame):
    """Performs multivariate analysis using a pair plot."""
    print("--- Multivariate Analysis (Pair Plot) ---")
    pairplot = sns.pairplot(df[NUMERICAL_FEATURES + [TARGET_COLUMN]], hue=TARGET_COLUMN)
    pairplot.savefig(os.path.join(FIGURES_PATH, 'multivariate_pairplot.png'))
    plt.close()
    print("--- End Multivariate Analysis ---")

def analyze_class_imbalance(df: pd.DataFrame):
    """Analyzes and reports the class imbalance of the target variable."""
    print("--- Class Imbalance Analysis ---")
    target_counts = df[TARGET_COLUMN].value_counts()
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
    print(target_counts)
    print("--- End Class Imbalance Analysis ---")

def create_customer_segments(df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    """Creates customer segments using KMeans clustering."""
    print("--- Customer Segmentation ---")
    cluster_features = ['tenure', 'MonthlyCharges']
    df_cluster = df[cluster_features].copy()
    df_cluster.dropna(inplace=True)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['Segment'] = kmeans.fit_predict(df_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Segment', data=df, palette='viridis')
    plt.title('Customer Segments')
    plt.savefig(os.path.join(FIGURES_PATH, 'customer_segments.png'))
    plt.close()
    print("--- End Customer Segmentation ---")
    return df

def generate_business_insights(df_segmented: pd.DataFrame):
    """Generates business insights from segmented customer data."""
    print("--- Business Insights from Segmentation ---")
    # Ensure target column is in the correct format for aggregation
    df_segmented['Churn_numeric'] = (df_segmented[TARGET_COLUMN] == 'Yes').astype(int)
    
    segment_analysis = df_segmented.groupby('Segment').agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'Churn_numeric': 'mean'
    }).rename(columns={'Churn_numeric': 'ChurnRate'})
    
    print("Segment Analysis:")
    print(segment_analysis)
    print("\nInsights:")
    for i, row in segment_analysis.iterrows():
        print(f"- Segment {i}: Average tenure of {row.tenure:.1f} months, average monthly charge of ${row.MonthlyCharges:.2f}, and a churn rate of {row.ChurnRate:.2%}.")
    print("--- End Business Insights ---")

if __name__ == '__main__':
    try:
        loader = TelcoDataLoader()
        churn_df = loader.load_raw_data()

        if churn_df is not None:
            churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce')
            churn_df.dropna(subset=['TotalCharges'], inplace=True)

            perform_univariate_analysis(churn_df, 'tenure')
            perform_bivariate_analysis(churn_df, 'Contract', 'Churn')
            analyze_class_imbalance(churn_df)
            
            segmented_df = create_customer_segments(churn_df)
            generate_business_insights(segmented_df)
    except ImportError as e:
        print(f"Error: {e}. Please run this script from the root of the project.")