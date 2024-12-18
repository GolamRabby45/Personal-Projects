import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    data_path = "../data/creditcard.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file not found. Please ensure creditcard.csv is in the data directory.")

    df = load_data(data_path)

    # Basic info
    print("Data Info:")
    print(df.info())

    print("\nData Description:")
    print(df.describe())

    # Class distribution
    class_counts = df['Class'].value_counts()
    print("\nClass Distribution:")
    print(class_counts)

    # Plot class distribution
    plt.figure(figsize=(6,4))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xticks([0,1], ['Non-Fraud', 'Fraud'])
    plt.ylabel("Count")
    plt.savefig("../outputs/class_distribution.png")
    plt.close()

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    # Histograms of a few features
    features_to_plot = ['V1','V2','V3','Amount']
    df[features_to_plot].hist(bins=30, figsize=(10,7))
    plt.tight_layout()
    plt.savefig("../outputs/feature_histograms.png")
    plt.close()

    # Correlation matrix (optional, since many are PCA components)
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Correlation Matrix")
    plt.savefig("../outputs/correlation_matrix.png")
    plt.close()

    print("EDA Completed. Check the outputs directory for plots.")
