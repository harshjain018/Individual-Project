import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency, chisquare
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
st.title("Imports and Exports Dataset Analysis")
dataset = pd.read_csv("Imports_Exports_Dataset.csv")

# Sidebar options
st.sidebar.title("Options")
sample_size = st.sidebar.slider("Select Sample Size", min_value=500, max_value=len(dataset), value=3001)
sample = dataset.sample(n=sample_size, random_state=55018)

# Numeric and categorical columns
ncat = sample[['Quantity', 'Value', 'Weight']]
cat = sample[['Country', 'Product', 'Import_Export', 'Category', 'Customs_Code', 'Shipping_Method', 'Supplier', 'Customer', 'Payment_Terms']]

# Display dataset preview
st.write("Dataset Preview:")
st.dataframe(sample.head())

# Boxplots for numerical data
st.subheader("Box Plots")
for col in ncat.columns:
    st.write(f"Box Plot of {col}")
    fig, ax = plt.subplots()
    ncat[col].plot(kind='box', showmeans=True, meanline=True, vert=False, ax=ax)
    st.pyplot(fig)

# Heatmap of correlation matrix
st.subheader("Correlation Matrix Heatmap")
correlation_matrix = ncat.corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Scatter plots
st.subheader("Scatter Plots")
scatter_columns = st.sidebar.multiselect("Select columns for Scatter Plot", ncat.columns, default=['Quantity', 'Weight'])
if len(scatter_columns) == 2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=ncat[scatter_columns[0]], y=ncat[scatter_columns[1]], ax=ax)
    st.pyplot(fig)

# Function to calculate and visualize statistics
def calculate_and_visualize(column_name):
    range_value = ncat[column_name].max() - ncat[column_name].min()
    std_dev = ncat[column_name].std()
    skewness = stats.skew(ncat[column_name])
    kurt = stats.kurtosis(ncat[column_name])
    coeff_var = std_dev / ncat[column_name].mean()

    st.write(f"### Stats for {column_name}")
    st.write(f"Range: {range_value}")
    st.write(f"Standard Deviation: {std_dev}")
    st.write(f"Skewness: {skewness}")
    st.write(f"Kurtosis: {kurt}")
    st.write(f"Coefficient of Variation: {coeff_var}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(ncat[column_name], ax=axes[0], kde=True)
    stats.probplot(ncat[column_name], plot=axes[1])
    st.pyplot(fig)

# Allow user to select column for detailed analysis
st.subheader("Detailed Analysis")
col_to_analyze = st.sidebar.selectbox("Select column for detailed analysis", ncat.columns)
calculate_and_visualize(col_to_analyze)

# Function for categorical analysis
def analyze_categorical_column(column_name):
    st.write(f"### Analysis of {column_name}")
    value_counts = cat[column_name].value_counts()
    st.write("Value Counts:")
    st.write(value_counts)

    # Pie or bar chart based on the number of categories
    fig, ax = plt.subplots()
    if len(value_counts) <= 10:
        value_counts.plot.pie(autopct='%1.1f%%', ax=ax)
    else:
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        plt.xticks(rotation=90)
    st.pyplot(fig)

# Categorical analysis
st.subheader("Categorical Analysis")
cat_col_to_analyze = st.sidebar.selectbox("Select categorical column for analysis", cat.columns)
analyze_categorical_column(cat_col_to_analyze)

# Time series analysis
if 'Date' in sample.columns:
    st.subheader("Time Series Analysis")
    sample['Date'] = pd.to_datetime(sample['Date'], errors='coerce')
    transactions_over_time = sample.groupby(sample['Date'].dt.to_period('M'))['Transaction_ID'].count()

    fig, ax = plt.subplots()
    transactions_over_time.plot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Allow user to display scatter plot with categories
st.subheader("Scatter Plot with Categories")
if st.sidebar.checkbox("Show Scatter Plot with Categories"):
    hue_column = st.sidebar.selectbox("Select a category for hue", cat.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x='Quantity', y='Value', hue=hue_column, data=sample, ax=ax)
    st.pyplot(fig)

st.write("## End of Analysis")

