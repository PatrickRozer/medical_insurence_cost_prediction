import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")
st.write("This page provides a full EDA of the Medical Insurance dataset.")

@st.cache_data
def load_data():
    return pd.read_csv(r"C:/Users/Bernietta/OneDrive/guvi/guvi_project/project_3/medical_insurance (1).csv")

df = load_data()

# -------------------------------------------
# SIDEBAR
# -------------------------------------------
st.sidebar.header("⚙️ EDA Controls")
show_raw = st.sidebar.checkbox("Show Raw Dataset", False)

if show_raw:
    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

# -------------------------------------------
# UNIVARIATE ANALYSIS
# -------------------------------------------
st.header("🔍 1. Univariate Analysis")

# Charges Distribution
st.subheader("💵 Distribution of Medical Insurance Charges")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['charges'], kde=True, ax=ax)
st.pyplot(fig)

# Age Distribution
st.subheader("👤 Age Distribution")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Smokers Count
st.subheader("🚬 Smokers vs Non-Smokers")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(data=df, x='smoker', ax=ax)
st.pyplot(fig)

# BMI Avg
st.subheader("⚖️ BMI Distribution")
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df["bmi"], kde=True, ax=ax)
st.pyplot(fig)

# Regions Count
st.subheader("🌍 Region Distribution")
fig, ax = plt.subplots(figsize=(7,4))
sns.countplot(data=df, x='region', ax=ax)
st.pyplot(fig)

# -------------------------------------------
# BIVARIATE ANALYSIS
# -------------------------------------------
st.header("🔍 2. Bivariate Analysis")

# Charges vs Age
st.subheader("📈 Charges vs Age")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='age', y='charges', ax=ax)
st.pyplot(fig)

# Smoker vs Charges
st.subheader("🚬 Smoker vs Insurance Charges")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=df, x='smoker', y='charges', ax=ax)
st.pyplot(fig)

# BMI impact
st.subheader("⚖️ BMI vs Charges (Colored by Smoker)")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', ax=ax)
st.pyplot(fig)

# Gender vs Charges
st.subheader("⚧ Gender vs Insurance Charges")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=df, x='sex', y='charges', ax=ax)
st.pyplot(fig)

# Children vs Charges
st.subheader("👶 Number of Children vs Charges")
fig, ax = plt.subplots(figsize=(8,5))
sns.stripplot(data=df, x='children', y='charges', ax=ax)
st.pyplot(fig)

# -------------------------------------------
# MULTIVARIATE ANALYSIS
# -------------------------------------------
st.header("🔍 3. Multivariate Analysis")

# Age + Smoking + Charges
st.subheader("🔥 Age & Smoking Impact on Charges")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker', ax=ax)
st.pyplot(fig)

# Region + Gender + Smokers
st.subheader("🌍 Region & Gender Impact Among Smokers")
smokers = df[df['smoker']=='yes']
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=smokers, x='region', y='charges', hue='sex', ax=ax)
st.pyplot(fig)

# Age + BMI + Smoking
st.subheader("⚖️ Age, BMI & Smoking Combined")
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='bmi', y='charges', size='age', hue='smoker', ax=ax, sizes=(20,200))
st.pyplot(fig)

# -------------------------------------------
# OUTLIERS
# -------------------------------------------
st.header("🔍 4. Outlier Detection")

# Charges Boxplot
st.subheader("💵 Charge Outliers")
fig, ax = plt.subplots(figsize=(7,4))
sns.boxplot(data=df, x='charges', ax=ax)
st.pyplot(fig)

# BMI Outliers
st.subheader("⚖️ BMI Outliers")
fig, ax = plt.subplots(figsize=(7,4))
sns.boxplot(data=df, x='bmi', ax=ax)
st.pyplot(fig)

# -------------------------------------------
# CORRELATION
# -------------------------------------------
st.header("🔍 5. Correlation Analysis")

st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.success("EDA Dashboard Loaded Successfully 🎉")
