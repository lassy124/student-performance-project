import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("🎓 Student Performance Prediction App")

uploaded_file = st.file_uploader("📂 Upload your student dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include="all"))

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------
    if st.checkbox("📈 Show Correlation Heatmap"):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough numeric columns for correlation heatmap.")

    # ----------------------------
    # Feature & Target Selection
    # ----------------------------
    st.subheader("⚙️ Model Training")
    features = st.multiselect("Select Features (X)", df.columns.tolist())
    target = st.selectbox("Select Target (y)", df.columns.tolist())

    if features and target:
        X = df[features]
        y = df[target]

        # 🔹 Convert categorical features to numeric
        X = pd.get_dummies(X, drop_first=True)

        # 🔹 Convert target to numeric if needed
        if y.dtype == "object" or not np.issubdtype(y.dtype, np.number):
            y = pd.factorize(y)[0]

        # 🚨 Ensure no NaN values
        X = X.fillna(0)
        y = pd.Series(y).fillna(0)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📉 Model Performance")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
        rmse = mean_squared_error(y_test, y_pred, squared=True) ** 0.5
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")

        # Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to continue.")


