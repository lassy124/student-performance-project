import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="🎓 Student Performance Prediction", layout="wide")

st.title("🎓 Student Performance Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your Student Performance CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Show summary
    if st.checkbox("Show Data Summary"):
        st.write(df.describe())

    # Visualization
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Model Training
    target = st.selectbox("Select Target Column (what you want to predict)", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

    if st.button("Train Model"):
        X = df[features]
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        st.success("✅ Model trained successfully!")
        st.write("📈 R2 Score:", r2_score(y_test, preds))
        st.write("📉 RMSE:", mean_squared_error(y_test, preds, squared=False))
        st.write("📉 MAE:", mean_absolute_error(y_test, preds))
