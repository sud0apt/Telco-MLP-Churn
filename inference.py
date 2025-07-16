import streamlit as st
import torch
import pandas as pd
from model import MLP
from dataset import TelcoDataset, get_cat_dims
import config

# Load dataset to reuse encoders/scaler
df = pd.read_csv("telco-churm.csv")
dataset = TelcoDataset(df, config.categorical_cols, config.numerical_cols, config.target_col)

# Load model
cat_dims = get_cat_dims(df, config.categorical_cols)
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]
model = MLP(cat_dims, emb_dims, len(config.numerical_cols), config.hidden_dims, config.output_dim)
model.load_state_dict(torch.load("mlp_telco_churn.pt"))
model.eval()

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Form", "Table Info", "Training Results"])

# ---- Header (Logo + Title) ----
st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d6/CelcomDigi_Logo.svg/800px-CelcomDigi_Logo.svg.png", width=600)
st.markdown(
    "<h1 style='text-align: center; color: white;'>Churn Predictor</h1>",
    unsafe_allow_html=True
)

# ---- TAB 1: FORM ----
if tab == "Form":
    with st.form("input_form"):
        gender = st.selectbox("Gender", dataset.label_encoders["gender"].classes_)
        senior = st.selectbox("Senior Citizen", dataset.label_encoders["SeniorCitizen"].classes_)
        partner = st.selectbox("Partner", dataset.label_encoders["Partner"].classes_)
        dependents = st.selectbox("Dependents", dataset.label_encoders["Dependents"].classes_)
        phone_service = st.selectbox("Phone Service", dataset.label_encoders["PhoneService"].classes_)
        multiple_lines = st.selectbox("Multiple Lines", dataset.label_encoders["MultipleLines"].classes_)
        internet_service = st.selectbox("Internet Service", dataset.label_encoders["InternetService"].classes_)
        online_security = st.selectbox("Online Security", dataset.label_encoders["OnlineSecurity"].classes_)
        online_backup = st.selectbox("Online Backup", dataset.label_encoders["OnlineBackup"].classes_)
        device_protection = st.selectbox("Device Protection", dataset.label_encoders["DeviceProtection"].classes_)
        tech_support = st.selectbox("Tech Support", dataset.label_encoders["TechSupport"].classes_)
        streaming_tv = st.selectbox("Streaming TV", dataset.label_encoders["StreamingTV"].classes_)
        streaming_movies = st.selectbox("Streaming Movies", dataset.label_encoders["StreamingMovies"].classes_)
        contract = st.selectbox("Contract", dataset.label_encoders["Contract"].classes_)
        paperless_billing = st.selectbox("Paperless Billing", dataset.label_encoders["PaperlessBilling"].classes_)
        payment_method = st.selectbox("Payment Method", dataset.label_encoders["PaymentMethod"].classes_)

        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        # Encode categoricals
        for col in config.categorical_cols:
            user_df[col] = dataset.label_encoders[col].transform(user_df[col])

        # Scale numericals
        user_df[config.numerical_cols] = dataset.scaler.transform(user_df[config.numerical_cols])

        x_cat = torch.tensor(user_df[config.categorical_cols].values, dtype=torch.long)
        x_num = torch.tensor(user_df[config.numerical_cols].values, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = model(x_cat, x_num)
            prob = torch.softmax(output, dim=1)[0][1].item()
            pred = int(prob > 0.5)

        if pred == 1:
            st.error(f"🔴 Prediction: Churn (Probability: {prob:.2%})")
        else:
            st.success(f"🟢 Prediction: No Churn (Probability: {prob:.2%})")

         # Debug display
        st.subheader("Input Diagnostics")
        st.write("Raw input:")
        st.write(user_df)

        st.write("Scaled numerical input:")
        st.write(pd.DataFrame(x_num.numpy(), columns=config.numerical_cols))

        st.write("Encoded categorical input:")
        st.write(pd.DataFrame(x_cat.numpy(), columns=config.categorical_cols))

# ---- TAB 2: TABLE INFO ----
elif tab == "Table Info":
    st.subheader("Full Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Column Types")
    st.write(df.dtypes)

    st.subheader("Categorical Columns & Classes")
    for col in config.categorical_cols:
        st.write(f"**{col}**: {df[col].nunique()} unique values → {df[col].unique()}")

    st.subheader("📊 Explore Column Stats")
    selected_col = st.selectbox("Select a column to view stats", df.columns)

    if selected_col:
        st.markdown(f"### Stats for `{selected_col}`")

        if df[selected_col].dtype in ['int64', 'float64']:
            st.write("**Mean:**", df[selected_col].mean())
            st.write("**Median:**", df[selected_col].median())
            st.write("**Standard Deviation:**", df[selected_col].std())
            st.write("**Min:**", df[selected_col].min())
            st.write("**Max:**", df[selected_col].max())
        else:
            st.write("**Unique Classes:**", df[selected_col].nunique())
            st.write("**Mode:**", df[selected_col].mode().values)
            st.write("**Value Counts:**")
            st.dataframe(df[selected_col].value_counts().reset_index(names=['Value', 'Count']))

# ---- TAB 3: TRAINING RESULTS ----
elif tab == "Training Results":
    st.subheader("Model Summary")
    st.text(model)

    try:
        df_log = pd.read_csv("training_metrics.csv")
        st.subheader("Training Metrics")

        # Show table
        st.dataframe(df_log)

        # Line charts
        st.subheader("Train vs Test Accuracy")
        if "Train_Accuracy" in df_log.columns and "Test_Accuracy" in df_log.columns:
            acc_plot = df_log[["Epoch", "Train_Accuracy", "Test_Accuracy"]].set_index("Epoch")
            st.line_chart(acc_plot, use_container_width=True)
        else:
            st.warning("Train/Test Accuracy not available in log. Re-train to include both.")


    except FileNotFoundError:
        st.warning("No training_metrics.csv file found. Please train the model first.")

