import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define segment explanations (customize as needed based on cluster profiling)
segment_explanations = {
    0: "🟢 **Segment 0**: Young or moderate-age customers with high income and moderate-to-high spending. Great candidates for premium loyalty programs.",
    1: "🟣 **Segment 1**: Older individuals with high income but low spending. Consider promotions or special offers to increase engagement.",
    2: "🟡 **Segment 2**: Young customers with low income and very high spending. Focused on value-for-money offerings, possibly students or budget-conscious millennials.",
    3: "🔵 **Segment 3**: Average income and average spending behavior. Mass marketing or general seasonal campaigns work well here.",
    4: "🔴 **Segment 4**: Low-income, low-spending customers. Might not be actively engaged. Push basic offers or cost-effective services."
}

st.title("🧠 Mall Customer Cluster Predictor")
st.markdown("Enter customer details to predict their segment and understand why they fall into that group.")

# Input fields
age = st.number_input("Age", min_value=15, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=60)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Predict button
if st.button("Predict Segment"):
    input_data = np.array([[age, income, score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Customer Segment: **{prediction}**")
    
    # Explanation panel
    with st.expander("📋 Why this segment?"):
        st.markdown(segment_explanations.get(prediction, "No details available for this segment."))
