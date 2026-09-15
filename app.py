import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# INSIDER THREAT DETECTION SYSTEM
# Real trained ML models + interactive prediction
# ============================================================

st.set_page_config(
    page_title="Insider Threat Detection",
)

# -----------------------------
# Load trained models
# -----------------------------

xgb_model = joblib.load("models/xgboost_model.pkl")
isolation_model = joblib.load("models/isolation_forest.pkl")
svm_model = joblib.load("models/one_class_svm.pkl")

FEATURES = joblib.load("models/features.pkl")

# -----------------------------
# Styling
# -----------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #9aa4b2;
}

.metric-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #161b22;
    border: 1px solid #30363d;
}

.result-normal {
    padding: 20px;
    border-radius: 12px;
    background-color: #123524;
    border: 1px solid #2ea043;
}

.result-suspicious {
    padding: 20px;
    border-radius: 12px;
    background-color: #3b2f0b;
    border: 1px solid #d29922;
}

.result-high {
    padding: 20px;
    border-radius: 12px;
    background-color: #3d1418;
    border: 1px solid #f85149;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title"> Insider Threat Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid AI/ML Security Monitoring using '
    'XGBoost + Isolation Forest + One-Class SVM</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Sidebar - User Input
# -----------------------------

st.sidebar.header(" Employee Activity")

employee_id = st.sidebar.text_input(
    "Employee ID",
    value="EMP-10023"
)

st.sidebar.markdown("### Enter behavioural activity")

login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0
)

file_access_frequency = st.sidebar.number_input(
    "File Access Frequency",
    min_value=0.0,
    max_value=200.0,
    value=10.0,
    step=1.0
)

file_modification_frequency = st.sidebar.number_input(
    "File Modification Frequency",
    min_value=0.0,
    max_value=200.0,
    value=3.0,
    step=1.0
)

usb_usage = st.sidebar.number_input(
    "USB Usage",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=1.0
)

external_email_ratio = st.sidebar.number_input(
    "External Email Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)

after_hours_ratio = st.sidebar.number_input(
    "After-Hours Activity Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.05
)

unique_files_accessed = st.sidebar.number_input(
    "Unique Files Accessed",
    min_value=0.0,
    max_value=500.0,
    value=10.0,
    step=1.0
)

web_activity = st.sidebar.number_input(
    "Web Activity",
    min_value=0.0,
    max_value=500.0,
    value=20.0,
    step=1.0
)

analyze = st.sidebar.button(
    "ANALYZE ACTIVITY",
    use_container_width=True
)

# -----------------------------
# Prediction function
# -----------------------------

def calculate_risk():

    data = pd.DataFrame([[
        login_frequency,
        file_access_frequency,
        file_modification_frequency,
        usb_usage,
        external_email_ratio,
        after_hours_ratio,
        unique_files_accessed,
        web_activity
    ]], columns=[
    "login_frequency",
    "file_access_frequency",
    "file_modification_frequency",
    "usb_usage",
    "external_email_ratio",
    "after_hours_ratio",
    "unique_files_accessed",
    "web_activity"
])

    # XGBoost probability
    xgb_probability = float(
        xgb_model.predict_proba(data)[0][1]
    )

    # Isolation Forest
    if_decision = float(
        isolation_model.decision_function(data)[0]
    )

    if_prediction = int(
        isolation_model.predict(data)[0]
    )

    # One-Class SVM
    svm_decision = float(
        svm_model.decision_function(data)[0]
    )

    svm_prediction = int(
        svm_model.predict(data)[0]
    )

    # Convert anomaly outputs into risk values
    # Negative decision values indicate anomalous behaviour.

    if_risk = 1 / (1 + np.exp(5 * if_decision))

    svm_risk = 1 / (1 + np.exp(5 * svm_decision))

    # Hybrid decision fusion
    hybrid_score = (
        0.50 * xgb_probability +
        0.30 * if_risk +
        0.20 * svm_risk
    )

    risk_score = int(
        np.clip(hybrid_score * 100, 0, 100)
    )

    # Decision
    if risk_score >= 70:
        decision = "HIGH RISK"
    elif risk_score >= 40:
        decision = "SUSPICIOUS"
    else:
        decision = "NORMAL"

    return {
        "data": data,
        "xgb_probability": xgb_probability,
        "if_decision": if_decision,
        "if_prediction": if_prediction,
        "if_risk": if_risk,
        "svm_decision": svm_decision,
        "svm_prediction": svm_prediction,
        "svm_risk": svm_risk,
        "hybrid_score": hybrid_score,
        "risk_score": risk_score,
        "decision": decision
    }


# -----------------------------
# Default screen
# -----------------------------

if not analyze:

    st.info(
        " Enter employee behavioural activity in the sidebar "
        "and click **ANALYZE ACTIVITY**."
    )

    st.subheader("How the system works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("###  XGBoost")
        st.write(
            "Supervised machine learning model that estimates "
            "the probability of insider activity."
        )

    with c2:
        st.markdown("###  Isolation Forest")
        st.write(
            "Detects unusual behavioural patterns by identifying "
            "observations that are isolated from normal activity."
        )

    with c3:
        st.markdown("###  One-Class SVM")
        st.write(
            "Models normal behavioural patterns and identifies "
            "deviations from them."
        )

    st.divider()

    st.subheader("Hybrid Decision Fusion")

    st.code(
        "Hybrid Risk = "
        "0.50 × XGBoost + "
        "0.30 × Isolation Forest + "
        "0.20 × One-Class SVM"
    )

else:

    result = calculate_risk()

    risk_score = result["risk_score"]
    decision = result["decision"]

    # -------------------------
    # Main result
    # -------------------------

    st.subheader("Detection Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
       st.metric(
           "Employee",
            employee_id
       )   

    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    with col3:
        st.metric(
            "XGBoost Probability",
            f"{result['xgb_probability'] * 100:.1f}%"
        )

    with col4:
        st.metric(
            "Decision",
            decision
        )

    # -------------------------
    # Alert
    # -------------------------

    if decision == "HIGH RISK":

        st.markdown(
            f"""
            <div class="result-high">
            <h2> HIGH RISK ACTIVITY DETECTED</h2>
            <p>Employee <b>{employee_id}</b> shows behavioural
            patterns strongly associated with insider threat activity.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif decision == "SUSPICIOUS":

        st.markdown(
            f"""
            <div class="result-suspicious">
            <h2> SUSPICIOUS ACTIVITY</h2>
            <p>Employee <b>{employee_id}</b> shows anomalous
            behavioural characteristics requiring investigation.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-normal">
            <h2> NORMAL ACTIVITY</h2>
            <p>Employee <b>{employee_id}</b> appears consistent
            with normal behavioural activity.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # -------------------------
    # ML Engine Results
    # -------------------------

    st.subheader("Hybrid Detection Engines")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("### XGBoost")

        st.metric(
            "Threat Probability",
            f"{result['xgb_probability'] * 100:.2f}%"
        )

        if result["xgb_probability"] >= 0.5:
            st.error("Threat pattern detected")
        else:
            st.success("Normal classification")

    with c2:

        st.markdown("###  Isolation Forest")

        st.metric(
            "Anomaly Risk",
            f"{result['if_risk'] * 100:.2f}%"
        )

        if result["if_prediction"] == -1:
            st.error("Anomalous behaviour")
        else:
            st.success("Normal behaviour")

    with c3:

        st.markdown("###  One-Class SVM")

        st.metric(
            "Anomaly Risk",
            f"{result['svm_risk'] * 100:.2f}%"
        )

        if result["svm_prediction"] == -1:
            st.error("Behavioural deviation")
        else:
            st.success("Normal behaviour")

    # -------------------------
    # Hybrid score
    # -------------------------

    st.divider()

    st.subheader(" Hybrid Risk Analysis")

    st.progress(
        risk_score / 100
    )

    st.write(
        f"**Combined Hybrid Risk Score: {risk_score}/100**"
    )

    st.caption(
        "The final score combines supervised classification "
        "and two independent anomaly-detection signals."
    )

    # -------------------------
    # Input activity table
    # -------------------------

    st.divider()

    st.subheader("Analysed Behavioural Features")

    display_data = pd.DataFrame({
        "Feature": FEATURES,
        "Entered Value": [
            login_frequency,
            file_access_frequency,
            file_modification_frequency,
            usb_usage,
            external_email_ratio,
            after_hours_ratio,
            unique_files_accessed,
            web_activity
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # Explanation
    # -------------------------

    st.divider()

    st.subheader(" Why was this activity flagged?")

    reasons = []

    if login_frequency > 15:
        reasons.append("Unusually high login frequency")

    if file_access_frequency > 30:
        reasons.append("High frequency of file access")

    if file_modification_frequency > 20:
        reasons.append("High file modification activity")

    if usb_usage > 5:
        reasons.append("Unusual USB activity")

    if external_email_ratio > 0.30:
        reasons.append("High external email communication")

    if unique_files_accessed > 50:
        reasons.append("Large number of unique files accessed")

    if web_activity > 80:
        reasons.append("Unusually high web activity")

    if reasons:

        for reason in reasons:
            st.warning("• " + reason)

    else:

        st.success(
            "No individual behavioural feature crossed "
            "the configured investigation thresholds."
        )

    # -------------------------
    # Technical information
    # -------------------------

    with st.expander("Technical Model Information"):

        st.write("**Features used:**")

        st.write(FEATURES)

        st.write("**Decision fusion:**")

        st.write(
            "XGBoost = 50% | "
            "Isolation Forest = 30% | "
            "One-Class SVM = 20%"
        )

        st.write(
            "Models loaded from the local trained model files."
        )

st.divider()

st.caption(
    "Insider Threat Detection System | Hybrid AI/ML Security Monitoring"
)