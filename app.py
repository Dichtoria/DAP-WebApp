import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ==========================
# ⚙️ CONFIG & ENV
# ==========================
st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")
st.title("💸 Salary Prediction & Data Analysis Dashboard")
st.markdown("#### 🧠 A Data Analysis Project by FPTU HCMC Students")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")

# ==========================
# 📂 LOAD DATA
# ==========================
@st.cache_data
def load_data(file_path="DATASET.xlsx"):
    return pd.read_excel(file_path)

uploaded_file = st.sidebar.file_uploader("📤 Upload a dataset (xlsx/csv)", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    df = load_data()

# ==========================
# 🧭 NAVIGATION
# ==========================
menu = st.sidebar.radio("📂 Navigation", ["Dashboard", "Chatbot", "About", "Prediction"])

# ==========================
# 📊 DASHBOARD PAGE
# ==========================
if menu == "Dashboard":
    st.subheader("📊 Explore Salary Prediction Factors")

    # ---- FILTERS ----
    st.sidebar.markdown("### 🔎 Filters")

    filter_columns = [col for col in df.columns if df[col].dtype == "object"]
    df_filtered = df.copy()
    for col in filter_columns:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) > 1 and len(unique_vals) < 25:  # tránh quá nhiều lựa chọn
            selected = st.sidebar.multiselect(f"{col}", unique_vals)
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

    # ---- METRICS ----
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", len(df_filtered))
    with col2:
        st.metric("Average GPA", round(df_filtered["GPA"].mean(), 2))
    with col3:
        st.metric("Avg Salary Expectation", f"{int(df_filtered['SALARY_EXPECT'].mean()):,} VND")

    st.divider()
    st.markdown("### 📈 Salary Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(df_filtered, x="MAJOR", y="SALARY_EXPECT", color="GENDER",
                      title="Salary Distribution by Major & Gender")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(df_filtered, x="AFTER_GRAD", y="SALARY_EXPECT", color="AFTER_GRAD",
                      title="Average Salary by Career Plan", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### 🎯 Factors Affecting Salary Expectation")

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.scatter(df_filtered, x="GPA", y="SALARY_EXPECT", color="MAJOR",
                          size="CODE_LEVEL", hover_data=["STRENGTH"],
                          title="Salary vs GPA (colored by Major)")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.box(df_filtered, x="STRENGTH", y="SALARY_EXPECT", color="STRENGTH",
                      title="Salary vs Skill Strength")
        st.plotly_chart(fig4, use_container_width=True)

    # ---- HEATMAP ----
    st.divider()
    st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
    num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
    corr = df_filtered[num_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # ---- DOWNLOAD ----
    st.divider()
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Dataset", csv, "filtered_salary_data.csv", "text/csv")

# ==========================
# 🤖 PREDICTION PAGE
# ==========================
elif menu == "Prediction":
    st.subheader("🎯 Predict Suitable Career Path")

    st.markdown("#### 🔧 Enter Your Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", df["GENDER"].unique())
        major = st.selectbox("Major", df["MAJOR"].unique())
        gpa = st.slider("GPA", 0.0, 10.0, 7.5, 0.1)
        salary = st.number_input("Expected Salary (VND)", 0, 50000000, 15000000, 1000000)
    with col2:
        strength = st.selectbox("Strength", df["STRENGTH"].unique())
        after_grad = st.selectbox("After Graduation Plan", df["AFTER_GRAD"].unique())
        code_lvl = st.slider("Coding Skill Level (1–5)", 1, 5, 3)
        job_factor = st.selectbox("Main Job Factor", df["JOB_FACTOR_MAPPED"].unique())

    if st.button("🚀 Predict Career Path"):
        if "AI" in major or "Trí tuệ" in major:
            prediction = "Dữ liệu & Trí tuệ nhân tạo"
        elif "phần mềm" in major.lower():
            prediction = "Phát triển phần mềm"
        else:
            prediction = "Kinh doanh & Phân tích dữ liệu"

        st.success(f"✅ Suitable Career Path: **{prediction}**")
        st.info("*(Note: This is demo logic — replace with trained ML model for final version)*")

# ==========================
# 💬 CHATBOT PAGE (Gemini)
# ==========================
elif menu == "Chatbot":
    st.subheader("💬 Gemini Career Assistant")
    st.markdown("#### Ask me anything about salaries, majors, or data insights!")

    if not api_key:
        st.error("❌ Gemini API key not found. Please configure .env file first.")
    else:
        model = genai.GenerativeModel("gemini-2.0-flash")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Type your question here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state["messages"].append({"role": "user", "content": prompt})

            try:
                response = model.generate_content(prompt)
                reply = response.text
            except Exception as e:
                reply = f"⚠️ Gemini API error: {e}"

            st.chat_message("assistant").markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})

# ==========================
# 📘 ABOUT PAGE
# ==========================
else:
    st.subheader("📘 About This Project")
    st.markdown("""
    **Project name:** *Salary Prediction Analysis of FPTU IT Students*  
    **Course:** Data Analysis Project (DAP)  
    **Objective:**  
    - Analyze factors affecting salary expectation of IT students  
    - Visualize key patterns and correlations  
    - Integrate Gemini chatbot for interactive data insights  
    ---
    **Team members:**  
    - 👩‍💻 Member 1 – Data Cleaning & Visualization  
    - 👨‍💻 Member 2 – Model Training & Salary Prediction  
    - 🧑‍💻 Member 3 – Web App & Deployment  
    ---
    Built with ❤️ using **Streamlit + Plotly + Gemini API**  
    """)
    #st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
