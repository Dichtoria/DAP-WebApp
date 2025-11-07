import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np

# ==========================
# ⚙️ CONFIG & ENV
# ==========================
st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")
st.title("💸 Salary Prediction & Data Analysis Dashboard")
st.markdown("#### 🧠 A Data Analysis Project by FPTU HCMC Students")

# Load Gemini API
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

df = load_data()

# ==========================
# 📦 LOAD MODEL
# ==========================
# @st.cache_resource
# def load_model():
#     with open("salary_model.pkl", "rb") as file:
#         model = pickle.load(file)
#     return model
#
# model = load_model()

# ==========================
# 🧭 NAVIGATION
# ==========================
menu = st.sidebar.radio("📂 Navigation", ["Dashboard", "Prediction", "Chatbot", "About"])

# ==========================
# 📊 DASHBOARD PAGE (Full Fixed)
# ==========================
if menu == "Dashboard":
    st.subheader("📊 Explore Student Career & Salary Insights")

    # --- Clean numeric columns ---
    for col in ["GPA", "SALARY_EXPECT", "CODE_LEVEL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_filtered = df.copy()

    # --- Sidebar Filters ---
    st.sidebar.markdown("### 🔎 Filters")

    filter_columns = [
        "GENDER",
        "MAJOR",
        "AFTER_GRAD",
        "COUNTRY_PREF",
        "WORK_STYLE",
        "STRENGTH",
        "JOB_FACTOR_MAPPED",
        "JOB_ASPIRATION_GROUPED"
    ]

    for col in filter_columns:
        if col in df.columns:
            opts = sorted([str(x) for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])
            picked = st.sidebar.multiselect(f"{col}", opts)
            if picked:
                df_filtered[col] = df_filtered[col].astype(str)
                df_filtered = df_filtered[df_filtered[col].isin(picked)]

    # --- Numeric range sliders ---
    col1, col2 = st.sidebar.columns(2)
    if "GPA" in df.columns:
        gmin, gmax = float(df["GPA"].min()), float(df["GPA"].max())
        g_lo, g_hi = col1.slider("GPA range", min_value=gmin, max_value=gmax, value=(gmin, gmax), step=0.1)
        df_filtered = df_filtered[(df_filtered["GPA"] >= g_lo) & (df_filtered["GPA"] <= g_hi)]

    if "SALARY_EXPECT" in df.columns:
        smin, smax = int(df["SALARY_EXPECT"].min()), int(df["SALARY_EXPECT"].max())
        s_lo, s_hi = col2.slider("Salary range (VND)", min_value=smin, max_value=smax, value=(smin, smax), step=500000)
        df_filtered = df_filtered[(df_filtered["SALARY_EXPECT"] >= s_lo) & (df_filtered["SALARY_EXPECT"] <= s_hi)]

    # --- Check for empty dataset ---
    if df_filtered.empty:
        st.warning("⚠️ No data available for the selected filters.")
        st.stop()

    # --- KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Responses", len(df_filtered))
    with k2:
        if "GPA" in df_filtered:
            st.metric("Average GPA", f"{df_filtered['GPA'].mean():.2f}")
    with k3:
        if "SALARY_EXPECT" in df_filtered:
            st.metric("Average Salary", f"{int(df_filtered['SALARY_EXPECT'].mean()):,} VND")
    with k4:
        if "MAJOR" in df_filtered:
            st.metric("Number of Majors", df_filtered["MAJOR"].nunique())

    st.divider()

    # --- Section 1: General Distributions ---
    # ==========================
    # 🌍 GENERAL DISTRIBUTION (Auto Visualization for all columns)
    # ==========================
    st.markdown("### 🌍 General Distribution")

    # Lặp qua tất cả các cột
    for col in df_filtered.columns:
        # Bỏ qua cột có quá nhiều giá trị duy nhất
        if df_filtered[col].nunique() > 25:
            continue

        st.markdown(f"#### 📊 Distribution of `{col}`")

        # Xử lý theo kiểu dữ liệu
        if df_filtered[col].dtype == "object" or df_filtered[col].dtype.name == "category":
            value_counts = df_filtered[col].value_counts().reset_index()
            value_counts.columns = [col, "Count"]

            # Nếu có ít hơn hoặc bằng 5 giá trị → dùng Pie chart
            if len(value_counts) <= 5:
                fig = px.pie(
                    value_counts,
                    names=col,
                    values="Count",
                    title=f"Distribution of {col}",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
            else:
                fig = px.bar(
                    value_counts,
                    x=col,
                    y="Count",
                    color=col,
                    title=f"Distribution of {col}",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

        else:
            # Numeric → Histogram
            fig = px.histogram(
                df_filtered,
                x=col,
                nbins=20,
                title=f"Distribution of {col}",
                color_discrete_sequence=["#2E86C1"]
            )

        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # --- Section 2: Salary Focus ---
    st.markdown("### 💰 Salary Focus")
    c3, c4 = st.columns(2)

    if {"MAJOR", "SALARY_EXPECT", "GENDER"}.issubset(df_filtered.columns):
        with c3:
            fig_box = px.box(df_filtered, x="MAJOR", y="SALARY_EXPECT", color="GENDER",
                             title="Salary Expectation by Major & Gender")
            st.plotly_chart(fig_box, use_container_width=True)

    if {"AFTER_GRAD", "SALARY_EXPECT"}.issubset(df_filtered.columns):
        with c4:
            avg_salary = df_filtered.groupby("AFTER_GRAD", as_index=False)["SALARY_EXPECT"].mean()
            fig_avg = px.bar(avg_salary, x="AFTER_GRAD", y="SALARY_EXPECT",
                             title="Average Salary by After-Graduation Plan", color="AFTER_GRAD")
            st.plotly_chart(fig_avg, use_container_width=True)

    st.divider()

    # --- Section 3: Relationships ---
    st.markdown("### 🎯 Relationships & Patterns")

    # --- Row 1: GPA vs Salary, Strength vs Salary ---
    c5, c6 = st.columns(2)

    if {"GPA", "SALARY_EXPECT", "MAJOR"}.issubset(df_filtered.columns):
        with c5:
            fig_scatter = px.scatter(
                df_filtered,
                x="GPA",
                y="SALARY_EXPECT",
                color="MAJOR",
                size="CODE_LEVEL" if "CODE_LEVEL" in df_filtered.columns else None,
                hover_data=["STRENGTH"] if "STRENGTH" in df_filtered.columns else None,
                title="Salary vs GPA by Major",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    if {"STRENGTH", "SALARY_EXPECT"}.issubset(df_filtered.columns):
        with c6:
            fig_strength = px.box(
                df_filtered,
                x="STRENGTH",
                y="SALARY_EXPECT",
                color="STRENGTH",
                title="Salary by Strength",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_strength, use_container_width=True)

    st.divider()

    # --- Row 2: Salary vs Country & Job Factor ---
    c7, c8 = st.columns(2)

    if {"COUNTRY_PREF", "SALARY_EXPECT"}.issubset(df_filtered.columns):
        with c7:
            fig_country = px.box(
                df_filtered,
                x="COUNTRY_PREF",
                y="SALARY_EXPECT",
                color="COUNTRY_PREF",
                title="Expected Salary by Country Preference",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_country, use_container_width=True)

    if {"JOB_FACTOR_MAPPED", "SALARY_EXPECT"}.issubset(df_filtered.columns):
        with c8:
            salary_by_factor = (
                df_filtered.groupby("JOB_FACTOR_MAPPED", as_index=False)["SALARY_EXPECT"].mean()
            )
            fig_jobfactor = px.bar(
                salary_by_factor,
                x="JOB_FACTOR_MAPPED",
                y="SALARY_EXPECT",
                color="JOB_FACTOR_MAPPED",
                title="Average Expected Salary by Job Factor",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_jobfactor, use_container_width=True)

    st.divider()

    # --- Row 3: Salary vs After Graduation & GPA vs Code Level ---
    c9, c10 = st.columns(2)

    if {"AFTER_GRAD", "SALARY_EXPECT"}.issubset(df_filtered.columns):
        with c9:
            fig_aftergrad = px.box(
                df_filtered,
                x="AFTER_GRAD",
                y="SALARY_EXPECT",
                color="AFTER_GRAD",
                title="Expected Salary by Post-Graduation Plan",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig_aftergrad, use_container_width=True)

    if {"GPA", "CODE_LEVEL"}.issubset(df_filtered.columns):
        with c10:
            fig_heatmap = px.density_heatmap(
                df_filtered,
                x="GPA",
                y="CODE_LEVEL",
                nbinsx=10,
                nbinsy=5,
                color_continuous_scale="Viridis",
                title="GPA vs Code Level Density"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # --- Download Filtered Dataset ---
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Dataset", csv, "filtered_dataset.csv", "text/csv")

# ==========================
# 🎯 PREDICTION PAGE
# ==========================
# ==========================
# 🤖 PREDICTION PAGE (Salary Prediction using trained model)
# ==========================
elif menu == "Prediction":
    st.subheader("🎯 Predict Expected Salary (VND/month)")
    st.markdown("#### 🔧 Enter your information below")

    # ============================================
    # 1️⃣ Load both models (rf1 + rf2)
    # ============================================
    try:
        with open("salary_model.pkl", "rb") as f:
            models = pickle.load(f)
        model_rf1 = models["model_rf1"]
        model_rf2 = models["model_rf2"]
    except Exception as e:
        st.error(f"❌ Could not load model file: {e}")
        st.stop()

    # ============================================
    # 2️⃣ Input fields
    # ============================================
    col1, col2 = st.columns(2)
    with col1:
        gpa = st.slider("GPA", 0.0, 10.0, 7.5, 0.1)
        code_level = st.slider("Coding Skill Level (1–5)", 1, 5, 3)
        sem = st.number_input("Number of Semesters (e.g., 6)", min_value=1, max_value=12, value=6, step=1)
        strength = st.selectbox("Strength", [
            "Kỹ năng mềm (teamwork, thuyết trình, quản lý)",
            "Kỹ năng chuyên môn (lập trình, giải quyết kỹ thuật)",
            "Cả 2"
        ])
    with col2:
        languages = st.selectbox("Languages you know", [
            "Tiếng Việt",
            "Tiếng Anh",
            "Tiếng Việt, Tiếng Anh",
            "Tiếng Anh, Tiếng Nhật",
            "Tiếng Việt, Tiếng Anh, Tiếng Nhật",
            "Tiếng Việt, Tiếng Anh, Tiếng Trung",
            "Tiếng Việt, Tiếng Anh, Tiếng Nhật, Tiếng Trung",
            "Tất cả đáp án trên"
        ])
        position = st.multiselect("Preferred job positions", [
            "Data / AI",
            "Design / UX",
            "Infrastructure / DevOps",
            "Management",
            "Other",
            "QA / Testing",
            "Software Development"
        ])
        expected_salary = st.number_input("💭 Your expected salary (VND/month):", min_value=0.0, step=500000.0)

    # ============================================
    # 3️⃣ Compute derived features
    # ============================================
    if st.button("🚀 Predict Salary"):
        try:
            # EstimatedExperienceYears
            EstimatedExperienceYears = sem * 0.1125

            # SeniorityScore
            w1, w2, w3, w4, w5 = 0.3, 0.5, 0.2, 0.15, 0.2
            if strength in [
                "Kỹ năng mềm (teamwork, thuyết trình, quản lý)",
                "Kỹ năng chuyên môn (lập trình, giải quyết kỹ thuật)"
            ]:
                SeniorityScore = 1 + w1 * gpa + w2 * code_level + w3 * EstimatedExperienceYears * w4
            else:  # "Cả 2"
                SeniorityScore = 1 + w1 * gpa + w2 * code_level + w3 * EstimatedExperienceYears * w5
            SeniorityScore = float(np.clip(SeniorityScore, 1.0, 3.0))

            # LANGUAGES encoding
            mapping_exact = {
                "Tiếng Việt": 1,
                "Tiếng Việt, Tiếng Anh": 1.3,
                "Tiếng Anh": 1.3,
                "Tiếng Anh, Tiếng Nhật": 1.3,
                "Tiếng Việt, Tiếng Anh, Tiếng Nhật": 1.5,
                "Tiếng Việt, Tiếng Anh, Tiếng Trung": 1.6,
                "Tiếng Việt, Tiếng Anh, Tiếng Nhật, Tiếng Trung": 2,
                "Tiếng Việt, Tiếng Anh, Nga , Tây Ban Nha": 2,
                "Tất cả đáp án trên": 2,
            }
            val = mapping_exact.get(languages, 1.3)
            if isinstance(languages, str) and ("," in languages) and (languages not in mapping_exact):
                val = 2.5
            if val == 2:
                val = 1.3
            LANGUAGES = float(val)

            # One-hot positions
            bool_cols = {
                "PositionEncode_Data / AI": 1 if "Data / AI" in position else 0,
                "PositionEncode_Design / UX": 1 if "Design / UX" in position else 0,
                "PositionEncode_Infrastructure / DevOps": 1 if "Infrastructure / DevOps" in position else 0,
                "PositionEncode_Management": 1 if "Management" in position else 0,
                "PositionEncode_Other": 1 if "Other" in position else 0,
                "PositionEncode_QA / Testing": 1 if "QA / Testing" in position else 0,
                "PositionEncode_Software Development": 1 if "Software Development" in position else 0,
            }

            # X input order
            X_input = np.array([[
                EstimatedExperienceYears,
                SeniorityScore,
                LANGUAGES,
                bool_cols["PositionEncode_Data / AI"],
                bool_cols["PositionEncode_Design / UX"],
                bool_cols["PositionEncode_Infrastructure / DevOps"],
                bool_cols["PositionEncode_Management"],
                bool_cols["PositionEncode_Other"],
                bool_cols["PositionEncode_QA / Testing"],
                bool_cols["PositionEncode_Software Development"],
            ]])

            # ============================================
            # 4️⃣ Predict: model1 → pseudo → model2
            # ============================================
            pseudo_salary = model_rf1.predict(X_input)[0]
            final_salary_annual = model_rf2.predict(X_input)[0]
            predicted_monthly = final_salary_annual / 12.0

            # ============================================
            # 5️⃣ Smooth & Convert like training pipeline
            # ============================================
            # Student salary
            if EstimatedExperienceYears < 0.3:
                base, bonus = 2_000_000, SeniorityScore * 300_000 + LANGUAGES * 200_000
                student_salary = np.clip(base + bonus, 1_000_000, 4_000_000)
            elif EstimatedExperienceYears < 0.6:
                base, bonus = 3_000_000, SeniorityScore * 400_000 + LANGUAGES * 250_000
                student_salary = np.clip(base + bonus, 2_000_000, 6_000_000)
            else:
                base, bonus = 4_000_000, SeniorityScore * 500_000 + LANGUAGES * 300_000
                student_salary = np.clip(base + bonus, 3_000_000, 8_000_000)

            # Model salary in VND/month (EU → VN)
            model_salary_vnd = (predicted_monthly * 27000) / 9

            # Sigmoid smoothing
            def smooth_salary(exp, student, model, threshold=1.0, k=5):
                w = 1 / (1 + np.exp(-k * (exp - threshold)))
                return (1 - w) * student + w * model

            vn_salary_final = smooth_salary(EstimatedExperienceYears, student_salary, model_salary_vnd)

            # Chênh lệch với kỳ vọng
            diff_show = abs(expected_salary - vn_salary_final)
            diff_cal = expected_salary - vn_salary_final

            # ============================================
            # 6️⃣ Display results
            # ============================================
            st.metric("💰 Estimated Salary (VNĐ/tháng)", f"{vn_salary_final:,.0f}")
            st.metric("📊 Difference vs Expected", f"{diff_show:,.0f}")

            if diff_cal > 0:
                st.warning("⚠️ Your expectation is higher than estimated range.")
            else:
                st.success("✅ Your expectation is within or below the estimated range.")

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")


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

        # === Local intent responses ===
        def handle_local_query(prompt: str):
            p = prompt.lower()

            # 1️⃣ GPA theo giới tính & kỳ học
            if "gpa" in p and "gender" in p or "kỳ học" in p:
                avg = df.groupby(['SEMESTER', 'GENDER'])['GPA'].mean().reset_index()
                result = avg.pivot(index='SEMESTER', columns='GENDER', values='GPA').round(2)
                return f"📊 GPA trung bình theo giới tính và kỳ học:\n\n{result.to_markdown()}"

            # 2️⃣ GPA cao chọn lĩnh vực nào
            elif "gpa cao" in p or "lĩnh vực" in p or "field" in p:
                top_field = df[df['GPA'] > 9]['FIELD_PREF_MAPPED'].value_counts().head(3)
                msg = "\n".join([f"- {i}: {v} sinh viên" for i, v in top_field.items()])
                return f"🧠 Trong nhóm GPA > 9, các lĩnh vực được chọn nhiều nhất là:\n{msg}"

            # 3️⃣ Job factor & Salary
            elif "job factor" in p or "yếu tố" in p or "mức lương" in p:
                env = df[df['JOB_FACTOR_MAPPED'] == 'Môi trường chuyên nghiệp']['SALARY_EXPECT'].mean()
                income = df[df['JOB_FACTOR_MAPPED'] == 'Thu nhập']['SALARY_EXPECT'].mean()
                return f"💼 Sinh viên coi trọng môi trường chuyên nghiệp có mức lương trung bình: {env:,.0f} VND.\n💸 Còn sinh viên coi trọng thu nhập: {income:,.0f} VND."

            # 4️⃣ Project Style vs Work Style
            elif "project" in p or "team" in p or "cá nhân" in p:
                table = pd.crosstab(df['PROJECT_STYLE'], df['WORK_STYLE'])
                return f"👥 So sánh xu hướng làm việc:\n\n{table.to_markdown()}"

            # 5️⃣ So sánh lương theo chuyên ngành
            elif "ai" in p and "web" in p or "ngành" in p and "lương" in p:
                pivot = pd.crosstab(df['MAJOR'], df['SALARY_EXPECT'], normalize='index') * 100
                pivot = pivot.loc[['Trí tuệ nhân tạo', 'Kỹ thuật phần mềm']].round(1)
                return f"💰 Tỉ lệ phân bố lương mong muốn (%):\n\n{pivot.to_markdown()}"

            # 6️⃣ Quốc gia làm việc & GPA
            elif "quốc gia" in p or "nước ngoài" in p or "việt nam" in p:
                df_valid = df[df["GPA"] > 0]
                vn = df_valid[df_valid["COUNTRY_PREF"].str.contains("Việt Nam", case=False, na=False)]["GPA"].mean()
                other = df_valid[~df_valid["COUNTRY_PREF"].str.contains("Việt Nam", case=False, na=False)]["GPA"].mean()
                return f"🌏 GPA trung bình:\n🇻🇳 Việt Nam: {vn:.2f}\n🌎 Nước ngoài: {other:.2f}"

            # 7️⃣ Mối liên hệ GPA & code level
            elif "code" in p and "gpa" in p or "trình độ" in p:
                df_valid = df[df["GPA"] > 0]
                high = df_valid[df_valid["GPA"] > 7]
                low = df_valid[df_valid["GPA"] <= 7]
                high_ratio = (high["CODE_LEVEL"] >= 4).mean() * 100
                low_ratio = (low["CODE_LEVEL"] >= 4).mean() * 100
                return f"💻 Trong nhóm GPA > 7, {high_ratio:.1f}% có code level ≥ 4.\nCòn GPA ≤ 7: {low_ratio:.1f}%."

            else:
                return None  # Không trùng → gọi Gemini

        # === Render chat history ===
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # === Chat input ===
        if prompt := st.chat_input("Type your question here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state["messages"].append({"role": "user", "content": prompt})

            local_reply = handle_local_query(prompt)
            if local_reply:
                reply = local_reply
            else:
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
    **Course:** AI-DS Project(DAP391m)  
    ---
    **Objective:**  
    - Analyze salary factors of IT students  
    - Predict potential salary using ML models  
    - Visualize insights through dashboard & chatbot  
    ---
    **Team members:**  
    - ‍👨💻 Mai Phạm Duy Khánh – Data Cleaning & Visualization & Web Support   
    - 👩💻 Đào Thị Linh Thư – Data Cleaning & Visualization & Report
    - ‍👩💻 Nguyễn Triệu Yến Nhi – Web App & Chatbot & Model Building Support 
    - ‍👨💻 Hồ Tấn Thành (Leader) – Model Building  
    """)

