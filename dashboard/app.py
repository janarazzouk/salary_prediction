import os
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
import time 

@st.cache_data
def load_salary_data() -> pd.DataFrame:
    return pd.read_csv("../data/raw/ds_salaries.csv")

salary_df = load_salary_data()


# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Gemini client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gemini client init failed: {e}")
else:
    print("GEMINI_API_KEY not found in environment.")


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d4f3c 0%, #1a5f4a 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .form-card {
        background: #f8f9fa;
        border-left: 5px solid #0F8F67;
    }
    .result-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
        border-left: 5px solid #0F8F67;
    }
    .insight-card {
        background: #fff3e0;
        border-left: 5px solid #0F8F67;
    }
    .chart-card {
        background: #e3f2fd;
        border-left: 5px solid #0F8F67;
    }
    .history-card {
        background: #f3e5f5;
        border-left: 5px solid #0F8F67;
    }
    .info-card {
        background: linear-gradient(135deg, #F0F8FF 0%, #FFFFFF 100%);
        border-left: 5px solid #0F8F67;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0d4f3c;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-title {
        display: inline-block;
        background: linear-gradient(135deg, #0F8F67 0%, #0B5D46 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 143, 103, 0.3);
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .summary-item:last-child {
        border-bottom: none;
    }
    .summary-key {
        font-weight: 500;
        color: #555;
    }
    .summary-value {
        font-weight: 600;
        color: #0d4f3c;
    }
    .btn-predict {
        background: #0d4f3c !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background 0.3s !important;
    }
    .btn-predict:hover {
        background: #1a5f4a !important;
    }
    .stButton>button {
        background: #0d4f3c !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        background: #1a5f4a !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <div class="main-title">Salary Prediction Dashboard</div>
    <div class="subtitle">Predict data science salaries with AI-powered insights</div>
    <div class="badge">AI-Powered Salary Intelligence</div>
</div>
""", unsafe_allow_html=True)


# =========================
# SUPABASE CONNECTION
# =========================
supabase: Client | None = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
else:
    st.info("Supabase credentials not found. Results will not be stored.")


# =========================
# HELPER FUNCTIONS
# =========================
def call_prediction_api(payload: dict) -> dict:
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()






def generate_llm_insight(payload: dict, predicted_salary: float) -> str:
    if gemini_client is None:
        return build_fallback_insight(payload, predicted_salary)

    prompt = f"""
You are writing insight text for a modern salary prediction dashboard.

Your job is to explain a predicted salary in a way that feels polished, natural, and useful to a non-technical user.

Prediction:
- Annual salary (USD): ${predicted_salary:,.2f}

Input profile:
- Job title: {payload['job_title']}
- Experience level: {payload['experience_level']}
- Employment type: {payload['employment_type']}
- Company size: {payload['company_size']}
- Remote ratio: {payload['remote_ratio']}
- Company location: {payload['company_location']}

Write a short dashboard insight with this exact structure:

1. First paragraph:
- 2 to 3 sentences only
- Clearly explain what the predicted salary means
- Mention the role and experience level naturally
- Make the wording sound confident, professional, and human

2. Second paragraph:
- 1 to 2 sentences only
- Explain which factors are likely influencing the estimate most
- Mention experience, location, company size, and remote setup only if relevant
- Do not just list the inputs again

3. Then add:
Key takeaways
- bullet 1
- bullet 2

Writing rules:
- Use simple professional English
- Sound like a business dashboard, not like an academic report
- Be concise but not robotic
- Avoid phrases like:
  "this estimate suggests"
  "market value consistent with"
  "can influence compensation"
  "under a FT employment arrangement"
- Avoid repeating the exact same wording from the inputs
- Do not be overly cautious or vague
- Do not invent facts not supported by the input
- Keep the total output under 120 words
- Return only the final text, ready to display in the UI
"""

    max_retries = 2
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if hasattr(response, "text") and response.text and response.text.strip():
                return response.text.strip()

        except Exception as e:
            error_text = str(e).upper()
            retryable = "503" in error_text or "UNAVAILABLE" in error_text

            if retryable and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue

            break

    return build_fallback_insight(payload, predicted_salary)


def build_fallback_insight(payload: dict, predicted_salary: float) -> str:
    exp_map = {
        "EN": "entry-level",
        "MI": "mid-level",
        "SE": "senior-level",
        "EX": "executive-level",
    }

    remote_map = {
        0: "on-site",
        50: "hybrid",
        100: "fully remote",
    }

    exp = exp_map.get(payload["experience_level"], payload["experience_level"])
    remote = remote_map.get(payload["remote_ratio"], str(payload["remote_ratio"]))

    return f"""
The predicted annual salary is **${predicted_salary:,.2f}** for a **{payload['job_title']}** role.

This estimate suggests a market value consistent with a **{exp}** position under a **{payload['employment_type']}** employment arrangement. Company size (**{payload['company_size']}**) and location (**{payload['company_location']}**) can influence compensation, while the work setup is **{remote}**.

**Key takeaways**
- More senior experience levels usually support higher salary expectations.
- Location, remote policy, and role specialization can all shift compensation up or down.
""".strip()

def save_to_supabase(payload: dict, predicted_salary: float, insight: str):
    if supabase is None:
        print("Supabase client is None - not saving")
        return

    record = {
        "experience_level": payload["experience_level"],
        "employment_type": payload["employment_type"],
        "company_size": payload["company_size"],
        "remote_ratio": payload["remote_ratio"],
        "company_location": payload["company_location"],
        "job_title": payload["job_title"],
        "predicted_salary_usd": predicted_salary,
        "llm_insight": insight,
        "created_at": datetime.utcnow().isoformat()
    }

    try:
        result = supabase.table("salary_predictions").insert(record).execute()
        print(f"Supabase insert result: {result}")
    except Exception as e:
        print(f"Error saving to Supabase: {e}")


def load_history() -> pd.DataFrame:
    if supabase is None:
        return pd.DataFrame()

    try:
        response = (
            supabase
            .table("salary_predictions")
            .select("*")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        data = response.data if response.data else []
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading history: {e}")
        return pd.DataFrame()
    


def plot_top_job_titles(df: pd.DataFrame, top_n: int = 10):
    chart_df = (
        df.groupby("job_title", as_index=False)["salary_in_usd"]
        .mean()
        .sort_values("salary_in_usd", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(chart_df["job_title"], chart_df["salary_in_usd"])
    ax.invert_yaxis()

    ax.set_title(f"Top {top_n} Job Titles by Average Salary")
    ax.set_xlabel("Average Salary in USD")
    ax.set_ylabel("Job Title")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return fig

def plot_job_title_salary_by_country(df: pd.DataFrame, selected_job_title: str, top_n: int = 10):
    chart_df = df[df["job_title"] == selected_job_title].copy()

    if chart_df.empty:
        return None

    chart_df = (
        chart_df.groupby("company_location", as_index=False)["salary_in_usd"]
        .mean()
        .sort_values("salary_in_usd", ascending=False)
        .head(top_n)
    )

    country_labels = {
        "US": "United States",
        "GB": "United Kingdom",
        "CA": "Canada",
        "DE": "Germany",
        "IN": "India",
        "FR": "France",
        "ES": "Spain",
        "GR": "Greece",
        "OTHERS": "Others"
    }

    chart_df["country_name"] = chart_df["company_location"].map(country_labels).fillna(chart_df["company_location"])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(chart_df["country_name"], chart_df["salary_in_usd"])
    ax.invert_yaxis()

    ax.set_title(f"Average Salary by Country for {selected_job_title}")
    ax.set_xlabel("Average Salary in USD")
    ax.set_ylabel("Country")
    ax.grid(True, axis="x", alpha=0.3)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.tick_params(axis='x', labelsize=9)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (width * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f'${width:,.0f}',
            va='center',
            fontsize=10
        )

    plt.tight_layout()
    return fig

# =========================
# INPUT FORM
# =========================
st.markdown('<div class="section-title">Salary Prediction Inputs</div><div class="card form-card">', unsafe_allow_html=True)

exp_options = ["Entry Level", "Mid Level", "Senior Level", "Executive Level"]
exp_codes = ["EN", "MI", "SE", "EX"]

emp_options = ["Full Time", "Part Time", "Contractor", "Freelancer"]
emp_codes = ["FT", "PT", "CT", "FL"]

comp_options = ["Small", "Medium", "Large"]
comp_codes = ["S", "M", "L"]

remote_options = ["On Site", "Hybrid", "Fully Remote"]
remote_codes = [0, 50, 100]

loc_options = ["United States", "United Kingdom", "Canada", "Germany", "India", "France", "Spain", "Greece", "Others"]
loc_codes = ["US", "GB", "CA", "DE", "IN", "FR", "ES", "GR", "OTHERS"]

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        experience_level_display = st.selectbox(
            "Experience Level",
            options=exp_options
        )
        experience_level = exp_codes[exp_options.index(experience_level_display)]

        employment_type_display = st.selectbox(
            "Employment Type",
            options=emp_options
        )
        employment_type = emp_codes[emp_options.index(employment_type_display)]

        company_size_display = st.selectbox(
            "Company Size",
            options=comp_options
        )
        company_size = comp_codes[comp_options.index(company_size_display)]

        remote_ratio_display = st.selectbox(
            "Work Arrangement",
            options=remote_options
        )
        remote_ratio = remote_codes[remote_options.index(remote_ratio_display)]

    with col2:
        company_location_display = st.selectbox(
            "Company Location",
            options=loc_options
        )
        company_location = loc_codes[loc_options.index(company_location_display)]
        job_title = st.selectbox(
            "Job Title",
            options=['Data Scientist', 'Machine Learning Scientist',
       'Big Data Engineer', 'Product Data Analyst',
       'Machine Learning Engineer', 'Data Analyst', 'Lead Data Scientist',
       'Business Data Analyst', 'Lead Data Engineer', 'Lead Data Analyst',
       'Data Engineer', 'Data Science Consultant', 'BI Data Analyst',
       'Director of Data Science', 'Research Scientist',
       'Machine Learning Manager', 'Data Engineering Manager',
       'Machine Learning Infrastructure Engineer', 'ML Engineer',
       'AI Scientist', 'Computer Vision Engineer',
       'Principal Data Scientist', 'Data Science Manager', 'Head of Data',
       '3D Computer Vision Researcher', 'Data Analytics Engineer',
       'Applied Data Scientist', 'Marketing Data Analyst',
       'Cloud Data Engineer', 'Financial Data Analyst',
       'Computer Vision Software Engineer',
       'Director of Data Engineering', 'Data Science Engineer',
       'Principal Data Engineer', 'Machine Learning Developer',
       'Applied Machine Learning Scientist', 'Data Analytics Manager',
       'Head of Data Science', 'Data Specialist', 'Data Architect',
       'Finance Data Analyst', 'Principal Data Analyst',
       'Big Data Architect', 'Staff Data Scientist', 'Analytics Engineer',
       'ETL Developer', 'Head of Machine Learning', 'NLP Engineer',
       'Lead Machine Learning Engineer', 'Data Analytics Lead']
        )

    submitted = st.form_submit_button("Predict Salary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =========================
# PREDICTION FLOW
# =========================
if submitted:
    payload = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
        "remote_ratio": remote_ratio,
        "company_location": company_location.strip().upper(),
        "job_title": job_title.strip()
    }

    try:
        api_result = call_prediction_api(payload)
        predicted_salary = api_result["predicted_salary_usd"]

       

        # Results Card
        st.markdown('<div class="section-title">Salary Prediction Results</div><div class="card result-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-value">${predicted_salary:,.0f}</div><div class="metric-label">Predicted Annual Salary (USD)</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-value">{job_title}</div><div class="metric-label">Job Title</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-value">{company_location_display}</div><div class="metric-label">Location</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input Summary Card
        st.markdown('<div class="section-title"> Input Summary</div><div class="card">', unsafe_allow_html=True)
        summary_items = [
            ("Experience Level", experience_level_display),
            ("Employment Type", employment_type_display),
            ("Company Size", company_size_display),
            ("Work Arrangement", remote_ratio_display),
            ("Company Location", company_location_display),
            ("Job Title", job_title)
        ]
        for key, value in summary_items:
            st.markdown(f'<div class="summary-item"><span class="summary-key">{key}:</span><span class="summary-value">{value}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Insight Card
        st.markdown('<div class="section-title"> AI Executive Analysis</div><div class="card insight-card">', unsafe_allow_html=True)
        with st.spinner("Generating AI insight..."):
            insight = generate_llm_insight(payload, predicted_salary)
        st.write(insight)
        st.markdown('</div>', unsafe_allow_html=True)

        # Visualization Card
        st.markdown('<div class="section-title"> Salary Analytics</div><div class="card chart-card">', unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Average Salary by Country for Selected Role")
            fig_country = plot_job_title_salary_by_country(salary_df, job_title)

            if fig_country is not None:
                st.pyplot(fig_country)
            else:
                st.info(f"No country-level salary data found for {job_title}.")

        with col_chart2:
            st.subheader(" Top Job Titles by Average Salary")
            fig_jobs = plot_top_job_titles(salary_df)
            st.pyplot(fig_jobs)

        st.markdown('</div>', unsafe_allow_html=True)

        save_to_supabase(payload, predicted_salary, insight)

    except requests.exceptions.RequestException as e:
        st.error(f"❌ FastAPI request failed: {e}")
    except KeyError as e:
        st.error(f"❌ Unexpected API response format: missing {e}")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")


# =========================
# HISTORY SECTION
# =========================
st.markdown('<div class="section-title">Recent Predictions</div><div class="card history-card">', unsafe_allow_html=True)

history_df = load_history()

if not history_df.empty:
    # Format the dataframe for better display
    display_df = history_df.copy()
    if 'predicted_salary_usd' in display_df.columns:
        display_df['predicted_salary_usd'] = display_df['predicted_salary_usd'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else x)
    if 'created_at' in display_df.columns:
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

    display_columns = [
        col for col in [
            "created_at",
            "job_title",
            "experience_level",
            "employment_type",
            "company_size",
            "remote_ratio",
            "company_location",
            "predicted_salary_usd"
        ] if col in display_df.columns
    ]
    st.dataframe(display_df[display_columns], use_container_width=True, hide_index=True)
else:
    st.write("No saved predictions found yet.")

st.markdown('</div>', unsafe_allow_html=True)