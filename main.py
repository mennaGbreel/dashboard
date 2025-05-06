# mental_stress_dashboard.py – Streamlit app with 10 distinct charts
# Author: ChatGPT (OpenAI o3)
# -----------------------------------------------------------------------------
# This version guards against optional dependencies (Plotly, WordCloud) not being
# present in the execution environment, so the app fails gracefully instead of
# crashing with ModuleNotFoundError.
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Optional libraries – guarded imports
# -----------------------------------------------------------------------------
try:
    import plotly.express as px  # type: ignore
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud  # type: ignore
    HAS_WORDCLOUD = True
except ModuleNotFoundError:
    HAS_WORDCLOUD = False

# -----------------------------------------------------------------------------
# Page configuration & theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="US Population Dashboard",  # keep original title for continuity
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.themes.enable("dark")

# -----------------------------------------------------------------------------
# Custom CSS tweaks (unchanged from the user's snippet)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }
    [data-testid="stVerticalBlock"] {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 15px 0;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stMetricDeltaIcon-Up"] {
        position: relative;
        left: 38%;
        transform: translateX(-50%);
    }
    [data-testid="stMetricDeltaIcon-Down"] {
        position: relative;
        left: 38%;
        transform: translateX(-50%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading & preprocessing
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load the three CSVs once and cache them."""
    campaign = pd.read_csv("data/Mental_Health_Campaign_News_Dataset.csv")
    session = pd.read_csv("data/Counseling_Center_Statistics_Dataset.csv")
    stress = pd.read_csv("data/Student_Stress_Survey_Dataset.csv")

    # Campaign: extract year reliably from date column (assumed yyyy-mm-dd)
    campaign["Year"] = pd.to_datetime(campaign["Date"], errors="coerce").dt.year

    # Ensure session Year column is int (already present in CSV)
    session["Year"] = session["Year"].astype(int)

    # Stress survey: ensure categorical columns exist if present
    if "Stress_Level" in stress.columns:
        stress["Stress_Level"] = stress["Stress_Level"].astype(str)
    return campaign, session, stress

campaign_df_all, session_df_all, stress_df_all = load_data()

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Student Stress Dashboard")

    year_options = sorted(session_df_all["Year"].unique())
    year_options.insert(0, "All")
    selected_year = st.selectbox("Select a year", year_options)

    if selected_year != "All":
        yr = int(selected_year)
        session_df = session_df_all[session_df_all["Year"] == yr]
        campaign_df = campaign_df_all[campaign_df_all["Year"] == yr]
        stress_df = (
            stress_df_all[stress_df_all["Year"] == yr]
            if "Year" in stress_df_all.columns else stress_df_all
        )
    else:
        session_df, campaign_df, stress_df = (
            session_df_all,
            campaign_df_all,
            stress_df_all,
        )

    # Optional University filter if column exists
    if "University" in session_df.columns:
        univ_options = sorted(session_df["University"].unique())
        univ_options.insert(0, "All")
        selected_univ = st.selectbox("Select a university", univ_options)

        if selected_univ != "All":
            session_df = session_df[session_df["University"] == selected_univ]
            campaign_df = campaign_df[campaign_df["University"] == selected_univ]
            if "University" in stress_df.columns:
                stress_df = stress_df[stress_df["University"] == selected_univ]

# -----------------------------------------------------------------------------
# Helper to notify about missing optional deps
# -----------------------------------------------------------------------------

def _warn_missing(lib_name: str, pip_str: str = ""):
    st.warning(
        f"Optional dependency **{lib_name}** is not installed. "
        f"Install with `pip install {pip_str or lib_name.lower()}` to enable this chart.")

# -----------------------------------------------------------------------------
# Chart‑builder helper functions (10 total)
# -----------------------------------------------------------------------------

def draw_sessions_trend(df: pd.DataFrame):
    if df.empty:
        st.info("No data for selected filters.")
        return
    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.4)
        .encode(
            x="Year:O",
            y="sum(Sessions_Held):Q",
            color="University:N",
            tooltip=["University", "Year", "sum(Sessions_Held)"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def draw_sessions_bubble(df: pd.DataFrame):
    if df.empty:
        st.info("No data for selected filters.")
        return
    if HAS_PLOTLY:
        fig = px.scatter(
            df,
            x="Sessions_Held",
            y="Students_Served",
            size="Avg_Session_Duration",
            color="University" if "University" in df.columns else None,
            hover_data=["Year"],
            title="Workload Bubble Chart",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        _warn_missing("plotly-express", "plotly")


def draw_sessions_heatmap(df: pd.DataFrame):
    if df.empty:
        st.info("No data for selected filters.")
        return
    pivot = (
        df.pivot_table(index="University", columns="Year", values="Sessions_Held", aggfunc="sum")
        .reset_index()
        .melt("University", var_name="Year", value_name="Sessions_Held")
    )
    chart = (
        alt.Chart(pivot)
        .mark_rect()
        .encode(
            x="Year:O",
            y="University:N",
            color=alt.Color("Sessions_Held:Q", scale=alt.Scale(scheme="inferno")),
            tooltip=["University", "Year", "Sessions_Held"]
        ).properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def draw_stress_stacked_bar(df: pd.DataFrame):
    if df.empty or "Stress_Level" not in df.columns:
        st.info("Stress level data unavailable for selected filters.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="count():Q",
            y="University:N",
            color="Stress_Level:N",
            tooltip=["University", "Stress_Level", "count()"]
        ).properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def draw_sleep_violin(df: pd.DataFrame):
    if df.empty or "Avg_Sleep_Hours" not in df.columns:
        st.info("Sleep data unavailable for selected filters.")
        return
    if HAS_PLOTLY:
        fig = px.violin(
            df,
            y="Avg_Sleep_Hours",
            x="Stress_Level" if "Stress_Level" in df.columns else None,
            color="Gender" if "Gender" in df.columns else None,
            box=True,
            points="all",
            title="Sleep Hours vs Stress Level",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        _warn_missing("plotly-express", "plotly")


def draw_stress_treemap(df: pd.DataFrame):
    if df.empty or "Primary_Stress_Factor" not in df.columns:
        st.info("Stress factor data unavailable for selected filters.")
        return
    if HAS_PLOTLY:
        fig = px.treemap(df, path=["Primary_Stress_Factor"], title="Share of Primary Stress Factors")
        st.plotly_chart(fig, use_container_width=True)
    else:
        _warn_missing("plotly-express", "plotly")


def draw_stress_sunburst(df: pd.DataFrame):
    required = {"Stress_Level", "Seeks_Help", "Gender"}
    if df.empty or not required.issubset(df.columns):
        st.info("Sunburst requires Stress_Level, Seeks_Help, and Gender columns.")
        return
    if HAS_PLOTLY:
        fig = px.sunburst(df, path=["Stress_Level", "Seeks_Help", "Gender"], title="Help‑Seeking Path by Stress Level")
        st.plotly_chart(fig, use_container_width=True)
    else:
        _warn_missing("plotly-express", "plotly")


def draw_age_hist(df: pd.DataFrame):
    if df.empty or "Age" not in df.columns:
        st.info("Age data unavailable for selected filters.")
        return
    if HAS_PLOTLY:
        fig = px.histogram(
            df,
            x="Age",
            color="Stress_Level" if "Stress_Level" in df.columns else None,
            nbins=20,
            marginal="box",
            title="Age Distribution of Survey Participants",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Altair fallback histogram
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x="Age:Q", y="count():Q", tooltip=["count()"])
            .properties(height=300, title="Age Distribution (Altair fallback)")
        )
        st.altair_chart(chart, use_container_width=True)


def draw_campaign_timeline(df: pd.DataFrame):
    if df.empty:
        st.info("No campaign data for selected filters.")
        return
    agg = df.groupby(["Year", "University" if "University" in df.columns else df.columns[0]]).size().reset_index(name="Headlines")
    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x="Year:O",
            y="Headlines:Q",
            color="University:N" if "University" in agg.columns else alt.value("#BBBBBB"),
            tooltip=[alt.Tooltip("Headlines:Q"), "Year"]
        ).properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def draw_session_notes_wordcloud(df: pd.DataFrame):
    if df.empty or "Session_Notes" not in df.columns:
        st.info("Session notes unavailable for selected filters.")
        return
    if not HAS_WORDCLOUD:
        _warn_missing("wordcloud")
        return
    text = " ".join(df["Session_Notes"].dropna().astype(str).tolist())
    if not text.strip():
        st.info("Session notes are empty after filtering.")
        return
    wc = WordCloud(width=800, height=400, background_color="black", colormap="plasma").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Dashboard layout – sections & expanders
# -----------------------------------------------------------------------------
st.header("Counseling Center Statistics")
with st.expander("Utilization Charts", expanded=True):
    draw_sessions_trend(session_df)
    draw_sessions_bubble(session_df)
    draw_sessions_heatmap(session_df)

st.header("Student Stress Survey")
with st.expander("Stress Insights", expanded=True):
    draw_stress_stacked_bar(stress_df)
    draw_sleep_violin(stress_df)
    draw_stress_treemap(stress_df)
    draw_stress_sunburst(stress_df)
    draw_age_hist(stress_df)

st.header("Mental‑Health Campaign News")
with st.expander("Visibility over Time", expanded=True):
    draw_campaign_timeline(campaign_df)

st.header("Session Notes NLP")
with st.expander("Common Themes in Counseling Notes", expanded=False):
    draw_session_notes_wordcloud(session_df)

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
