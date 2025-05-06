#############################
#  main.py  –  Student‑Mental‑Health Dashboard
#############################
from pathlib import Path
import streamlit as st               # Streamlit import first!
import pandas as pd
import altair as alt
import plotly.express as px
import nltk, re
from collections import Counter

# ───────────────────────────
# Streamlit page settings ─ must be the first Streamlit call
# ───────────────────────────
st.set_page_config(
    page_title="US Student‑Mental‑Health Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.themes.enable("dark")

# ───────────────────────────
# Locate the data folder (repo‑root/data)
# ───────────────────────────
REPO_ROOT = Path(__file__).resolve().parent            # ← one level up is enough
DATA_DIR   = REPO_ROOT / "data"

# Optional one‑time check: list CSVs Streamlit can see
# st.write("CSV files found:", list(DATA_DIR.glob("*.csv")))

# ───────────────────────────
# Load data
# ───────────────────────────
def read_csv_must_exist(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Dataset missing → {label}: {path}")
        st.stop()
    return pd.read_csv(path)

campaign_df = read_csv_must_exist(DATA_DIR / "Mental_Health_Campaign_News_Dataset.csv",
                                  "Campaign news")
session_df  = read_csv_must_exist(DATA_DIR / "Counseling_Center_Statistics_Dataset.csv",
                                  "Counselling centre stats")
stress_df   = read_csv_must_exist(DATA_DIR / "Student_Stress_Survey_Dataset.csv",
                                  "Student stress survey")

# Ensure the date / year columns exist
campaign_df["Year"] = pd.to_datetime(campaign_df["Date"]).dt.year.astype(str)
session_df["Year"]  = session_df["Year"].astype(str)

# Harmonise university naming if needed
for df in (campaign_df, session_df, stress_df):
    if "University" in df.columns:
        df["University"] = df["University"].str.strip()

# ───────────────────────────
# Sidebar filters
# ───────────────────────────
with st.sidebar:
    st.title("📊 Filters")
    years = sorted({*campaign_df["Year"], *session_df["Year"]})
    universities = sorted(stress_df["University"].unique())

    year_sel = st.multiselect("Year", years, default=years)
    uni_sel  = st.multiselect("University", universities, default=universities)

# Apply filters
session_f = session_df[session_df["Year"].isin(year_sel) &
                       session_df["University"].isin(uni_sel)]
stress_f  = stress_df[stress_df["University"].isin(uni_sel)]
camp_f    = campaign_df[campaign_df["Year"].isin(year_sel) &
                        campaign_df["University"].isin(uni_sel)]

# ───────────────────────────
# Helper cache decorators
# ───────────────────────────
@st.cache_data
def bigram_freq(series, n=20):
    tokenizer = nltk.RegexpTokenizer(r"[A-Za-z']+")
    bigrams = Counter()
    for txt in series.dropna().astype(str):
        tokens = tokenizer.tokenize(txt.lower())
        bigrams.update(zip(tokens, tokens[1:]))
    return pd.DataFrame(bigrams.most_common(n), columns=["bigram", "count"])

# ───────────────────────────
# 1 & 2 — Counselling load over time
# ───────────────────────────
with st.expander("1‑2 | Counselling centre load over time"):
    c1, c2 = st.columns(2)

    # 1️⃣ line chart: students vs sessions
    line = (
        alt.Chart(session_f.melt(
            id_vars=["Year", "University"],
            value_vars=["Sessions Held", "Students Served"],
            var_name="Metric"))
        .mark_line(point=True)
        .encode(
            x="Year:O",
            y=alt.Y("value:Q", title="Count"),
            color="Metric",
            tooltip=["University", "Metric", "value"]
        )
        .facet(row="University")
        .properties(height=120)
    )
    c1.altair_chart(line, use_container_width=True)

    # 2️⃣ stacked‑area share of sessions
    share = (
        alt.Chart(session_f)
        .mark_area()
        .encode(
            x="Year:O",
            y=alt.Y("sum(Sessions Held):Q", stack="normalize", title="Share"),
            color="University",
            tooltip=["University", "sum(Sessions Held)"]
        )
    )
    c2.altair_chart(share, use_container_width=True)

# ───────────────────────────
# 3 — Avg session duration heat‑map
# ───────────────────────────
with st.expander("3 | Session duration heat‑map"):
    heat = (
        alt.Chart(session_f)
        .mark_rect()
        .encode(
            x="Year:O",
            y="University:N",
            color=alt.Color("mean(Avg Session Duration):Q",
                            scale=alt.Scale(scheme="magma")),
            tooltip=["University", "Year", "mean(Avg Session Duration)"]
        )
    )
    st.altair_chart(heat, use_container_width=True)

# ───────────────────────────
# 4 — Stress‑level distribution per university
# ───────────────────────────
with st.expander("4 | Stress‑level distribution"):
    order = ["Low", "Moderate", "High"]
    bar = (
        alt.Chart(stress_f)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Students"),
            y="University:N",
            color=alt.Color("Stress Level:N",
                            scale=alt.Scale(domain=order,
                                            range=["green", "orange", "red"])),
            column=alt.Column("Stress Level:N", sort=order)
        )
        .properties(height=200)
    )
    st.altair_chart(bar, use_container_width=True)

# ───────────────────────────
# 5 — Sleep vs stress violin
# ───────────────────────────
with st.expander("5 | Sleep hours vs stress level"):
    vio = px.violin(
        stress_f, y="Avg Sleep Hours", x="Stress Level",
        box=True, points="all", hover_data=["University"]
    )
    st.plotly_chart(vio, use_container_width=True)

# ───────────────────────────
# 6 — Sunburst of stress factors
# ───────────────────────────
with st.expander("6 | Primary stress factors"):
    sun = px.sunburst(
        stress_f,
        path=["Stress Level", "Primary Stress Factor"],
        values=None,
        color="Stress Level",
        color_discrete_map={"Low": "green", "Moderate": "orange", "High": "red"}
    )
    st.plotly_chart(sun, use_container_width=True)

# ───────────────────────────
# 7 — Text bigram frequency
# ───────────────────────────
with st.expander("7 | Common bigrams in counselling notes / campaign headlines"):
    source_col = st.radio("Choose text source",
                          ["Counselling Notes", "Campaign Headline"])
    if source_col == "Counselling Notes" and "Notes" in session_f.columns:
        text_series = session_f["Notes"]
    else:
        text_series = camp_f["Headline"]
    top_bi = bigram_freq(text_series)

    bigram_chart = (
        alt.Chart(top_bi)
        .mark_bar()
        .encode(
            x="count:Q",
            y=alt.Y("bigram:N", sort="-x"),
            tooltip=["count"]
        )
    )
    st.altair_chart(bigram_chart, use_container_width=True)

# ───────────────────────────
# 8 — Capacity vs demand bubble
# ───────────────────────────
with st.expander("8 | Counselling capacity vs demand"):
    level_map = {"Low": 1, "Moderate": 2, "High": 3}
    bubble_df = (
        session_f.groupby("University")
        .agg(Sessions=("Sessions Held", "sum"),
             Students=("Students Served", "sum"))
        .reset_index()
        .merge(
            stress_f.groupby("University")["Stress Level"]
            .apply(lambda s: s.map(level_map).median())
            .reset_index(),
            on="University", how="left"
        )
        .rename(columns={"Stress Level": "MedianStress"})
    )
    fig8 = px.scatter(
        bubble_df, x="Sessions", y="Students",
        size="MedianStress", hover_name="University",
        size_max=40
    )
    st.plotly_chart(fig8, use_container_width=True)

# ───────────────────────────
# 9 — Animated bar‑race of campaign articles
# ───────────────────────────
with st.expander("9 | Mental‑health news momentum"):
    race = px.bar(
        camp_f.groupby(["Year", "University"]).size()
        .reset_index(name="Articles"),
        x="Articles", y="University", color="University",
        orientation="h", animation_frame="Year",
        range_x=[0, camp_f.groupby(["Year", "University"])
                 .size().max() * 1.2]
    )
    st.plotly_chart(race, use_container_width=True)

# ───────────────────────────
# 10 — Choropleth sessions per‑capita
# ───────────────────────────
with st.expander("10 | Sessions per‑capita by state"):
    percap = (
        session_f.groupby(["Year", "State"])
        .agg(TotalSessions=("Sessions Held", "sum"))
        .reset_index()
        .merge(
            pop_df.rename(columns={"state": "State",
                                   "year": "Year",
                                   "population": "Population"}),
            on=["State", "Year"]
        )
    )
    percap["SessionsPer100k"] = (
        percap["TotalSessions"] / percap["Population"] * 100_000
    )

    year_choice = st.select_slider(
        "Year for map",
        sorted(percap["Year"].unique()),
        value=max(year_sel)
    )
    map_df = percap[percap["Year"] == str(year_choice)]

    fig10 = px.choropleth(
        map_df,
        locations="State", locationmode="USA-states",
        color="SessionsPer100k",
        color_continuous_scale="blues",
        scope="usa",
        labels={"SessionsPer100k": "Sessions / 100k"}
    )
    st.plotly_chart(fig10, use_container_width=True)

st.caption("© 2025 Student‑Mental‑Health Dashboard | Built with Streamlit, Altair, and Plotly")
