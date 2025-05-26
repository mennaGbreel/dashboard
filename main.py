import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

pio.templates.default = 'plotly_white'

st.set_page_config(
    page_title="Student Stress Dashboard",
    page_icon="😩",
    layout="wide",
)


def load_data():
    campaign = pd.read_csv("data/Mental_Health_Campaign_News_Dataset.csv")
    session = pd.read_csv("data/Counseling_Center_Statistics_Dataset.csv")
    stress = pd.read_csv("data/Student_Stress_Survey_Dataset.csv")

    campaign["Year"] = pd.to_datetime(campaign["Date"], errors="coerce").dt.year
    session["Year"] = session["Year"].astype(int)

    if "Stress_Level" in stress.columns:
        stress["Stress_Level"] = stress["Stress_Level"].astype(str)
    if "Gender" in stress.columns:
        stress = stress[stress["Gender"].isin(["Male", "Female"])]

    return campaign, session, stress


campaign_df_all, session_df_all, stress_df_all = load_data()

with st.sidebar:
    st.title("Student Stress Dashboard")
    session_df = session_df_all.copy()
    campaign_df = campaign_df_all.copy()
    stress_df = stress_df_all.copy()

    universities = sorted(session_df["University"].unique())
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select All", use_container_width=True):
            for uni in universities:
                st.session_state[uni] = True
    with c2:
        if st.button("Deselect All", use_container_width=True):
            for uni in universities:
                st.session_state[uni] = False

    selected = [
        uni for uni in universities
        if st.checkbox(uni, key=uni, value=st.session_state.get(uni, True))
    ]
    session_df = session_df[session_df["University"].isin(selected)]
    stress_df = stress_df[stress_df["University"].isin(selected)]
    campaign_df = campaign_df[campaign_df["University"].isin(selected)]


def sessions_held_area_chart(df):
    summary = df.groupby(['Year', 'University'], as_index=False)['Sessions_Held'].sum()
    fig = px.area(summary, x='Year', y='Sessions_Held', color='University', title='Sessions Held by Year')
    fig.for_each_trace(lambda trace: trace.update(fillcolor=trace.line.color))
    st.plotly_chart(fig)


def stress_factors_stacked_bar_chart(df):
    counts = df.groupby(['University', 'Primary_Stress_Factor']).size().reset_index(name='count')
    fig = px.bar(
        counts, x='count', y='University', color='Primary_Stress_Factor', orientation='h',
        barmode='stack', title='Primary Stress Factors by University'
    )
    st.plotly_chart(fig)


def gender_dist_pie_chart(df):
    counts = df['Gender'].value_counts().reset_index()
    counts.columns = ['Gender', 'count']
    fig = px.pie(counts, names='Gender', values='count', title='Gender Distribution')
    fig.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
    st.plotly_chart(fig)


def gender_stress_box_plot(df):
    mapping = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    df2 = df[df['Stress_Level'].isin(mapping)].copy()
    df2['Stress_Num'] = df2['Stress_Level'].map(mapping)
    fig = px.box(df2, x='Gender', y='Stress_Num', color='Gender', title='Stress Level Distribution by Gender')
    st.plotly_chart(fig)


def university_students_bar_chart(df):
    counts = df['University'].value_counts().reset_index()
    counts.columns = ['University', 'count']
    fig = px.bar(counts, x='count', y='University', text='count', orientation='h', title='Students per University')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)


def preprocess_wordcloud(text):
    stopwords = set(ENGLISH_STOP_WORDS)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stopwords and len(w) > 1]
    return tokens



def draw_headline_wordcloud(df):
    texts = df['Headline'].dropna().astype(str)
    stopwords = set(ENGLISH_STOP_WORDS)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [w for w in text.split() if w not in stopwords and len(w) > 1]
        return tokens

    all_tokens = []
    for t in texts:
        all_tokens.extend(preprocess(t))

    wc = WordCloud(width=2000, height=1200, background_color='#F7F9FA', collocations=False).generate(' '.join(all_tokens))
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def draw_stress_factor_breakdown(df: pd.DataFrame):
    required = ['University', 'Primary_Stress_Factor', 'Stress_Level']
    counts = df.groupby(required).size().reset_index(name='count')
    level_colors = {'Low': '#FFCCCC', 'Moderate': '#FF6666', 'High': '#CC0000', 'Severe': '#990000'}
    fig = px.treemap(
        counts, path=['University', 'Primary_Stress_Factor', 'Stress_Level'], values='count',
        color='Stress_Level', color_discrete_map=level_colors, title='Stress Factor Breakdown by University'
    )
    grey = '#ECEFF1'
    new_colors = []
    for item in fig.data[0]['ids']:
        parts = item.split('/')
        if len(parts) == 3:
            lvl = parts[2]
            new_colors.append(level_colors.get(lvl, grey))
        else:
            new_colors.append(grey)
    fig.data[0].marker.colors = new_colors
    st.plotly_chart(fig)


def draw_stress_vs_served(df_stress: pd.DataFrame, df_session: pd.DataFrame):
    level_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    df = df_stress[df_stress['Stress_Level'].isin(level_map)].copy()
    df['Stress_Num'] = df['Stress_Level'].map(level_map)
    stress_avg = df.groupby('University')['Stress_Num'].mean().reset_index(name='Avg_Stress_Level')

    served_avg = (
        df_session.groupby(['University', 'Year'])['Students_Served']
        .sum().reset_index()
        .groupby('University')['Students_Served'].mean()
        .reset_index(name='Avg_Students_Served_Per_Year')
    )

    merged = pd.merge(stress_avg, served_avg, on='University')
    fig = px.scatter(
        merged,
        x='Avg_Students_Served_Per_Year',
        y='Avg_Stress_Level',
        text='University',
        title='Avg Stress Level vs Avg Students Served'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)


st.header("Student Stress Dashboard")
st.subheader("Mennatallah Gbreel")

if stress_df.empty or session_df.empty or campaign_df.empty:
    st.info("Please select one or more universities.")
else:
    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        gender_dist_pie_chart(stress_df)
    with sub2:
        gender_stress_box_plot(stress_df)
    with sub3:
        university_students_bar_chart(stress_df)

    col1, col2 = st.columns(2)
    with col1:
        stress_factors_stacked_bar_chart(stress_df)

    with col2:
        sessions_held_area_chart(session_df)

    draw_stress_factor_breakdown(stress_df)

    st.subheader("Headline Word Cloud")
    draw_headline_wordcloud(campaign_df)
