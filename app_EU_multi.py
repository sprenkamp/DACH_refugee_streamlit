import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
#check different leafmap backends
import leafmap.foliumap as leafmap
# import leafmap.kepler as leafmap #dark themed
#Config must be first line in script
st.set_page_config(layout="wide")
# TODO remove countries not used
# TODO styler package

gdf = gpd.read_file("data/germany_switzerland.geojson")

def dummy_function_space():
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

@st.cache()
def load_telegram_data():
    df = pd.read_csv("data/df_telegram.csv")
    #print(df.head(1))
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    return df
@st.cache
def load_news_data():
    df = pd.read_csv("data/df_news.csv")
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    return df
@st.cache()
def load_twitter_data():
    df = pd.read_csv("data/df_twitter.csv")
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    return df

def create_df_value_counts(df):
    messages_per_week_dict = dict(df.value_counts("date"))
    df_value_counts = df.value_counts(["cluster", "date"]).reset_index()
    df_value_counts.columns = ["cluster", "date", "occurence_count"]
    return df_value_counts

def modify_df_for_table(df_mod, country_select, state_select, cluster_select, date_slider, metric_select=None):
    if country_select!="all countries analysed":
        df_mod = df_mod[df_mod.country==country_select]
    if state_select not in ["all states analysed", "all german states", "all swiss cantons"]:
        df_mod = df_mod[df_mod.state==state_select]
    if not "all found clusters" in cluster_select:
        df_mod = df_mod[df_mod.cluster.isin(cluster_select)]
    df_mod = df_mod[df_mod.date.between(date_slider[0], date_slider[1])]
    return df_mod

df_telegram = load_telegram_data()
df_twitter = load_twitter_data()
df_news = load_news_data()
st.title('Identification of the most relevant topics in the context of the Ukrainian Refugee Crisis in the media and social media')
text_col1, text_col2, text_col3  = st.columns(3)
with text_col1:
    country_select = st.selectbox(
        "Select a country of interest",
        ["all countries analysed", "Germany", "Switzerland"],
        )
with text_col2:
    states = ["all states analysed"] + gdf.state.unique().tolist()
    if country_select=="Germany":
        states = ["all german states"] + gdf[gdf["country"]=="Germany"].state.unique().tolist()
    if country_select=="Switzerland":
        states = ["all swiss cantons"] + gdf[gdf["country"]=="Switzerland"].state.unique().tolist()
    state_select = st.selectbox(
        'Choose a state of interest',
        states,
        )

with text_col3:
    date_slider = st.slider('Choose date range of interest',
        min_value=df_telegram.date.min(), 
        value=(df_telegram.date.min(), df_telegram.date.max()), 
        max_value=df_telegram.date.max()
        )

# Using "with" notation
with st.sidebar:
    cluster_select_telegram = st.multiselect(
        'Choose the topics of interest within the telegram data',
        ["all found clusters"] + df_telegram.cluster.unique().tolist(),
        ["all found clusters"]
        )
    cluster_select_twitter = st.multiselect(
        'Choose the topics of interest within the twitter data',
        ["all found clusters"] + df_twitter.cluster.unique().tolist(),
        ["all found clusters"]
        )
    cluster_select_news = st.multiselect(
        'Choose the topic of interest within the news data',
        ["all found clusters"] + df_news.cluster.unique().tolist(),
        ["all found clusters"]
        )
    dummy_function_space()
    source_select_df = st.selectbox(
        "Read text data from source",
        ["Telegram", "Twitter", "News"],
        )

df_telegram_mod = modify_df_for_table(df_mod=df_telegram, country_select=country_select, state_select=state_select, cluster_select=cluster_select_telegram, date_slider=date_slider)
df_value_counts_telegram = create_df_value_counts(df=df_telegram_mod)
df_twitter_mod = modify_df_for_table(df_mod=df_twitter, country_select=country_select, state_select=state_select, cluster_select=cluster_select_twitter, date_slider=date_slider)
df_value_counts_twitter = create_df_value_counts(df=df_twitter_mod)    
df_news_mod = modify_df_for_table(df_mod=df_news, country_select=country_select, state_select=state_select, cluster_select=cluster_select_news, date_slider=date_slider)
df_value_counts_news = create_df_value_counts(df=df_news_mod) 


visual_col1, visual_col2= st.columns(2)
with visual_col1:
    if country_select=="all countries analysed":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["country"].isin(["Switzerland", "Germany"])], layer_name="Countries choosen", fill_colors=["red"])
        m.to_streamlit()
    
    if country_select=="Switzerland" and state_select=="all swiss cantons":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["country"]!=country_select], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["country"]==country_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()
    
    if country_select=="Switzerland" and state_select!="all swiss cantons":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["state"]!=state_select], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["state"]==state_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()

    if country_select=="Germany" and state_select=="all german states":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["country"]==country_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.add_gdf(gdf[gdf["country"]!=country_select], layer_name="Countries", fill_colors=["blue"])
        m.to_streamlit()
    
    if country_select=="Germany" and state_select!="all german states":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["state"]==state_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.add_gdf(gdf[gdf["state"]!=state_select], layer_name="Countries", fill_colors=["blue"])
        m.to_streamlit()

    if country_select=="Germany" or country_select=="Switzerland" or country_select=="all countries analysed":
        fig = px.line(df_value_counts_news.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on News within {country_select}')
    else:
        fig = px.line(df_value_counts_news.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on News within {state_select}')
    st.plotly_chart(fig, use_container_width=True)


with visual_col2:
    if country_select=="Germany" or country_select=="Switzerland" or country_select=="all countries analysed":
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on Telegram within {country_select}')
    else:
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on Telegram within {state_select}')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='margin-top: 150px;'</p>", unsafe_allow_html=True)

    if country_select=="Germany" or country_select=="Switzerland" or country_select=="all countries analysed":
        fig = px.line(df_value_counts_twitter.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on Twitter within {country_select}')
    else:
        fig = px.line(df_value_counts_twitter.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=f'Cluster over time on Twitter within {state_select}')
    st.plotly_chart(fig, use_container_width=True)

if source_select_df=="Telegram":
    st.dataframe(df_telegram_mod) 
if source_select_df=="Twitter":
    st.dataframe(df_twitter_mod)
if source_select_df=="News":
    st.dataframe(df_news_mod)
