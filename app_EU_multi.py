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

gdf = gpd.read_file("data/europe.geojson")

@st.cache()
def load_telegram_data():
    df = pd.read_csv("data/df_telegram.csv")
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
    # df_value_counts['occurence_count_norm'] = df_value_counts.apply(lambda x: x.occurence_count/list(messages_per_week_dict.values())[list(messages_per_week_dict.keys()).index(x.week)], axis=1)
    return df_value_counts

def modify_df_for_table(df_mod, region_select, cluster_select, date_slider, metric_select=None):
    if region_select!="all Countries analysed":
        df_mod = df_mod[df_mod.region==region_select]
    if not "all found clusters" in cluster_select:
        df_mod = df_mod[df_mod.cluster.isin(cluster_select)]
    df_mod = df_mod[df_mod.date.between(date_slider[0], date_slider[1])]
    return df_mod

df_telegram = load_telegram_data()
df_twitter = load_twitter_data()
df_news = load_news_data()
st.title('Identification of the most relevant topics in the context of the Ukrainian crisis in the media and social media')

text_col1, text_col2, text_col3  = st.columns(3)
with text_col1:
    region_selection = ["all Countries analysed", "Germany", "Austria", "Switzerland"]
    region_select = st.selectbox(
        "Select a country of interest",
        region_selection,
        )
with text_col2:
    # cluster = df_telegram.cluster.unique().tolist()
    # cluster.remove("no_class") 
    cluster = ["all found clusters"] + list(range(25)) 
    cluster_select = st.multiselect(
        'Choose the topic of interest',
        cluster,
        ["all found clusters"]
        )

with text_col3:
    # date_list = pd.date_range(start=df_telegram.date.min(), end=df_telegram.date.max(), freq='D').tolist()    
    # print(df_twitter.date.min(), type(df_twitter.date.min()))
    date_slider = st.slider('Choose date range of interest',
        min_value=df_telegram.date.min(), 
        value=(df_telegram.date.min(), df_telegram.date.max()), 
        max_value=df_telegram.date.max()
        )

df_telegram_mod = modify_df_for_table(df_mod=df_telegram, region_select=region_select, cluster_select=cluster_select, date_slider=date_slider)
df_value_counts_telegram = create_df_value_counts(df=df_telegram_mod)
df_twitter_mod = modify_df_for_table(df_mod=df_twitter, region_select=region_select, cluster_select=cluster_select, date_slider=date_slider)
df_value_counts_twitter = create_df_value_counts(df=df_twitter_mod)    
df_news_mod = modify_df_for_table(df_mod=df_news, region_select=region_select, cluster_select=cluster_select, date_slider=date_slider)
df_value_counts_news = create_df_value_counts(df=df_news_mod)    

visual_col1, visual_col2= st.columns(2)
with visual_col1:
    if region_select=="all Countries analysed":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["NAME"].isin(["Austria", "Switzerland", "Germany"])], layer_name="Countries", fill_colors=["red"])
        m.to_streamlit()
    
    if region_select!="all Countries analysed":
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7)
        m.add_gdf(gdf[gdf["NAME"]!=region_select], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["NAME"]==region_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()

    
    st.text(f'We identified the following clusters on traditional news within {region_select}')
    fig = px.line(df_value_counts_news.sort_values(['date']), x="date", y="occurence_count", color='cluster', title='Cluster over time')
    st.plotly_chart(fig, use_container_width=True)

with visual_col2:
    st.text(f'We identified the following clusters on Telegram within {region_select}')
    fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='cluster', title='Cluster over time')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='margin-top: 112px;'</p>", unsafe_allow_html=True)


    st.text(f'We identified the following clusters on Twitter within {region_select}')
    fig = px.line(df_value_counts_twitter.sort_values(['date']), x="date", y="occurence_count", color='cluster', title='Cluster over time')
    st.plotly_chart(fig, use_container_width=True)

