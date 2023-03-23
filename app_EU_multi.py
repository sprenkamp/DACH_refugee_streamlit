import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
#check different leafmap backends
import leafmap.foliumap as leafmap
import openai
import tiktoken
#Config must be first line in script
st.set_page_config(layout="wide")

# Set OpenAI API key
openai.organization = "org-VnNq2FQvPmbE5cDNFayBZJHW"
openai.api_key = "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"

max_input_tokens=3900
max_tokens_output=500
encoding = "cl100k_base"

# calculate number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# run gpt
def run_gpt(prompt, max_tokens_output, timeout=60):
    completion = openai.ChatCompletion.create(
      model = 'gpt-3.5-turbo',
      messages = [
        {'role': 'user', 'content': prompt}
      ],
      max_tokens = max_tokens_output,
      n = 1,
      stop = None,
      temperature=0.5,
      timeout=timeout
    )
    return completion['choices'][0]['message']['content']

# create start prompt
def start_prompt_creator(message_type, cluster):
    if len(cluster) > 1:
        cluster = ", ".join(cluster)
    else:
        cluster = cluster[0]
    if message_type == "Telegram":
            start_prompt = f"looking at this telegram messages about {cluster} what are the up top 5 needs of refugees?"
            return start_prompt, cluster
    if message_type == "Twitter":
            start_prompt = f"looking at this twitter messages about {cluster} what are the up top 5 issues? If possibile focus refugees"
            return start_prompt, cluster
    if message_type == "News":
            start_prompt = f"looking at this news articles about {cluster} what are the up top 5 issues? If possibble focus on refugees"
            return start_prompt, cluster

# sample from df
def sample_df_gpt_analysis(df, start_prompt, max_input_tokens):
    current_input_tokens = num_tokens_from_string(start_prompt, encoding_name=encoding)
    text_list = []
    text_list.append(start_prompt)
    while max_input_tokens > current_input_tokens:
        df_sample = df.sample(n=1, replace=False)
        current_input_tokens += df_sample["tokens"].values[0]
        if current_input_tokens > max_input_tokens:
            break
        text_list.append(df_sample["text"].values[0])
    
    text = '\n'.join(text_list)
    return text

# write output to streamlit
def write_output(text, summary_select, cluster):
    st.header(f"Please find the output on {summary_select} messages about {cluster} below:")
    st.write(text)

# load geopandas data
gdf = gpd.read_file("data/germany_switzerland.geojson")

#function dummy space in sidebar
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

def dummy_function_space_small():
    st.write("\n")

#functions to load data
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

# manipulate data
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

# load data
df_telegram = load_telegram_data()
df_twitter = load_twitter_data()
df_news = load_news_data()
st.title('Identification of the most relevant topics in the context of the Ukrainian Refugee Crisis in the media and social media')

# create text columns for country, state and time selection
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
    summary_select = st.selectbox(
        'show summary of needs',
        ["Telegram", "Twitter", "News"],
        )
    calculate_summary = st.button("calculate summary")
    dummy_function_space_small()
    show_table = st.button('show data in table')

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

if calculate_summary:
    if summary_select=="Telegram":
        df_mod = df_telegram_mod
        cluster = cluster_select_telegram
    if summary_select=="Twitter":
        df_mod = df_twitter_mod
        cluster = cluster_select_twitter
    if summary_select=="News":
        df_mod = df_news_mod
        cluster = cluster_select_news

    dummy_text_summary = st.header("Calculating your summary ⏳😊")
    start_prompt, cluster_str = start_prompt_creator(message_type=summary_select, cluster=cluster)
    prompt = sample_df_gpt_analysis(df=df_mod, start_prompt=start_prompt, max_input_tokens=max_input_tokens-max_tokens_output)
    try:
        text = run_gpt(prompt, max_tokens_output, timeout=10)
    except openai.OpenAIError as e:
        text = "Sorry, request timed out. Please try again."
    dummy_text_summary.empty()
    write_output(text, summary_select, cluster_str)

if show_table:
    if summary_select=="Telegram":
        st.dataframe(df_telegram_mod) 
    if summary_select=="Twitter":
        st.dataframe(df_twitter_mod)
    if summary_select=="News":
        st.dataframe(df_news_mod)
