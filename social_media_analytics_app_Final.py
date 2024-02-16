import openai
import numpy as np
import pandas as pd
import time, os
import streamlit as st
import concurrent.futures
import json
from tqdm import tqdm_notebook
from tqdm import tqdm 
import re
import math
import plotly.express as px

from PIL import Image
import csv
import pickle
import os
import time
import chromadb
import openai
from dotenv import load_dotenv
import langchain
# from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFDirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import pandas as pd
import backoff
import requests, time
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
# from llama_index import SimpleDirectoryReader
import tempfile
from dotenv import load_dotenv
import openai
import tiktoken
from os.path import exists
import csv
import logging
logging.basicConfig(level=logging.DEBUG)
# <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
#<sup style='position: relative; top: 5px; color: #c05aaf;'>by Affine</sup>
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
st.markdown("""

    <div style='text-align: center; margin-top:-10px; margin-bottom: 5px;margin-left: -10px;'>
    <h2 style='font-size: 40px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                                  REVIEW INSIGHT TOOL
    </span>
    <span style='font-size: 40%;'>     
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.title("Select Genre")
    genre_list = ['Action-adventure','Sports','Shooter']
    genre = st.selectbox(label = "Choose a genre", options = genre_list)
    if genre == 'Action-adventure':
        games_list = ['Assassins Creed Valhalla']
    elif genre == 'Sports':
        games_list = ['FIFA']
    elif genre == 'Shooter':
        games_list = ['Call of Duty']
    st.title("Select Game Name")
    games = st.selectbox(label = "Choose a game", options = games_list)
    st.title("Select The Data File")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

# openai.api_type = "azure"
# openai.api_base = "https://allbirds.openai.azure.com/"
# openai.api_version = "2023-03-15-preview"

# Set your OpenAI API key here
# openai.api_key = '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'
# os.environ["OPENAI_API_KEY"] = "580084f0fe9941d1b815c49329668488"
# os.environ["OPENAI_API_KEY"] = '1dfa2422e0ba43a88044e87df4655c4c'--
# os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
# os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"--
# os.environ["OPENAI_API_BASE"] = "https://aipractices.openai.azure.com/"--
os.environ["OPENAI_API_BASE"] ="https://allbirds.openai.azure.com/"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = 'd67d19908eba4902af4903c270547bba'
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")

df = pd.DataFrame()

# Modify your analyze function to return a coroutine
def analyze(text, max_tokens=1024, stop=None):
    messages = [
       {"role": "user", "content": "Please extract similar aspects expressions which are present in the example, related segments, related sentiments and overall review sentiment from the following text, text segment should be tagged to any of the related or similar aspect there should not be the case where the text segment is not tagged to any of the aspect and format output in JSON."},
        {"role": "system", "content":"""example 1:
                                                {
                                                  "Review": "The game offers a lot of fun and keeps you engaged, which is a definite plus.Exploring the beautifully designed world is a treat, even though some areas feel lacking in detail. Character's progression adds depth to the experience...The story manages to hold your attention, although it might not be everyone's cup of tea.While it falls short in polish compared to its predecessors, it's still a decent game with potential....but the graphical issues can be frustrating."
                                                  "overall_satisfaction":  "NA",
                                                  "storytelling_and_narrative": 0,
                                                  "gameplay_mechanics_and_fun_factor":  1,
                                                  "open_world_and_exploration": 1,
                                                  "value_and_longevity": "NA",
                                                  "character_development": 1,
                                                  "technical_performance_and_bugs": -1,
                                                  "comparison_to_similar_games_or_prequels": 0,
                                                  "overall_satisfaction_segment": "NA",
                                                  "storytelling_and_narrative_segment": "The story manages to hold your attention, although it might not be everyone's cup of tea.",
                                                  "gameplay_mechanics_and_fun_factor_segment": "The game offers a lot of fun and keeps you engaged, which is a definite plus.",
                                                  "open_world_and_exploration_segment": "Exploring the beautifully designed world is a treat, even though some areas feel lacking in detail.",
                                                  "value_and_longevity_segment": "NA,
                                                  "character_development_segment": "Character's progression adds depth to the experience...",
                                                  "technical_performance_and_bugs_segment": "...but the graphical issues can be frustrating.",
                                                  "comparison_to_similar_games_or_prequels_segment": "While it falls short in polish compared to its predecessors, it's still a decent game with potential.",
                                                  "overall_review_sentiment": "positive"
                                                }

                                                example 2:

                                                {
                                                  "Review": "If youre an RPG fan you owe it to yourself to give The Witcher 3 a shot It rivals the best that Bethesda and Bioware have to offer and youll be hardpressed to find a better RPG this year",
                                                  "overall_satisfaction": "NA",
                                                  "storytelling_and_narrative": 1,
                                                  "gameplay_mechanics_and_fun_factor": "NA,
                                                  "open_world_and_exploration": 1,
                                                  "value_and_longevity": "NA",
                                                  "character_development": "NA",
                                                  "technical_performance_and_bugs": "NA",
                                                  "comparison_to_similar_games_or_prequels": 1,
                                                  "overall_satisfaction_segment": "NA",
                                                  "storytelling_and_narrative_segment": "If youre an RPG fan you owe it to yourself to give The Witcher 3 a shot",
                                                  "gameplay_mechanics_and_fun_factor_segment": "NA",
                                                  "open_world_and_exploration_segment": "It rivals the best that Bethesda and Bioware have to offer",
                                                  "value_and_longevity_segment": "NA",
                                                  "character_development_segment": "NA",
                                                  "technical_performance_and_bugs_segment": "NA",
                                                  "comparison_to_similar_games_or_prequels_segment": "youll be hardpressed to find a better RPG this year",
                                                  "overall_review_sentiment": "positive"
                                                }

                                                example 3:

                                                {
                                                  "Review": "I didnt like this game for a few reasons I didnt like the story and the gameplay was nothing new or special I would much rather play Assassins Creed 		or something The game had some glitches and it actually crashed my PS4",
                                                  "overall_satisfaction": -1,
                                                  "storytelling_and_narrative": -1,
                                                  "gameplay_mechanics_and_fun_factor": -1,
                                                  "open_world_and_exploration": "NA",
                                                  "value_and_longevity": "NA",
                                                  "character_development": "NA",
                                                  "technical_performance_and_bugs": -1,
                                                  "comparison_to_similar_games_or_prequels": -1,
                                                  "overall_satisfaction_segment": "I didnt like this game",
                                                  "storytelling_and_narrative_segment": "I didnt like the story",
                                                  "gameplay_mechanics_and_fun_factor_segment": "the gameplay was nothing new or special",
                                                  "open_world_and_exploration_segment": "NA",
                                                  "value_and_longevity_segment": "NA",
                                                  "character_development_segment": "NA",
                                                  "technical_performance_and_bugs_segment": "The game had some glitches and it actually crashed my PS4",
                                                  "comparison_to_similar_games_or_prequels_segment": "I would much rather play Assassins Creed or something",
                                                  "overall_review_sentiment": "negative"
                                                }
"""},
        {"role": "user", "content": text}
    ]

    response = openai.ChatCompletion.create(
        engine='chatgpt',
        messages=messages,
        max_tokens=max_tokens,
        stop=stop,
    )
    
    return response

#preprocessing on the reviews to remove unnecessary content like author name, any links, special character
def tweet_preprocessing(review):
# Remove all special characters
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)

    # Remove user mentions (@username)
    review = re.sub(r'@[a-zA-Z]+', '', review)

    # Remove URLs
    review = re.sub(r'http\S+', '', review)

    # Remove any leading or trailing whitespaces
    review = review.strip()

    # Remove repeated spaces
    review = re.sub(r'\s+', ' ', review)

    return review



if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        review_data = df.copy()
        df['cleaned_review']=df['text'].apply(tweet_preprocessing)
    

analysis_results = []
extra_prompts = []


def process_review(index, text):
    
    try:
        res = analyze(
            text=text,
            max_tokens=2500,  # Adjust max_tokens based on your needs
        )
        raw_json = res["choices"][0].message['content'].strip()
    except Exception as e:
        print("Exception1:", e,index)

    try:
        global analysis_results 
        global extra_prompts
        json_data = json.loads(raw_json)
        json_data['index'] = index
        analysis_results.append(json_data)
        # print(analysis_results)
        # log.debug(f"JSON response: {pprint(json_data)}")
        extra_prompts.append(f"\n{text}\n{raw_json}")
        
    except Exception as e:
        # global analysis_results
        # print("Exception 2",e,index)
        # log.error(f"Failed to parse '{raw_json}' -> {e}")
        analysis_results.append({'index':index})
        # analysis_results.append([])
# sentiment_data = None
def process_data(df):
    global sentiment_data
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor: 
        for i in range(0,len(df)):  # Adjust the range as needed
            text = df.loc[i, "cleaned_review"]
            executor.submit(process_review, i, text)

    df["analysis"] = analysis_results  
    # df.to_csv("aspects_for_social_media_app.csv", index=False) 
    
    data_list = df.analysis.tolist()
    sentiment_data = pd.DataFrame(data_list)
    sentiment_data.replace('NA', np.nan, inplace=True)
    sentiment_data = sentiment_data.fillna(-3)
    sentiment_data = sentiment_data.sort_values(by=['index'])
    # print(sentiment_data)
    return sentiment_data

def create_or_load_final_dataframe():
    if 'final_data' not in st.session_state:
        st.session_state.final_data = process_data(df)
    return st.session_state.final_data

    
if df.empty:
    centered_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 10vh;">
        <p style="text-align: center;">Upload csv File!</p>
    </div>
    """

    # Display the centered text
    st.markdown(centered_html, unsafe_allow_html=True)
    # st.markdown('Upload csv File!')
else:
    st.markdown("""
            <style>
                .st-dq {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
            </style>
            """, unsafe_allow_html=True)
    print("this runs 1")

    with st.spinner('Wait for it...'):
    
    # with st.form("df_form"):
        sentiment_data = create_or_load_final_dataframe()

        aspect_segment_data = sentiment_data.copy()

        print("this runs 2")
        positive_percentage_overall_satisfaction = math.ceil((sentiment_data['overall_satisfaction'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_storytelling_and_narrative = math.ceil((sentiment_data['storytelling_and_narrative'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_gameplay_mechanics_and_fun_factor = math.ceil((sentiment_data['gameplay_mechanics_and_fun_factor'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_open_world_and_exploration = math.ceil((sentiment_data['open_world_and_exploration'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_value_and_longevity = math.ceil((sentiment_data['value_and_longevity'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_character_development = math.ceil((sentiment_data['character_development'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_technical_performance_and_bugs = math.ceil((sentiment_data['technical_performance_and_bugs'] == 1).sum() / len(sentiment_data) * 100)
        positive_percentage_comparison_to_similar_games_or_prequels = math.ceil((sentiment_data['comparison_to_similar_games_or_prequels'] == 1).sum() / len(sentiment_data) * 100)

        # }
        data = {
                    "Aspect": ["Overall satisfaction", "Storytelling and narrative", "Gameplay mechanics and fun factor", "Open world and exploration", "Value and longevity", "Character development", "Technical performance and bugs", "Comparison to similar games or prequels"],
                    "Positive Percentage": [positive_percentage_overall_satisfaction, positive_percentage_storytelling_and_narrative, positive_percentage_gameplay_mechanics_and_fun_factor, positive_percentage_open_world_and_exploration, positive_percentage_value_and_longevity, positive_percentage_character_development, positive_percentage_technical_performance_and_bugs, positive_percentage_comparison_to_similar_games_or_prequels],
                }


        percentage_data = pd.DataFrame(data).set_index("Aspect").T
        percentage_data.index.name = 'Aspects'
        percentage_data.reset_index(inplace=True)
        
        # st.title("Game Review Analysis")
        st.markdown("""
            <div style='text-align: center;'>
            <h4 style='font-size: 25px; font-family: Arial, sans-serif; 
                            letter-spacing: 2px; text-decoration: none;'>
            <a href='https://affine.ai/' target='_blank' rel='noopener noreferrer'
                        style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                text-shadow: none; text-decoration: none;'>
                                Game Review Analysis
            </a>
            </h4>
            </div>
            """, unsafe_allow_html=True)
        st.write("Aspects Positive Percentage Ratings:")


        st.dataframe(percentage_data.iloc[:,1:], height=50, width=1500,hide_index=True)

    
    with st.form("review_form"):
        # selected_review = st.selectbox("Select a Review", list(sentiment_data["Review"][:20]))
        selected_review = st.selectbox("Select a Review", list(sentiment_data[(sentiment_data['Review'] != -3)]['Review']))
        
        # Create a form submit button
        submitted = st.form_submit_button("Submit")
    # Check if the form has been submitted
        if submitted:
            # Find the row corresponding to the selected review
            selected_aspect_values = sentiment_data[sentiment_data["Review"] == selected_review]

            # Display aspect values for the selected review
            st.write("Aspect Values for the Selected Review:")
            variables = ['storytelling_and_narrative','gameplay_mechanics_and_fun_factor','open_world_and_exploration','value_and_longevity','character_development','technical_performance_and_bugs','comparison_to_similar_games_or_prequels','overall_review_sentiment']
            for i in variables:
                if i != 'overall_review_sentiment':
                    selected_aspect_values[i] = selected_aspect_values[i].astype(int)
            # selected_aspect_values = selected_aspect_values.loc[:,int_variables.columns].T.reset_index()
            
            # selected_aspect_values = selected_aspect_values.loc[:, selected_aspect_values.columns != 'overall_review_sentiment'].apply(pd.to_numeric)
            selected_aspect_values = selected_aspect_values.loc[:,variables].T.reset_index()
            # print(selected_aspect_values)
            
            selected_aspect_values = pd.DataFrame(selected_aspect_values.set_axis(['Aspects', 'Aspects Sentiments'], axis='columns'))
            selected_aspect_values = selected_aspect_values[(selected_aspect_values['Aspects Sentiments'] != -3)]
            # selected_aspect_values = selected_aspect_values['Aspects Sentiments']
            st.dataframe(selected_aspect_values,use_container_width=True,hide_index=True)
            # st.dataframe(selected_aspect_values.iloc[:1,:9].set_index('Review').T,use_container_width=True)
            
            legend = """
               -1 : Negative\n
                0 : Neutral\n
                1 : Positive\n
            """
            # Define CSS style for the legend box
            legend_style = """
                background-color: black;
                color: white;
                padding: 10px;
                border-radius: 5px;
            """

            # Display the legend box with the text
            st.write('<div style="{}">Legend: -1 : Negative\n 0 : Neutral\n 1 : Positive\n</div>'.format(legend_style), unsafe_allow_html=True)

            
            # st.markdown(legend)
            
        # st.success('Task Completed!')


#####################################Sumarrization##############################################

# st.title("Summarised Game Reviews")

st.markdown("""
<div style='text-align: center;'>
<h4 style='font-size: 25px; font-family: Arial, sans-serif; 
                   letter-spacing: 2px; text-decoration: none;'>
<a href='https://affine.ai/' target='_blank' rel='noopener noreferrer'
               style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      text-shadow: none; text-decoration: none;'>
                      Summarised Game Reviews
</a>
</h4>
</div>
""", unsafe_allow_html=True)

# games_list = ["Assassin's Creed Valhalla","The Witcher 3: Wild Hunt", "Immortals Fenyx Rising","Mass Effect: Andromeda", "Dark Souls III"]
aspects_segment_list = ['overall_satisfaction_segment', 'storytelling_and_narrative_segment', 'gameplay_mechanics_and_fun_factor_segment',
                            'open_world_and_exploration_segment', 'value_and_longevity_segment', 'character_development_segment',
                        'technical_performance_and_bugs_segment', 'comparison_to_similar_games_or_prequels_segment']
aspects_list = ['overall_satisfaction', 'storytelling_and_narrative', 'gameplay_mechanics_and_fun_factor', 'open_world_and_exploration',
                    'value_and_longevity', 'character_development', 'technical_performance_and_bugs', 'comparison_to_similar_games_or_prequels']


# output_directory = "positive_output"


# os.makedirs(output_directory, exist_ok=True)
# Function to count the number of tokens
def num_tokens(text: str, Encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(Encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def analyze_and_save(max_tokens=2500, stop=None):
    # for game in games_list:
    game_info = []  
    for segment, aspect in zip(aspects_segment_list, aspects_list):
        try:
            reviews = aspect_segment_data[(aspect_segment_data[aspect] == 1) & (aspect_segment_data[segment] != -3)][segment].tolist()
            reviews_text = ""
            counts=0
            for review in reviews:
                
                # next_article = review
                if (
                    num_tokens(reviews_text + review)
                    > max_tokens
                ):
                    break
                else:
                    counts+=1
                    
                    reviews_text += review
            # st.text(counts)
            print('^^^^^^^^^^^^^^^^^^tocken count^^^^^^^^^^^^^^^^',counts)
            # reviews_text = "\n".join(reviews[:20])
            
            messages = [
                {"role": "user", "content": "Please summarize the paragraph in 2-3 lines if there is no paragraph available to summarize then say No one has talked about thgis aspect in the reviews do not try to make up answerds on your own"},
                {"role": "system", "content": reviews_text},
            ]
            
            response = openai.ChatCompletion.create(
                engine='chatgpt',
                messages=messages,
                max_tokens=max_tokens,
                stop=stop,
            )
            print('summarization started:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(response)
            game_info.append(f"{aspect}:\n{response['choices'][0].message['content'].strip()}\n")

        except Exception as e:
            print(e)
    return  game_info




if uploaded_file is not None:
    print("Im here 1...")
    if 'review_summarization_1' not in st.session_state:
        print("Im here 2...")
        with st.spinner('Summrizing the reviews for you...'):
            game_info = analyze_and_save(max_tokens=3000)
            st.session_state.review_summarization_1 = "\n".join(game_info)
    st.markdown(st.session_state.review_summarization_1)


###################################### Chat bot##################################

st.markdown("""
<div style='text-align: center;'>
<h4 style='font-size: 25px; font-family: Arial, sans-serif; 
                   letter-spacing: 2px; text-decoration: none;'>
<a href='https://affine.ai/' target='_blank' rel='noopener noreferrer'
               style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      text-shadow: none; text-decoration: none;'>
                      Chat Bot
</a>
</h4>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:

    with st.spinner('Bot is getting ready for you...'):
        # temp_dir = tempfile.TemporaryDirectory()

        # temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        # with open(temp_file_path, 'wb') as temp_file:
        #     temp_file.write(uploaded_file.read())

        # print("//////////////////////",review_data)
        
        
        review_data.iloc[:,1].to_csv('dataset.csv',index=False)
        loader = CSVLoader('dataset.csv', encoding='utf-8')  # Adjust the encoding as needed
        print(loader)
        data = loader.load()
        documents = data[:100]
        print(documents)


        # loader = CSVLoader(file_path=temp_file_path, encoding='utf-8')  # Adjust the encoding as needed

        # data = loader.load()
        # documents = data[1:]## needs to change

        user_input_chunk_size = 5000
        user_input_chunk_overlap =0
        embedding_save_path = "test_chroma_db"
        collection_name = "chroma_vectorstore"



        text_splitter = RecursiveCharacterTextSplitter(chunk_size = user_input_chunk_size,
                                        chunk_overlap = user_input_chunk_overlap,
                                        length_function = len,separators="\n")

        document_chunks = text_splitter.split_documents(documents)

        len(document_chunks)
    
        # __file__ = "./social_media_analytics_app_Final.py"
        # __file__ = "D:/Sony_Project/Social Media Analytics/Sentiment_analysis_using_gpt/notebooks/social_media_analytics_app_v1_final"
        __file__ = "./"
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        # DB_DIR = os.path.join(ABS_PATH, temp_file_path.split("\\")[-1].split('.')[0])
        DB_DIR = os.path.join(ABS_PATH, f'{uploaded_file.name[:-4]}')
        # print("Directory name/////////////////////////",DB_DIR)


# ada-text-embeddings-002

        # file_exists = exists(DB_DIR)

        embeddings = OpenAIEmbeddings(deployment ="text-embedding-ada-002", model = "text-embedding-ada-002", chunk_size = 1, max_retries = 5)
        # LoadData to vectorstore
        client_settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=DB_DIR,
                anonymized_telemetry=False
            )
        if not exists(DB_DIR):

            # initialised vectorstore
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                client_settings=client_settings,
                persist_directory=DB_DIR,
            )
            # inserting documents one-by-one to it using add_documents:
            vectorstore.add_documents(documents=document_chunks, embedding=embeddings)
        
            vectorstore.persist()
            # print(vectorstore)

        vectorstore = Chroma(
            embedding_function=embeddings,
            client_settings=client_settings,
            persist_directory=DB_DIR,
        )

        custom_prompt_template = """Use the following pieces of context about the game to answer the question at the end. If you cannot extract the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer only from the given document chunk in English:"""

        PROMPT = PromptTemplate(
                        template = custom_prompt_template, input_variables = ["context", "question"]
                        )
        chain_type_kwargs = {"prompt": PROMPT}

        llm = AzureChatOpenAI(temperature = 0, 
                            deployment_name = "chatgpt",
                            openai_api_version = "2023-07-01-preview",
                            openai_api_key = "d67d19908eba4902af4903c270547bba",
                            openai_api_base = "https://allbirds.openai.azure.com/",
                            openai_api_type = "azure",
                            #deployment_name = model,
                            # model_kwargs = {
                            #                 "api_key": os.environ["OPENAI_API_KEY"],
                            #                 "api_base": os.environ["OPENAI_API_BASE"],
                            #                 "api_type": os.environ["OPENAI_API_TYPE"],
                            #                 "api_version": os.environ["OPENAI_API_VERSION"],
                            #                 }
                )

        chroma_db = Chroma( collection_name=collection_name,
            embedding_function=embeddings,
            client_settings=client_settings,
            persist_directory=DB_DIR
                        )


        def get_response(query):

            for k in range(100,0,-5):
                custom_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                    retriever=chroma_db.as_retriever(type = "similarity", search_kwargs={"k":k}),
                                                    chain_type_kwargs=chain_type_kwargs,
                                                    return_source_documents = True
                                                    )
                        

                try:
                    return custom_qa({"query": query}),k
                    break

                except:
                    pass

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    start=time.time()
                    with get_openai_callback() as cz:

                        # full_response = get_response(prompt)[0]
                        full_response = get_response(prompt)[0]
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@@@@@@@@@@@@@2",full_response)
                    print(cz)
                    end=time.time()
                    total_time=end-start
                    message_placeholder.markdown(full_response['result'])
                    print("*"*50)
                    print(full_response)
                    print(f"Total Time :: {round(total_time)} sec")
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response['result']}
                )
        # query = st.text_input("Ask Question:")

        # # Display the user input
        # st.write("You entered:", query)


        # Test single query:
        # import csv

        # print("Query is --->",  query)
        # if query:
        #     with get_openai_callback() as cb: 
        #         # to check response time:
        #         start_time = time.time()
                
        #         #response=custom_qa({"query": query})
        #         response,number_of_relevant_reviews = get_response(query)
        #         response_time = time.time() - start_time

        #         cb = str(cb)
        #         with open('data8.csv', mode='a',errors="ignore", newline='') as file:
        #             writer = csv.writer(file)

        #             total_tokens_used = cb.split(':')[1].split()[0]
        #             prompt_tokens = cb.split(':')[2].split()[0]
        #             completion_tokens = cb.split(':')[3].split()[0]
        #             successful_requests = cb.split(':')[4].split()[0]
        #             total_llm_cost = cb.split(':')[5].strip()
        #             source_document=[]
        #             matched_chunk_list=[]

        #             for i, doc in enumerate(response["source_documents"]):
        #                 source_document.append(doc.metadata) # document name
        #                 matched_chunk = doc.page_content.encode('utf-8')
        #                 matched_chunk = matched_chunk.decode('utf-8')
        #                 matched_chunk = matched_chunk.replace("\t"," ")
        #                 matched_chunk = matched_chunk.replace("\n"," ")
        #                 matched_chunk_list.append(matched_chunk)
        #             query = query.encode('utf-8')
        #             answer = str(response['result']).encode('utf-8')
        #             answer = answer.decode('utf-8')
        #             num_words_in_answer = len(answer.split(' '))
        #             print(query)
        #             print(answer)
        #             print(num_words_in_answer)
        #             print(response_time)
        #             print(source_document)
        #             print(matched_chunk_list)
        #             print("No. of reviews considered: ",number_of_relevant_reviews)
        #             print(total_tokens_used)
        #             print(prompt_tokens)
        #             print(completion_tokens)
        #             print(successful_requests)
        #             print(total_llm_cost)
        #             writer.writerow([query, answer, num_words_in_answer, response_time, source_document, matched_chunk_list,number_of_relevant_reviews, total_tokens_used, prompt_tokens, completion_tokens, successful_requests, total_llm_cost])

            # st.markdown(
            #     f'<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">'
            #     f'<b>User Input:</b> {answer}'
            #     f'</div>',
            #     unsafe_allow_html=True
            # )
        # temp_dir.cleanup()
