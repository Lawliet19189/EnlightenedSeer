import streamlit as st
import networkx as nx
import datetime
from ast import literal_eval
from predict import predict_query, rank_suggested_queries
import pickle
from annotated_text import annotated_text
import timeit

import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import tensorflow as tf

from predict import get_embedding
import faiss
import gc
from numba import cuda

# -- Set page config
apptitle = 'Search Query Completion'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:", layout="wide")


@st.cache(allow_output_mutation=True)
def load_artefacts(version=1):
    gc.collect()
    #device = cuda.get_current_device()
    #device.reset()

    if version==1:
        G = nx.read_gpickle("index/query_graph.gpickle")
        print("Loaded Query Graph")
        roots_list = [i for i, j in G.nodes(data="starts", default=1) if j == True]
        ids_question_map = pickle.load(open("index/ids_question_map.pkl", "rb"))
        question_ids_map = pickle.load(open("index/question_ids_map.pkl", "rb"))
        track_leafs = pickle.load(open("index/track_leafs.pkl", "rb"))
        print("Loaded extra artefacts")
        return G, roots_list, ids_question_map, question_ids_map, track_leafs
    else:
        G = nx.read_gpickle("index/v2/query_graph.gpickle")
        print("Loaded Query Graph")
        roots_list = [i for i, j in G.nodes(data="starts", default=1) if j == True]
        ids_question_map = pickle.load(open("index/v2/ids_question_map.pkl", "rb"))
        question_ids_map = pickle.load(open("index/v2/question_ids_map.pkl", "rb"))
        track_leafs = pickle.load(open("index/v2/track_leafs.pkl", "rb"))
        print("Loaded extra artefacts")
        gpus = tf.config.experimental.list_physical_devices('GPU')

        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        assert get_embedding(embed, "Hello World").shape == (1, 512)
        print("Loaded USE model")

        X = np.load(open("index/v2/user_queries.npy", "rb"))
        D = 512
        K = 10
        kmeans = faiss.Kmeans(d=D, k=round(4 * (X.shape[0] ** (1 / 2))), niter=20, verbose=True, gpu=True)
        kmeans.train(X.astype(np.float32))
        print("Loaded Kmeans centroids")

    #return G, deepcopy(roots_list), deepcopy(ids_question_map), deepcopy(question_ids_map), deepcopy(track_leafs)
    return G, roots_list, ids_question_map, question_ids_map, track_leafs, embed, kmeans


st.sidebar.markdown("## Configurations")
select_event = st.sidebar.selectbox('Training set used',
                                    ['Please select from the dropdown', 'MS-marco Articles', 'MS-marco Query & Articles'])

if select_event == 'MS-marco Articles':
    with st.spinner(text='Loading Artefacts... should aprox. take less than 2 min'):
        G, roots_list, ids_question_map, question_ids_map, track_leafs = load_artefacts(version=1)

    st.title("Search Query Completion using only MS-MARCO corpus articles")
    query = st.text_input("Enter Search Query", "hwat ia the lst day to file")
    if st.button("Autocomplete Query Suggest!"):
        query = query.lower()
        with st.spinner(text='Suggesting Autocomplete query'):
            start_time = timeit.default_timer()
            suggested_queries = predict_query(query, G, roots_list, ids_question_map, track_leafs)
            end_time = timeit.default_timer()
        if suggested_queries!=[]:
            st.warning("Showing " + str(min(len(suggested_queries), 10)) + " results out of " + str(
                len(suggested_queries)) + " (Not-sorted by any relevance)")
        else:
            st.warning("Sorry! No Suggested queries Found!")

        annotated_text(("Suggested Queries " + "in " + str(round(end_time - start_time, 2)) + "s", "", "#afa"), height=40)
        st.markdown("""---""")
        #st.write(suggested_queries)
        for idx, q in enumerate(suggested_queries[:10]):
            annotated_text((str(idx+1) + ". ", "", "#8ef"), (q, "", "#faa"), height=40)
        #st.write("Yup!")
elif select_event == 'MS-marco Query & Articles':
    with st.spinner(text='Loading Artefacts... should aprox. take less than 4 min'):
        G, roots_list, ids_question_map, question_ids_map, track_leafs, embed, kmeans = load_artefacts(version=2)

    st.title("Search Query Completion using MS-MARCO corpus articles & user queries")

    query = st.text_input("Enter Search Query", "hwat ia the lst day to file")
    if st.button("Autocomplete Query Suggest!"):
        query = query.lower()

        with st.spinner(text='Suggesting Autocomplete query'):
            start_time = timeit.default_timer()
            suggested_queries = predict_query(query, G, roots_list, ids_question_map, track_leafs)[:500]
            end_time = timeit.default_timer()
            extra_time = 0
            if suggested_queries!=[]:
                suggested_queries, extra_time = rank_suggested_queries(embed, query, suggested_queries, kmeans)
                st.warning("Showing " + str(min(len(suggested_queries), 10))  + " results out of " + str(len(suggested_queries)) + " (Sorted by relevance)")
            else:
                st.warning("Sorry! No Suggested queries Found!")
        annotated_text(("Suggested Queries " + "in " + str(round(end_time - start_time + extra_time, 2)) + "s", "", "#afa"), height=40)
        st.markdown("""---""")
        #st.write(suggested_queries)
        for idx, q in enumerate(suggested_queries[:10]):
            annotated_text((str(idx+1) + ". ", "", "#8ef"), (q, "", "#faa"), height=40)
        #st.write("Yup!")
else:
    st.header("Welcome to our Query Autocomplete Demo! Please select a dataset from left-navbar dropbox")
    st.info("For implementation details of the demo, please refer our Github repository: https://github.com/Lawliet19189/EnlightenedSeer")
    st.markdown("""---""")
    st.warning("Since the demo is deployed locally, the UI/UX will have a slight delay!")