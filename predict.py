import Levenshtein as ln
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import timeit

def predict_query(query, G, roots_list, ids_question_map, track_leafs):
    tokenized_query = query.split()
    len_tokenized_query = len(tokenized_query)
    node_list = []

    def recurssive_search(comparison_term, G, idx=1, root=False, node_list=[], forked=False):
        new_term = []
        for node in (roots_list if root else G[" ".join(comparison_term.split()[:-1])]):

            # don't auto-correct on last-word
            if ((ln.distance(comparison_term, node) <= 2) if idx != len_tokenized_query - 1 else tokenized_query[
                                                                                                     -1] in node):
                if idx != len_tokenized_query - 1:
                    node_list.extend(recurssive_search(node + " " + tokenized_query[idx + 1], G, idx + 1, root=False,
                                                       node_list=node_list, forked=True))
                else:
                    if node in G and node in track_leafs:
                        new_term.append(node)

        if forked:
            if idx == len_tokenized_query - 1:
                return [ids_question_map[child] for term in new_term for child in track_leafs[term]]

            return ""
        else:
            if len(tokenized_query) == 1:
                if idx == len_tokenized_query - 1:
                    return [ids_question_map[child] for term in new_term for child in track_leafs[term]]
            return node_list

    res = recurssive_search(tokenized_query[0], G, idx=0, root=True)
    return res


def get_embedding(embed, text):
    if isinstance(text, str):
        return embed([text]).numpy()
    return embed(text).numpy()


def rank_suggested_queries(embed, query, suggested_queries, kmeans):
    query_len = len(query)
    test_x = get_embedding(embed, suggested_queries)
    start_time = timeit.default_timer()
    query_embed = get_embedding(embed, query)
    sorted_res = np.argsort(np.max(cosine_similarity(test_x, kmeans.centroids + query_embed), axis=-1))[::-1]
    sorted_res = np.argsort(
        [c_score * (1 / (1 + ln.distance(query, suggested_queries[pos][:query_len]))) for pos, c_score in
         enumerate(np.max(cosine_similarity(test_x, kmeans.centroids + query_embed), axis=-1))])[::-1]
    filtered_queries = [suggested_queries[tag] for tag in sorted_res]
    end_time = timeit.default_timer()
    return filtered_queries, end_time-start_time