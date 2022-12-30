import math
import struct

import numpy as np
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from inverted_index_gcp import *
import re

# according to print in assignment 3
# NUM_OF_DOCUMENTS = 6348910
NUM_OF_DOCUMENTS = 2000
TUPLE_SIZE = 6

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenizer(text):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    query_tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    query_tokens = [token for token in query_tokens if token not in all_stopwords]
    return query_tokens


# assignment 4
def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    I changes the functions a bit - returns dictionary: {key:term, value: tf*idf}
    """

    epsilon = .0000001
    # total_vocab_size = len(index.term_total)
    # Q = np.zeros((total_vocab_size))
    Q = {}
    # term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            # df = index.df[token] # deleted
            # idf = math.log((NUM_OF_DOCUMENTS) / (df + epsilon), 10)  # smoothing  #deleted

            try:
                # ind = term_vector.index(token)
                # Q[ind] = tf * idf
                # Q[token] = tf * idf #deleted
                Q[token] = tf
            except:
                pass
    return Q


def cosine_similarity_given_a_query(index, query, DL):
    cosine_dict = {}
    docs_dict = create_docs_tf_dict(index, query, DL)  # tf normalized in doc_len
    query_dict = generate_query_tfidf_vector(query, index)  # tf*idf normalized in query len
    query_norm = sum([val ** 2 for val in query_dict.values()])

    print(docs_dict)
    for doc_id, term_dict in docs_dict.items():
        numerator_cosine_per_doc = 0
        for term in query_dict.keys():  # calculate tf * idf for each doc
            tf_doc = term_dict.get(term)
            tf_query = query_dict.get(term)
            idf = math.log2(len(DL) / index.df[term])  # i think its ok, # equation from lecture 2 slide 25
            if tf_doc is None or tf_query is None or idf is None:  # doc / query dont share term, irrelevant scenario
                continue
            numerator_cosine_per_doc += tf_doc * tf_query * idf
        # denominator_cosine_per_doc = math.sqrt(query_norm * (DL.get(doc_id)**2))
        denominator_cosine_per_doc = math.sqrt(query_norm * (1 ** 2))
        cosine_dict[doc_id] = numerator_cosine_per_doc / denominator_cosine_per_doc
    return cosine_dict


def create_docs_tf_dict(index, query, DL):
    """
        the function return dict {key:doc_id, value: dict {key:term, value: tf/doc_len}
    """
    doc_dic = {}
    # changes
    for term in query:
        term_psl = read_posting_list(index, term)
        for doc_id, tf in term_psl:
            if doc_id not in doc_dic.keys():
                doc_dic[doc_id] = {}
            if term not in doc_dic[doc_id].keys():
                # doc_dic[doc_id][term] = tf / DL.get(doc_id)  # equation (first) from lecture 2 slide 24
                doc_dic[doc_id][term] = tf / 1  # equation (first) from lecture 2 slide 24
    return doc_dic

    # tf_idf_of_query_dict = generate_query_tfidf_vector(query, index)
    # cosine_dict = {}
    #
    # for term in query:
    #     if term not in tf_idf_of_query_dict:
    #         tf_idf_of_query_dict[term] = 0
    #     try:
    #         term_psl = read_posting_list(index, term)
    #     except:
    #         continue
    #     for item in term_psl:
    #         if item[0] not in cosine_dict.keys():
    #             cosine_dict[item[0]] = 0
    #         cosine_dict[item[0]] += tf_idf_of_query_dict[term] * item[1]
    # return {k: v for k, v in sorted(cosine_dict.items(), key=lambda tup: tup[1], reverse=True)}


def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        print(locs)
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def cosine_similarity(index, query):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score
    Parameters:
    -----------
    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarity score.
    """
    # YOUR CODE HERE
    Q_norm = np.sqrt(np.sum(Q ** 2))
    return {doc_id: (np.dot(Q, doc) / (Q_norm * np.sqrt(np.sum(doc ** 2)))) for doc_id, doc in D.iterrows()}


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
    N: Integer (how many documents to retrieve). By default N = 3
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]
