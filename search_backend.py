from nltk.corpus import stopwords
from inverted_index_gcp import *
import numpy as np
import nltk
import math
import re

# according to print in assignment 3
NUM_OF_DOCUMENTS = 6348910
TUPLE_SIZE = 6
# values for BM25
BM25_K1 = 3  # between 0-3
BM25_b = 0.13  # between 0-1
# values for pagerank norm
MAX_PR = 9913.72878216078  # belongs to doc_id 3434750
MIN_PR = 0.1501208493870428  # belongs to doc_id 1404

nltk.download('stopwords')

def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.
    -----------
    Parameters:
    text: text to tokenize.
    -----------
    Returns:
    stopwords filtered list of tokens.
    """
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def read_posting_list(inverted, w, index_loc):
    """
    This function returns the posting list of the word/term specified in the given inverted index.
    -----------
    Parameters:
    inverted: inverted index file.
    w: word/term that we want to get its posting list.
    index_loc: location name of the inverted index file.
    -----------
    Returns:
    posting list of the word/term specified.
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, index_loc)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def doc_binary_search(index, query, index_loc):
    """
    This function returns a sorted list by the binary score of ALL doc_ids that query terms appears in.
    -----------
    Parameters:
    index: inverted index file in which we want to search.
    query: text representing the query in which we want to search.
    index_loc: location name of the inverted index file.
    -----------
    Returns:
    a sorted list by the binary score of ALL doc_ids that query terms appears in.
    """
    dict_query_docs = {}
    for query_term in np.unique(tokenize(query)):  # if a query consist one term more than once, we will count it once
        try:
            term_psl = read_posting_list(index, query_term, index_loc)
        except:  # avoid terms that do not appear in the index
            continue
        for doc_id, tf in term_psl:
            if doc_id in dict_query_docs:
                dict_query_docs[doc_id] += 1
            else:
                dict_query_docs[doc_id] = 1
    return sorted(dict_query_docs, key=dict_query_docs.get, reverse=True)


def cosine_similarity_tf_idf(index, tokenized_query, DL):
    """
    This function computes tfidf cosine similarity between the query and all the relevant docs in the index.
    and returns a dictionary {key: doc_id, value: query-doc_id tfidf cosine similarity score}.
    -----------
    Parameters:
    index: inverted index file in which we want to search.
    tokenized_query: list of the query tokens.
    DL: dictionary {key: doc_id, value: len of doc (in tokens)}.
    -----------
    Returns: a dictionary as follows: {key: doc_id, value: query-doc_id tfidf cosine similarity score}.
    """
    cosine_dict = {}
    final_cosine_dict = {}
    sum_doc_squared = {}
    sum_query_squared = 0
    query_len = len(tokenized_query)
    query_counter = Counter(tokenized_query)
    epsilon = .0000001
    for term in tokenized_query:
        try:
            term_psl = read_posting_list(index, term, 'postings_gcp')
        except:  # avoid terms that do not appear in the index
            continue
        tf_query = query_counter[term] / query_len
        idf = math.log(NUM_OF_DOCUMENTS/(index.df[term]+epsilon), 10)  # smoothing
        tf_idf_query_score = tf_query * idf
        sum_query_squared += tf_idf_query_score ** 2
        for doc_id, tf in term_psl:
            tf_doc = tf / DL[doc_id]
            tf_idf_doc_score = tf_doc * idf
            if doc_id not in sum_doc_squared.keys():
                sum_doc_squared[doc_id] = tf_idf_doc_score ** 2
            else:
                sum_doc_squared[doc_id] += tf_idf_doc_score ** 2
            if doc_id not in cosine_dict.keys():
                cosine_dict[doc_id] = 0
            cosine_dict[doc_id] += tf_idf_query_score * tf_idf_doc_score
    for doc_id, score in cosine_dict.items():
        final_cosine_dict[doc_id] = cosine_dict[doc_id] / DL[doc_id] * math.sqrt(sum_doc_squared[doc_id])
    return final_cosine_dict

def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores.
    -----------
    Parameters:
    sim_dict: dictionary of similarity score as follows: {key: doc_id, value: score}
    N: Integer (how many documents to retrieve). By default N = 3
    -----------
    Returns:
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


#### merge function from assignment 4
def merge_results(title_scores, BM25_scores, title_weight=0.6, bm25_weight=0.4):
    """
    this function merges the title_scores dict with the BM25_scores dict to one dict.
    -----------
    Parameters:
    title_scores: dictionary of title scores as follows: {key: doc_id, value: title_score}
    BM25_scores: dictionary of BM25 scores as follows: {key: doc_id, value: BM25_score}
    title_weight: the weight we want to give to the title in the formula.
    bm25_weight: the weight we want to give to the BM25 in the formula.
    -----------
    Returns:
    a dictionary of merged scores as follows: {key: doc_id, value: merged_score}
    """
    merged_results = {}  # dictionary {key: doc_id, value: weighted score}
    # iterate on title_scores
    for doc_id, score in title_scores.items():
        if doc_id not in merged_results.keys():
            merged_results[doc_id] = 0
        merged_results[doc_id] += score * title_weight
    # iterate on BM25 scores
    for doc_id, score in BM25_scores.items():
        if doc_id not in merged_results.keys():
            merged_results[doc_id] = 0
        merged_results[doc_id] += score * bm25_weight
    # sort dictionary by score
    merged_results = {key: value for key, value in
                      sorted(merged_results.items(), key=lambda item: item[1], reverse=True)}
    return merged_results


def query_in_title_docs_norm(index, query):
    """
    this function finds the most relevant docs according to title and returns a normed sorted dictionary as follows:
    {key: doc_id, value: number of tokens from the query that found in the doc's title / len of query}
    the sort is by the values.
    -----------
    parameters :
    index: title inverted index file.
    query: list of tokens represent the query.
    -----------
    return:
    a sorted dictionary as follows: {key: doc_id, value: number of tokens from the query that found in the doc's title / len of query}
    the sort is by the values.
    """
    dict_query_docs = {}
    query_len = len(query)
    for term in np.unique(query):  # if a query consist one term more than once, we will count it once
        try:
            term_psl = read_posting_list(index, term, 'postings_gcp_title_big')
        except:
            continue
        for doc_id, tf in term_psl:
            if doc_id not in dict_query_docs.keys():
                dict_query_docs[doc_id] = 0
            dict_query_docs[doc_id] += 1 / query_len
    dict_query_docs = {key: value for key, value in
                       sorted(dict_query_docs.items(), key=lambda item: item[1], reverse=True)}
    return dict_query_docs


def normalize_dict(myDict):
    """
    this function normalize the values of the dictionary using Min-Max normalization so that all scores are between 0-1
    -----------
    parameters :
    myDict: dictionary we want to normalize.
    -----------
    return:
    dictionary with normalized values.
    """
    if myDict == {}:
        return {}
    normalized = {}
    x_min = min(list(myDict.values())[:100])
    # x_min = min(list(myDict.values()))
    x_max = max(list(myDict.values()))
    for doc_id, score in myDict.items():
        x = myDict[doc_id]
        normalized[doc_id] = (x - x_min) / (x_max - x_min)
    return normalized


def new_search_using_BM25(query_tokens, index, DL_dict):
    """
    this function calculates BM25 for all relevant docs to a given query,
    all parameters of the BM25 equation are init in the top of the file BM25_b, BM25_K1 and can be set there.
    -----------
    Parameters:
    query_tokens: list of tokens represent the query.
    index: relevant index to search - textIndex.
    DL_dict: dictionary {key: doc_id, value: len of doc (in tokens)}.
    -----------
    Returns:
    a sorted dictionary by BM25 score as follows: {key: doc_id, value: BM25 score}.
    """
    scores = {}  # dict of {key: doc_id, value: BM25 value given a query}
    idf = calc_idf(query_tokens, index)
    AVGDL = sum(DL_dict.values()) / NUM_OF_DOCUMENTS
    for term in query_tokens:
        try:
            # term_psl = read_posting_list(index, term, 'postings_gcp')
            term_frequencies = dict(read_posting_list(index, term, 'postings_gcp'))
        except:
            continue
        for doc_id, tf in term_frequencies.items():  # tuple of (doc_id, tf)
            if doc_id not in scores.keys():
                scores[doc_id] = 0
            freq = term_frequencies[doc_id]
            numerator = idf[term] * freq * (BM25_K1 + 1)
            denominator = freq + BM25_K1 * (1 - BM25_b + BM25_b * DL_dict.get(doc_id) / AVGDL)
            scores[doc_id] += (numerator / denominator)
    sorted_scores = {key: value for key, value in
                     sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    return sorted_scores

def calc_idf(list_of_tokens, index):
    """
    This function calculate the idf values according to the BM25 idf formula for each term in the query.
    -----------
    Parameters:
    query: list of token representing the query. For example: ['look', 'blue', 'sky'].
    -----------
    Returns:
    a dictionary of idf scores as follows: {key: term, value: bm25 idf score}.
    """
    idf = {}
    for term in list_of_tokens:
        if term in index.df.keys():
            n_ti = index.df[term]
            idf[term] = math.log(1 + (NUM_OF_DOCUMENTS - n_ti + 0.5) / (n_ti + 0.5))
        else:
            pass
    return idf
