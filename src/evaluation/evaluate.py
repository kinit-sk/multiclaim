"""
A library of functions that can be used to evaluate the results produced by _result generators_. Result generators are all the different methods
that can be used to retrieve fact-checks.

`process_result_generator` is the main API that should be used to create evaluation results.

Currently supported:
    bm25 - BM25
    dummy - Returns fixed order of fact-checks. Used for debugging.
    embedding - General method that supports different embedding models and then calculate cosine similarity between the vectors.
"""
from collections import defaultdict
import logging
from typing import Generator

import numpy as np
import pandas as pd

from evaluation.metrics import standard_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predicted_ranks(predicted_ids: np.array, desired_ids: np.array, default_rank: int = None):
    """
    Return sorted ranks of the `desired_ids` in the `predicted_ids` array.
    
    If `default_rank` is set, the final array will be padded with the value for all the ids that were not present in the `predicted_ids` array.
    """
    
    predicted_ranks = dict()
    
    for desired in desired_ids:
        
        try:
            rank = np.where(predicted_ids == desired)[0][0] + 1  # +1 so that the first item has rank 1, not 0
        except IndexError:
            rank = default_rank
       
        if rank is not None:
            predicted_ranks[desired] = rank
        
    return predicted_ranks


def process_result_generator(gen: Generator, default_rank: int = None, csv_path: str = None):
    """
    Take the results generated from `gen` and process them. By default, only calculate metrics, but dumping the results into a csv file is also supported
    via `csv_path` attribute. For `default_rank` see `predicted_ranks` function.
    """
    
    ranks = list()
    rows = list()
    
    for predicted_ids, desired_ids, post_id in gen:
        post_ranks = predicted_ranks(predicted_ids, desired_ids, default_rank)
        ranks.append(post_ranks.values())
        
        if csv_path:
            rows.append((post_id, post_ranks, predicted_ids[:100]))
            
    logger.info(f'{sum(len(query) for query in ranks)} ranks produced.')
            
    if csv_path:
        pd.DataFrame(rows, columns=['post_id', 'desired_fact_check_ranks', 'predicted_fact_check_ids']).to_csv(csv_path, index=False)
      
    return standard_metrics(ranks)


def result_generator(func):
    """
    This is a decorator function that should be used on result generators. The generators return by default: `predicted_fact_check_ids` and `post_id`.
    Here, the results are enriched with the `desired_fact_check_ids` as indicated by `dataset.fact_check_post_mapping`
    """
    
    def wrapper(dataset, *args, **kwargs):
        
        desired_fact_check_ids = defaultdict(lambda: list())
        for fact_check_id, post_id in dataset.fact_check_post_mapping:
            desired_fact_check_ids[post_id].append(fact_check_id)
        
        for predicted_fact_check_ids, post_id in func(dataset, *args, **kwargs):
            yield predicted_fact_check_ids, desired_fact_check_ids[post_id], post_id
        
    return wrapper