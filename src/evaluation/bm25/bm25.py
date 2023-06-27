from argparse import Namespace
import logging
import os
import shutil
import string

import numpy as np
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from unidecode import unidecode

from datasets.dataset import Dataset
from evaluation.evaluate import result_generator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@result_generator
def bm25_results(dataset: Dataset, use_unidecode: bool = True):
    """
    Attributes:
        dataset: Dataset
        use_unidecode: bool  Should all texts be processed via `unidecode`. This will transform all characters into ASCII. 
    """
    
    # Initialize PyTerrier server
    if not pt.started():
        pt.init()

    # Set up path of the index and clear previous index
    pt_index_path = os.path.join('/', 'labs', 'cache', 'pyterrier_index')
    if os.path.isdir(pt_index_path):
        shutil.rmtree(pt_index_path)
    
    # Prepare fact checks for PyTerrier indexing
    docs = pd.DataFrame.from_dict(dataset.id_to_fact_check.items())
    docs.columns = ['docno', 'text']
    docs['docno'] = docs['docno'].astype(str)
    if use_unidecode:
        docs['text'] = docs['text'].apply(unidecode)
    
    logger.info('Creating PyTerrier index.')
    df_indexer = pt.DFIndexer(pt_index_path, verbose=True)
    index_ref = df_indexer.index(docs['text'], docs['docno'])
    index = pt.IndexFactory.of(os.path.join(pt_index_path, 'data.properties'))
    model = pt.BatchRetrieve(index, wmodel='BM25')
    logger.info('Index created.')


    for post_id, post_text in tqdm(dataset.id_to_post.items()):
        
        # Transform non-ascii characters into ascii
        if use_unidecode:
            post_text = unidecode(post_text)
            
        # Remove punctuation because of Terrier parser
        post_text = ''.join(ch for ch in post_text if ch not in string.punctuation).strip()
        
        # Hand empty text cases
        if not post_text:
            post_text = 'unk'
            
        query = pd.DataFrame({'qid': [post_id], 'query': [post_text]})
        result = model.transform(query)
        yield np.array(result['docno'].astype(int)), post_id
