import itertools
import logging
from typing import Generator, List, Tuple

from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from torch_scatter import scatter
from tqdm import tqdm

from datasets.cleaning import replace_stops, replace_whitespaces
from datasets.dataset import Dataset
from evaluation.embedding.vectorizers.vectorizer import Vectorizer
from evaluation.evaluate import result_generator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def slice_text(text, window_type, window_size, window_stride=None) -> List[str]:
    """
    Split a `text` into parts using a sliding window. The windows slides either across characters or sentences, based on the value of `window_tyoe`.
    
    Attributes:
        text: str  Text that is to be splitted into windows.
        window_type: str  Either `sentence` or `character`. The basic unit of the windows.
        window_size: int  How many units are in a window.
        window_stride: int  How many units are skipped each time the window moves.
    """

    text = replace_whitespaces(text)
    
    if window_stride is None:
        window_stride = window_size
    
    if window_size < window_stride:
        logger.warning(f'Window size ({window_size}) is smaller than stride length ({window_stride}). This will result in missing chunks of text.')

        
    if window_type == 'sentence':       
        text = replace_stops(text)
        sentences = sent_tokenize(text)                
        return [
            ' '.join(sentences[i:i+window_size]) 
            for i in range(0, len(sentences), window_stride)
        ]
    
    elif window_type == 'character':
        return [
            text[i:i+window_size] 
            for i in range(0, len(text), window_stride)
        ]

    
def gen_sliding_window_delimiters(post_lengths: List[int], max_size: int) -> Generator[Tuple[int, int], None, None]:
    """
    Calculate where to split the sequence of `post_lenghts` so that the individual batches do not exceed `max_size`
    """
    range_length = start = cur_sum = 0
    
    for post_length in post_lengths:
        if (range_length + post_length) > max_size: # exceeds memory
            yield (start, start + range_length)
            start = cur_sum
            range_length = post_length
        else: # memory still avail in current split
            range_length += post_length
        cur_sum += post_length
        
    if range_length > 0:
        yield (start, start + range_length)


@result_generator
def embedding_results(
    dataset: Dataset,
    vectorizer_fact_check: Vectorizer,
    vectorizer_post: Vectorizer,
    sliding_window: bool = False,
    sliding_window_pooling: str = 'max',
    sliding_window_size: int = None,
    sliding_window_stride: int = None,
    sliding_window_type: str = None,
    post_split_size: int = 256,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
    save_if_missing: bool = False

):
    """
    Generate results using cosine similarity based on embeddings generated via vectorizers.
    
    Attributes:
        dataset: Dataset
        vectorizer_fact_check: Vectorizer  Vectorizer used to process fact-checks
        vectorizer_post: Vectorizer  Vectorizer used to process posts
        sliding_window: bool  Should sliding window be used or should texts be process without slicing.
        sliding_window_pooling: str  One of 'sum', 'mul', 'mean', 'min', 'max' as defined here: https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        sliding_window_size, sliding_window_stride, sliding_window_type:  See `slice_text`
        post_split_size: int  Batch size for post embeddings for sim calculation
        dtype: torch.dtype  Data type in which calculate sim
        device: str  Device on which calculate sim
        save_if_missing: bool  Should the vectors in `dict` be saved after new vectors are calculated? This makes sense for models that will
                be used more than once.
    """
        
    logger.info('Calculating embeddings for fact checks')
    fact_check_embeddings = vectorizer_fact_check.vectorize(
        dataset.id_to_fact_check.values(),
        save_if_missing=save_if_missing,
        normalize=True
    )
    fact_check_embeddings = fact_check_embeddings.transpose(0, 1)  # Rotate for matmul

    fact_check_embeddings = fact_check_embeddings.to(device=device, dtype=dtype)

        
    # We need to split the calculations because of memory limitations, sims matrix alone requires 200k x 25k x 4 = ~20GB RAM 
    # memory = 2**30 # assume 4gb free memory - 2**31 = 2gb for both `sims` and `sorted_ids`
    # post_split_size = memory // len(dataset.id_to_fact_check) // 4  # // 4 because of float32
    post_ids = iter(dataset.id_to_post.keys())
    
    if sliding_window:
        
        logger.info('Splitting posts into windows.')
        windows = [
            slice_text(post, sliding_window_type, sliding_window_size, sliding_window_stride)
            for post in tqdm(dataset.id_to_post.values())
        ]

        logger.info('Calculating embeddings for the windows')
        post_embeddings = vectorizer_post.vectorize(
            list(itertools.chain(*windows)),
            save_if_missing=save_if_missing,
            normalize=True
        ) 
        
        # We need to split the matrix matmul so that all the windows from each post belong to the same batch.
        post_lengths = [len(post) for post in windows]
        segment_array = torch.tensor([
            i 
            for i, num_windows in enumerate(post_lengths) 
            for _ in range(num_windows)
        ])
        delimiters = list(gen_sliding_window_delimiters(post_lengths, post_split_size))
            
        logger.info('Calculating similarity for data splits')
        
        for start_id, end_id in tqdm(delimiters):

            sims = torch.mm(
                post_embeddings[start_id:end_id].to(device=device, dtype=dtype), 
                fact_check_embeddings
            )

            segments = segment_array[start_id:end_id]
            segments -= int(segments[0])

            sims = scatter(
                src=sims,
                index=segments,
                dim=0,
                reduce=sliding_window_pooling,
            )

            sorted_ids = torch.argsort(sims, descending=True, dim=1)

            fact_check_ids = {i: fc_id for i, fc_id in enumerate(dataset.id_to_fact_check.keys())}
            for row in sorted_ids:
                row = row.cpu().numpy()
                row = np.vectorize(fact_check_ids.__getitem__)(row)
                yield row, next(post_ids)

          
    else:
        
        logger.info('Calculating embeddings for posts')
        post_embeddings = vectorizer_post.vectorize(
            dataset.id_to_post.values(),
            save_if_missing=save_if_missing,
            normalize=True
        )
        
        logger.info('Calculating similarity for data splits')
        for start_id in tqdm(range(0, len(dataset.id_to_post), post_split_size)):
            end_id = start_id + post_split_size

            sims = torch.mm(
                post_embeddings[start_id:end_id].to(device=device, dtype=dtype), 
                fact_check_embeddings
            )

            # TODO: argsort does not duplicities into account, the results might not be deterministic
            sorted_ids = torch.argsort(sims, descending=True, dim=1)

            fact_check_ids = {i: fc_id for i, fc_id in enumerate(dataset.id_to_fact_check.keys())}
            for row in sorted_ids:
                row = row.cpu().numpy()
                row = np.vectorize(fact_check_ids.__getitem__)(row)
                yield row, next(post_ids)