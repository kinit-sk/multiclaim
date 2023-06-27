from typing import Generator

import numpy as np

from datasets.dataset import Dataset
from evaluation.evaluate import result_generator


@result_generator
def dummy_results(dataset: Dataset) -> Generator:
    """
    Return a sorted list of fact-check ids as a result for all posts.
    
    Used for testing and development purposes.
    """
    
    result = np.array(sorted(dataset.id_to_fact_check))
    for post_id in dataset.id_to_post:
        yield result, post_id 
        