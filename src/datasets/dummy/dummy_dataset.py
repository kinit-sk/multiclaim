from datasets.dataset import Dataset
from datasets.our.our_dataset import OurDataset


class DummyDataset(Dataset):
    """
    Very small dataset based on OurDataset with 100 fact-checks an 100 posts.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        
        dt = OurDataset(split='test').load()
        dt.fact_check_post_mapping = list(dt.fact_check_post_mapping)[:100]
        
        fact_check_ids, post_ids = map(set, zip(*dt.fact_check_post_mapping))
        
        dt.id_to_fact_check = {
            k: v
            for k, v in dt.id_to_fact_check.items()
            if k in fact_check_ids
        }
        
        dt.id_to_post = {
            k: v
            for k, v in dt.id_to_post.items()
            if k in post_ids
        }
        return dt