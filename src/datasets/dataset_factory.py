from datasets.checkthat2021.checkthat2021_dataset import Checkthat2021Dataset
from datasets.crowdchecked.crowdchecked_dataset import CrowdcheckedDataset
from datasets.dataset import Dataset
from datasets.our.our_dataset import OurDataset


def dataset_factory(name, **kwargs) -> Dataset:
    dataset = {
        'our': OurDataset,
        'crowdchecked': CrowdcheckedDataset,
        'checkthat2021': Checkthat2021Dataset,
    }[name](**kwargs)
    return dataset