# MultiClaim dataset

## Paper

More details about the dataset and our experiments can be found in our paper _Multilingual Previously Fact-Checked Claim Retrieval_.

- Available at arXiv: https://arxiv.org/abs/2305.07991
- Accepted to the EMNLP 2023 conference 

## Structure

- `cache` - folder for intermediate results, such as models, result dumps, logs, etc.
- `config` - folder for the configuration files reside.
- `drive` - folder for datasets, vectors, etc.
- `src` - folder for code that implement core functionality that is reusable. The code here should just provide an appropriate APIs for experiments.

## Running code in the environment
- Scripts should be run from the root, e.g.:

```bash
python3 src/datasets/...py
```

The `src` folder is added to the `PYTHONENV` environment variable, so that you can reference it anywhere in your code, e.g.:

```python
from datasets.ours.utils import SOCIAL_MEDIA
```

## Docker

### Setup
1. Add your `wandb.config` to the `config` folder, if you plan to use it.

2. Copy the dataset files (three `.csv` files) into `drive/datasets/ours`

3. Build your image. This might take a while the first time:

```bash
docker build . -t multiclaim
```

### Usage

You can run a Jupyter Lab with:

```bash
docker run -p 8888:8888 -v ${PWD}:/labs -it multiclaim
```

If you want to use GPUs use `--gpus all` flag:

```bash
docker run --gpus all -p 8888:8888 -v ${PWD}:/labs -it multiclaim
```

You can use terminal built in Jupyter Lab, but you can also run standalone terminal with:

```bash
docker run -p 8888:8888 -v ${PWD}:/labs --entrypoint bash -it multiclaim
```

You might need to use `--mount` flag for Windows (This example also includes WandB config file, see below):

```bash
docker run --env-file ./config/wandb.conf --gpus all --name multiclaim --rm -p 8888:8888 --mount type=bind,source=${PWD},target=/labs --entrypoint bash -it multiclaim
```

### Updating requirements

All the necessary python package requirements are in the `requirements.txt`, which is generated via `pip freeze` from
within the environment. If you wish to install new or update existing packages, you can do this from the Jupyter Lab 
terminal via `pip install` commands. Then update the list of up-to-date packages via `pip freeze --exclude torch-scatter > requirements.txt`.
`requirements.in` is the original list of packages that was used for bootstrapping the requirements. If you want to add
a new library that is not yet included, add it to the `.in` file as well.


## WandB

### Installation

1. You need to have a WandB account in our your WandB instance.
2. Generate your API key in the app by logging in, going to settings and generating a new API key in the _API keys_ section.
3. Create your `wandb.conf` file by following the example `wandb.conf.example` file available in the `config` directory.

### Usage

1. Use `--env-file ./config/wandb.conf` (in docker run command) to import the WandB related environment variables, e.g.:
2. You can now start using wandb by using `import wandb`.

## Evaluation and training

### Evaluation

This will yield results for `all-MiniLM-L12-v2` sentence encoder.

```python
import os

from datasets.dummy.dummy_dataset import DummyDataset
from evaluation.embedding.vectorizers.sentence_transformer.sentence_transformer_vectorizer import SentenceTransformerVectorizer
from evaluation.evaluate import process_result_generator
from evaluation.embedding.embedding import embedding_results

dt = DummyDataset().load()  # Small debug dataset
cache = os.path.join('/', 'labs', 'cache', 'vectors', 'test')
vct = SentenceTransformerVectorizer(dir_path=cache, model_handle='all-MiniLM-L12-v2')  # create vectorizer that can calculate embeddings
results = embedding_results(dt, vct, vct)  # create result generator, that will start producing predictions for the dataset
process_result_generator(results)  # process the generator and calculate final metrics
```


### Training

An example of a training run. This will evaluate performance for Slovak, Czech and Polish data using the entire training dataset.
Everything is running with the English version of the dataset.

```python
from datasets.our.our_dataset import OurDataset
from training.backend import Config
from training.train import train

eval_languages = ['slk', 'ces', 'pol']

train_dataset = OurDataset(split='train', version='english').load()
dev_datasets = {
    f'dev_{l}': OurDataset(split='dev', version='english', language=l).load()
    for l in eval_languages
}
test_datasets = {
    f'test_{l}': OurDataset(split='test', version='english', language=l).load()
    for l in eval_languages
}

cfg = Config()
cfg.train.num_epochs = 30
cfg.wandb.project = 'experiment_1'
cfg.model.name = 'sentence-transformers/gtr-t5-base'
cfg.model.pooling = 'mean'
train(cfg, train_dataset, dev_datasets, test_datasets)
```
