from typing import List

import os
import subprocess
import torch

from .embed import embed_sentences
from evaluation.embedding.vectorizers.vectorizer import Vectorizer


class LaserVectorizer(Vectorizer):
    """
    Vectorizer for `LASER` (Language-Agnostic SEntence Representations) model.
    """
    
    def __init__(
        self,
        dir_path: str,
        batch_size: int = 32
    ):
        """
        Attributes:
            dir_path: str  Path to cached vectors and vocab files.
            batch_size: The number of samples
        """
        
        super().__init__(dir_path)
        self.encoder_path = './cache/laser/laser2.pt'
        self.spm_model = './cache/laser/laser2.spm'
        self.check_model()
            
        self.batch_size = batch_size

    def check_model(self) -> None:
        """
        Function to chceck if LASER model is downloaded.
        
        If the model is not downloaded, the model is downloaded along with the remaining necessary files using `download_laser.sh`.
        """
        if not os.path.exists(self.encoder_path) or not os.path.exists(self.spm_model):
            subprocess.run(["bash", "./src/evaluation/embedding/vectorizers/laser/download_laser.sh"])
        
        
    def _calculate_vectors(self, texts: List[str]) -> torch.tensor:
        
        embeddings = embed_sentences(
            sentences=texts,
            encoder_path=self.encoder_path,
            cpu=False,
            batch_size=self.batch_size,
            spm_model=self.spm_model
        )

        return torch.from_numpy(embeddings)