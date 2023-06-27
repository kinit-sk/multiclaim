import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5Config


class Model(nn.Module):
    def __init__(self, model_name, pooling='default'):
        super(Model, self).__init__()
        self.pooling = pooling
        
        config = AutoConfig.from_pretrained(model_name)
        
        if isinstance(config, T5Config):         
            from transformers import T5EncoderModel
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            self.model = T5EncoderModel.from_pretrained(model_name, config=config)
        else:
            self.model = AutoModel.from_pretrained(model_name)
            

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == 'default':
            return model_output[1]
        elif self.pooling == 'mean':       
            return self.mean_pooling(model_output[0], attention_mask)

        
    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
def get_model_tokenizer(model_name: str, **kwargs):
    return Model(model_name, **kwargs), AutoTokenizer.from_pretrained(model_name)