import config
import transformers
import torch.nn as nn

class MultilabelClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MultilabelClassifier, self).__init__()
        
        self.distill_bert = transformers.DistilBertModel.from_pretrained(config.DISTILBERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, n_classes)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.bert_drop(pooled_output)
        output = self.out(output_1)
        return output