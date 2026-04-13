from torch import nn

class BertRegressor(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        return self.linear(pooled_output).squeeze(-1)