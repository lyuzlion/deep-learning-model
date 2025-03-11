from torch import nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, d_model, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1) # padding_idx参数指定了词汇表中用于表示填充位置的特殊标记的索引
