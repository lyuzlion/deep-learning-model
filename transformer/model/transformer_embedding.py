from torch import nn
from model.positional_encoding import PositionalEncoding
from model.token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(d_model=d_model, vocab_size=vocab_size)
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=max_len, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.tok_emb(x) + self.pos_emb(x))
