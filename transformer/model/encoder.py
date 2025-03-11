from torch import nn
from model.encoder_layer import EncoderLayer
from model.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, ffn_hidden, n_heads, n_layers, dropout, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, max_len=maxlen, vocab_size=vocab_size, dropout=dropout, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, ffn_hidden=ffn_hidden, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src, src_mask):
        x = self.emb(src)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
