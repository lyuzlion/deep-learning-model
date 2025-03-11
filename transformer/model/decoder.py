import torch
from torch import nn
from model.decoder_layer import DecoderLayer
from model.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, vocab_size, maxlen, d_model, ffn_hidden, n_heads, n_layers, dropout, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, 
                                        max_len=maxlen, 
                                        vocab_size=vocab_size, 
                                        dropout=dropout, 
                                        device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, 
                                                  n_heads=n_heads, 
                                                  ffn_hidden=ffn_hidden, 
                                                  dropout=dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, trg, trg_mask, encoder_output, src_mask):
        '''
        trg: target sequence
        trg_mask: target mask
        encoder_output: output from encoder
        src_mask: source mask
        '''
        x = self.emb(trg)
        for layer in self.layers:
            x = layer(x, trg_mask, encoder_output, src_mask)
        x = self.linear(x)
        return x



