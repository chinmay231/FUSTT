import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding
import numpy as np
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        configs.d_model = configs.seq_len

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear', nn.Linear(configs.seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.enc_in), requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)

        # ⬇ extra stochastic point (helps MC-Dropout variability)
        self.mc_extra_dropout = nn.Dropout(p=configs.dropout)
        self._last_attns = None

    @torch.no_grad()
    def get_last_attn(self):
        """Return aggregated encoder attention list; each item [Q,K] per layer."""
        if self._last_attns is None:
            return {}
        out = []
        for a in self._last_attns:  # a: [B,H,Q,K] or [H,Q,K]
            if a.dim() == 4: a = a.mean(0)   # mean over batch -> [H,Q,K]
            if a.dim() == 3: a = a.mean(0)   # mean over heads -> [Q,K]
            out.append(a)                    # [Q,K]
        return {"encoder_attn": out}

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc = self.revin_layer(x_enc, 'norm')       # [B,L,F]
        enc_out = x_enc.permute(0, 2, 1)              # [B,F,L]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        self._last_attns = attns

        dec_out = self.proj(enc_out)                  # [B,F,P]
        dec_out = dec_out.permute(0, 2, 1)            # [B,P,F]
        linear_out = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        # ⬇ active during MC-Dropout passes
        dec_out = self.mc_extra_dropout(dec_out)

        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec*linear_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out
