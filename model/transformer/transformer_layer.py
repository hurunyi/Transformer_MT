from typing import Dict, List, Optional

import torch
import torch.nn as nn
from model.utils import MultiheadAttention
from model.utils import Dropout, LayerNorm
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from torch.nn import functional as F


class TransformerEncoderLayerBase(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.embed_dim = 512
        self.quant_noise = 0
        self.quant_noise_block_size = 8
        self.self_attn = self.build_self_attention(self.embed_dim)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = Dropout(dropout)
        self.activation_fn = F.relu
        activation_dropout_p = 0
        self.activation_dropout_module = Dropout(float(activation_dropout_p))
        self.fc1 = self.build_fc1(
            input_dim=self.embed_dim,
            output_dim=2048,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            input_dim=2048,
            output_dim=self.embed_dim,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    @staticmethod
    def build_fc1(input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    @staticmethod
    def build_fc2(input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim):
        return MultiheadAttention(
            embed_dim,
            8,
            dropout=0.0,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    @staticmethod
    def residual_connection(x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x: (seq_len, bsz, embed_dim)
            encoder_padding_mask: (bsz, seq_len)
            attn_mask: None

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self, dropout, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = 512
        self.dropout_module = Dropout(dropout)
        self.quant_noise = 0
        self.quant_noise_block_size = 8

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.activation_fn = F.relu
        activation_dropout_p = 0
        self.activation_dropout_module = Dropout(float(activation_dropout_p))

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            2048,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            2048,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    @staticmethod
    def build_fc1(input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    @staticmethod
    def build_fc2(input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            8,
            dropout=0,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim):
        return MultiheadAttention(
            embed_dim,
            8,
            kdim=512,
            vdim=512,
            dropout=0,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    @staticmethod
    def residual_connection(x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x: (seq_len, bsz, embed_dim)
            encoder_out: (seq_len, bsz, embed_dim)
            encoder_padding_mask: (bsz, seq_len)

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        # self attention
        residual = x
        y = x
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.self_attn_layer_norm(x)

        # encoder attention
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            x = self.encoder_attn_layer_norm(x)

        # feed forward
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.final_layer_norm(x)
        return x, attn, None
