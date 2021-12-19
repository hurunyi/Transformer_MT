from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from model.transformer.transformer_encoder import TransformerEncoderBase
from model.transformer.transformer_decoder import TransformerDecoderBase

from torch import Tensor
import torch.nn.functional as F


class BaseModel(nn.Module):
	def __init__(self):
		super().__init__()
		self._is_generation_fast = False

	def get_targets(self, sample, net_output):
		"""Get targets from either the sample or the net's output."""
		return sample["target"]

	def get_normalized_probs(
			self,
			net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
			log_probs: bool,
			sample: Optional[Dict[str, Tensor]] = None,
	):
		"""Get normalized probabilities (or log probs) from a net's output."""
		if hasattr(self, "decoder"):
			return self.decoder.get_normalized_probs(net_output, log_probs, sample)
		elif torch.is_tensor(net_output):
			# syntactic sugar for simple models which don't have a decoder
			# (e.g., the classification tutorial)
			logits = net_output.float()
			if log_probs:
				return F.log_softmax(logits, dim=-1)
			else:
				return F.softmax(logits, dim=-1)
		raise NotImplementedError


class EncoderDecoderModel(BaseModel):
	"""Base class for encoder-decoder models.

	:
		encoder (Encoder): the encoder
		decoder (Decoder): the decoder
	"""

	def __init__(self, encoder, decoder):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
		"""
		Run the forward pass for an encoder-decoder model.

		First feed a batch of source tokens through the encoder. Then, feed the
		encoder output and previous decoder outputs (i.e., teacher forcing) to
		the decoder to produce the next outputs::

			encoder_out = self.encoder(src_tokens, src_lengths)
			return self.decoder(prev_output_tokens, encoder_out)

		:
			src_tokens (LongTensor): tokens in the source language of shape
				`(batch, src_len)`
			src_lengths (LongTensor): source sentence lengths of shape `(batch)`
			prev_output_tokens (LongTensor): previous decoder outputs of shape
				`(batch, tgt_len)`, for teacher forcing

		Returns:
			tuple:
				- the decoder's output of shape `(batch, tgt_len, vocab)`
				- a dictionary with any model-specific outputs
		"""
		encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
		decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
		return decoder_out

	def forward_encoder(self, net_input: Dict[str, Tensor]):
		encoder_input = {
			k: v for k, v in net_input.items() if k != "prev_output_tokens"
		}
		return self.encoder(**encoder_input)

	def forward_decoder(self, prev_output_tokens, **kwargs):
		return self.decoder(prev_output_tokens, **kwargs)

	def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
		"""
		Similar to *forward* but only return features.

		Returns:
			tuple:
				- the decoder's features of shape `(batch, tgt_len, embed_dim)`
				- a dictionary with any model-specific outputs
		"""
		encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
		features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
		return features

	def output_layer(self, features, **kwargs):
		"""Project features to the default output size (typically vocabulary size)."""
		return self.decoder.output_layer(features, **kwargs)

	def max_positions(self):
		"""Maximum length supported by the model."""
		return (self.encoder.max_positions(), self.decoder.max_positions())

	def max_decoder_positions(self):
		"""Maximum length supported by the decoder."""
		return self.decoder.max_positions()


class TransformerModelBase(EncoderDecoderModel):
	def __init__(self, encoder, decoder):
		super().__init__(encoder, decoder)

	@classmethod
	def build_model(cls, args, src_dict, tgt_dict):
		"""Build a new model instance."""
		encoder_embed_tokens = cls.build_embedding(src_dict, embed_dim=512)
		decoder_embed_tokens = cls.build_embedding(tgt_dict, embed_dim=512)
		encoder = cls.build_encoder(args.dropout, src_dict, encoder_embed_tokens)
		decoder = cls.build_decoder(args.dropout, tgt_dict, decoder_embed_tokens)
		return cls(encoder, decoder)

	@classmethod
	def build_embedding(cls, dictionary, embed_dim):
		num_embeddings = len(dictionary)
		padding_idx = dictionary.pad()
		emb = Embedding(num_embeddings, embed_dim, padding_idx)
		return emb

	@classmethod
	def build_encoder(cls, dropout, src_dict, embed_tokens):
		return TransformerEncoderBase(dropout, src_dict, embed_tokens)

	@classmethod
	def build_decoder(cls, dropout, tgt_dict, embed_tokens):
		return TransformerDecoderBase(dropout, tgt_dict, embed_tokens, no_encoder_attn=False)

	def forward(
			self,
			src_tokens,
			prev_output_tokens,
			return_all_hiddens: bool = True,
			features_only: bool = False
	):
		"""
		Run the forward pass for an encoder-decoder model.
		"""
		encoder_out = self.encoder(src_tokens, return_all_hiddens=return_all_hiddens)
		decoder_out = self.decoder(
			prev_output_tokens,
			encoder_out=encoder_out,
			features_only=features_only
		)
		return decoder_out


def Embedding(num_embeddings, embedding_dim, padding_idx):
	m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
	nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
	nn.init.constant_(m.weight[padding_idx], 0)
	return m
