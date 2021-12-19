import torch


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
