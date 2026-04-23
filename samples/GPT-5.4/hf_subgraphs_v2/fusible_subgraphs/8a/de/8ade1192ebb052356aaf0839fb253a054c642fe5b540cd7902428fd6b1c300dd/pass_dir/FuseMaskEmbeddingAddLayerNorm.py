import torch

from pass_dir.shared_fused_mask_embedding_add_layernorm import replacement_func


def pattern(x, gamma, beta):
    return torch.nn.functional.layer_norm(x, (1024,), gamma, beta, 1e-05)


def replacement_args(x, gamma, beta):
    return (x, gamma, beta)