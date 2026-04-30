import torch
import triton
import triton.language as tl
from pass_dir._kernels import _dispatch_embed_add_ln, replacement_func  # noqa: F401 – shared object


def pattern(in_0, in_1, in_2, in_3, in_4, tmp_14):
    """
    Matches: word_embed + pos_embed + add + layer_norm + dropout(inference=noop)
    H = 768.
    """
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_15 = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_1, in_2, in_3, in_4, tmp_14):
    return (in_0, in_1, in_2, in_3, in_4, tmp_14, "768")