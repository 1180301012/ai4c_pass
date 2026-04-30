import torch
import sys
import os

# Add pass_dir to sys.path for importing shared module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared import fused_embedding_layernorm_wrapper


# Pattern for float32/all-mpnet-base-v2: hidden_dim=768, eps=1e-05
# Matches only the embedding + add + layer_norm + dropout path
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "f32_mpnet_768_7")


def replacement_func():
    return fused_embedding_layernorm_wrapper