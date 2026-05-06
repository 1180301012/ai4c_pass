import torch
from pass_dir.shared_kernels import do_position_bias_triton


# ---------------------------------------------------------------------------
# Pattern: 14×14 relative-position-bias subgraph (for normalized_shape=432)
#   Matches bfloat16/9/convit_small, float32/9/convit_small,
#   float16/9/convit_small where LN is absent or the pos-bias-only graph
# ---------------------------------------------------------------------------
def pattern(x):
    zeros = torch.zeros(1, 196, 196, 3)
    r_a   = torch.arange(14)
    r_v1  = r_a.view(1, -1)
    r_v2  = r_a.view(-1, 1)
    diff  = r_v1 - r_v2
    r_rep = diff.repeat(14, 14)
    r_ri  = diff.repeat_interleave(14, dim=0)
    r_ri2 = r_ri.repeat_interleave(14, dim=1)
    sq1   = r_ri2 ** 2
    sq2   = r_rep ** 2
    sum_sq = sq1 + sq2
    usq   = sum_sq.unsqueeze(0)
    zeros[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = usq
    zeros[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = usq
    zeros[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = r_ri.unsqueeze(0)
    return zeros


def replacement_args(x):
    return (x,)


def replacement_func():
    return do_position_bias_triton