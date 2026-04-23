import torch
import triton
import triton.language as tl
import math

# ============================================================
# Pattern: getitem[0] + two identity dropouts (p=0, training=False)
# Matches: mha_result[0] -> dropout(0.0) -> dropout(0.0)
# The dropouts are no-ops, so we can eliminate them.
# ============================================================
def pattern(mha_result):
    tmp_5 = mha_result[0]
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7

def replacement_args(mha_result):
    return (mha_result,)

@torch.fx.wrap
def getitem0_no_dropout(mha_result):
    """Get first element and skip two identity dropouts."""
    return mha_result[0]

def replacement_func():
    return getitem0_no_dropout