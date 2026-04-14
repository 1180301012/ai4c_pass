import torch
import triton
import triton.language as tl
from pass_dir.unfold_kernels import dispatch_unfold_perm_reshape


# ---------------------------------------------------------------------------
# Full chain: unfold(in_1)+permute+reshape + unfold(in_2)+permute+reshape
#             + cat([in2_patches, in1_patches, in_0]) + to(float16)
# If this matches, we avoid ALL intermediate allocations.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    # a=in_0, route="full_chain", b=in_1, c=in_2
    return (in_0, "full_chain", in_1, in_2)


def replacement_func():
    return dispatch_unfold_perm_reshape