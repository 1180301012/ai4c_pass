"""
Shared dispatch module imported by all FuseSplitGetitem* passes.
Both pass files import `dispatch_wrapper` from here, guaranteeing a
single object identity — required by set_g_replacement_func()'s assert.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _squeeze_cont_kernel(
    in_ptr,
    out_ptr,
    stride_s,   # stride of the split tensor along the S dimension (e.g. 2)
    S,          # sequence length
):
    # Single block covering B*S elements (B=1 in practice, S<=32)
    pid = tl.program_id(0) * 32 + tl.arange(0, 32)
    mask = pid < S
    val = tl.load(in_ptr + pid * stride_s, mask=mask, other=0.0)
    tl.store(out_ptr + pid, val, mask=mask)


@torch.fx.wrap
def dispatch_wrapper(split_x, route):
    """
    Shared replacement for both split[i].squeeze(-1).contiguous() branches.
    `route` is "split0" or "split1" (identical logic, separate kernel instances
    kept for CUDA scheduling separation).
    """
    B = split_x.shape[0]
    S = split_x.shape[1]
    N = B * S
    out = torch.empty((B, S), dtype=split_x.dtype, device=split_x.device)
    grid = ((N + 31) // 32,)
    if route == "split0":
        _squeeze_cont_kernel[grid](split_x, out, split_x.stride(1), S)
    elif route == "split1":
        _squeeze_cont1_kernel[grid](split_x, out, split_x.stride(1), S)
    return out


@triton.jit
def _squeeze_cont1_kernel(
    in_ptr,
    out_ptr,
    stride_s,
    S,
):
    pid = tl.program_id(0) * 32 + tl.arange(0, 32)
    mask = pid < S
    val = tl.load(in_ptr + pid * stride_s, mask=mask, other=0.0)
    tl.store(out_ptr + pid, val, mask=mask)