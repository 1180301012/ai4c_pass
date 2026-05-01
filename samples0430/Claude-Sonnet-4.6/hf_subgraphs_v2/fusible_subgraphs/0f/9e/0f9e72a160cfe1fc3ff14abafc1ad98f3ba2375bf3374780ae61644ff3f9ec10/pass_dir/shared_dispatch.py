"""
Shared Triton kernels and dispatch wrapper used by FuseNormDiv and FuseScalarExpMul.
Importing from this module ensures both pass files return the SAME dispatch_wrapper
object from replacement_func(), satisfying output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _shared_l2_norm_kernel(
    x_ptr, out_ptr,
    BLOCK_D: tl.constexpr,
):
    """
    L2-normalize one row of BLOCK_D elements.
    D == BLOCK_D == 512 always for this workload — no masking needed.
    """
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    x = tl.load(x_ptr + row_id * BLOCK_D + offsets)
    x_f32 = x.to(tl.float32)
    norm = tl.sqrt(tl.sum(x_f32 * x_f32, axis=0))
    out = (x_f32 / norm).to(x.dtype)
    tl.store(out_ptr + row_id * BLOCK_D + offsets, out)


@triton.jit
def _shared_exp_mul_kernel(
    s_ptr, x_ptr, out_ptr,
    BLOCK: tl.constexpr,
):
    """
    Compute exp(scalar s) * tensor x element-wise.
    N == BLOCK == 512 always — no masking needed.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    s_val = tl.load(s_ptr).to(tl.float32)
    scale = tl.exp(s_val)
    x = tl.load(x_ptr + offsets)
    out = (x.to(tl.float32) * scale).to(x.dtype)
    tl.store(out_ptr + offsets, out)


@torch.fx.wrap
def dispatch_wrapper(*args):
    """
    Shared replacement function for FuseNormDiv and FuseScalarExpMul passes.
    Routes on the last (string) argument:
      "norm_div" -> L2 normalize x along last dim: out = x / ||x||_2
      "exp_mul"  -> Scale tensor by exp(scalar):   out = exp(s) * x
    """
    route = args[-1]
    if route == "norm_div":
        x = args[0]
        B = x.numel() // 512       # number of rows (D is always 512)
        out = torch.empty_like(x)
        _shared_l2_norm_kernel[(B,)](
            x_ptr=x,
            out_ptr=out,
            BLOCK_D=512,
        )
        return out
    else:  # route == "exp_mul"
        s = args[0]
        x = args[1]
        N = x.numel()
        n_blocks = (N + 511) // 512
        out = torch.empty_like(x)
        _shared_exp_mul_kernel[(n_blocks,)](
            s_ptr=s,
            x_ptr=x,
            out_ptr=out,
            BLOCK=512,
        )
        return out