import torch
import triton
import triton.language as tl

NEG_INF = -3.4028234663852886e+38

@triton.jit
def attention_mask_fused_kernel(
    in_0_ptr,
    out_ptr,
    N: tl.constexpr,
    NEG_INF_VAL: tl.constexpr,
):
    # Process all N*N elements in one program
    offsets = tl.arange(0, N * N)
    i = offsets // N  # row index
    j = offsets % N   # column index
    
    # Load attention mask value for position j
    # in_0 has shape [1, N] with int64 dtype
    mask_val = tl.load(in_0_ptr + j)
    
    # Causal condition: j <= i means we can attend to position j from position i
    causal_allowed = j <= i
    
    # Attention mask: mask_val == 1 means allowed (not padding)
    attn_allowed = mask_val == 1
    
    # Combined: allowed if causal AND attention mask allows
    allowed = causal_allowed & attn_allowed
    
    # Output: 0 if allowed, -inf if not
    result = tl.where(allowed, 0.0, NEG_INF_VAL)
    
    tl.store(out_ptr + offsets, result)

@torch.fx.wrap
def attention_mask_fused(in_0, route):
    N = in_0.shape[1]
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    
    grid = (1,)
    attention_mask_fused_kernel[grid](
        in_0_ptr=in_0,
        out_ptr=out,
        N=N,
        NEG_INF_VAL=NEG_INF,
    )
    
    return (out,)