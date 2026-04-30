import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["K"],
)
@triton.jit
def fused_linear_ln_kernel(
    # Input pointers
    in_8_ptr, in_7_ptr, in_6_ptr,
    # LayerNorm params
    in_3_ptr, in_2_ptr,
    # Output pointer
    out_ptr,
    # Shapes
    M, K,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for linear + layer_norm computation.
    """
    pid = tl.program_id(0)
    row_offset = pid * K
    
    # Compute linear: in_8[m, 0, :] @ in_7[:, :] + bias
    linear_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Matrix-vector multiplication
    for ki in range(0, K, BLOCK_SIZE):
        mask_ki = ki + tl.arange(0, BLOCK_SIZE) < K
        
        # Load in_8[m, 0, ki:ki+BLOCK_SIZE]
        in_8_vals = tl.load(
            in_8_ptr + row_offset + ki + tl.arange(0, BLOCK_SIZE),
            mask=mask_ki,
            other=0.0
        ).to(tl.float32)
        
        # Load in_7[ki:ki+BLOCK_SIZE, :]
        in_7_col_offsets = (ki + tl.arange(0, BLOCK_SIZE))[:, None] * K + tl.arange(0, BLOCK_SIZE)[None, :]
        in_7_vals = tl.load(
            in_7_ptr + in_7_col_offsets,
            mask=mask_ki[:, None] & mask_ki[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Accumulate dot product
        linear_vals = linear_vals + tl.sum(in_8_vals[:, None] * in_7_vals, axis=0)
    
    # Add bias
    linear_vals = linear_vals + tl.load(in_6_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    
    # LayerNorm on linear output
    mean = tl.sum(linear_vals, axis=0) / K
    var = tl.sum((linear_vals - mean) * (linear_vals - mean), axis=0) / K
    rstd = 1.0 / tl.sqrt(var + eps)
    ln = (linear_vals - mean) * rstd
    ln = ln * tl.load(in_3_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32) + tl.load(in_2_ptr + tl.arange(0, BLOCK_SIZE)).to(tl.float32)
    
    # Store result
    tl.store(out_ptr + row_offset + tl.arange(0, BLOCK_SIZE), ln, mask=tl.arange(0, BLOCK_SIZE) < K)


def pattern(in_2, in_3, in_6, in_7, in_8):
    """
    Pattern for linear + layer_norm fusion.
    """
    linear_out = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear_out, (256,), in_3, in_2, 1e-05)
    return tmp_9


def replacement_args(in_2, in_3, in_6, in_7, in_8):
    return (in_2, in_3, in_6, in_7, in_8)


@torch.fx.wrap
def fused_linear_ln_wrapper(in_2, in_3, in_6, in_7, in_8):
    """
    Wrapper for the fused linear + layer_norm kernel.
    """
    M = in_8.shape[0]  # 300
    K = in_8.shape[2]  # 256
    
    num_programs = M
    output = torch.empty((M, 1, K), dtype=in_8.dtype, device=in_8.device)
    
    fused_linear_ln_kernel[(num_programs,)](
        in_8, in_7, in_6,
        in_3, in_2,
        output,
        M, K,
        1e-05,
    )
    
    return output


def replacement_func():
    return fused_linear_ln_wrapper