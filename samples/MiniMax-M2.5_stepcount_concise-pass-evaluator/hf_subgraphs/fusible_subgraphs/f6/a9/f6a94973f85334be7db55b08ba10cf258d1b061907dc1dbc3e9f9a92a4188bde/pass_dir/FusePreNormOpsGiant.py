import torch
import triton
import triton.language as tl


def pattern(tmp_13, tmp_3, tmp_2, tmp_1, tmp_0):
    """
    Fuse pre-norm operations for vit_giant (hidden_dim=1408):
    dropout(p=0) -> layer_norm -> layer_norm
    """
    tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
    tmp_15 = torch.nn.functional.layer_norm(tmp_14, (1408,), tmp_3, tmp_2, 1e-05)
    tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1408,), tmp_1, tmp_0, 1e-05)
    return tmp_15, tmp_16


def replacement_args(tmp_13, tmp_3, tmp_2, tmp_1, tmp_0):
    return (tmp_13, tmp_3, tmp_2, tmp_1, tmp_0)


# Triton kernel for fused layer norms (skipping dropout)
@triton.jit
def fused_layernorm_kernel(
    x_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr,
    out1_ptr, out2_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + block_start + offsets, mask=mask, other=0.0)
    w1 = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    b1 = tl.load(bias1_ptr + offsets, mask=mask, other=0.0)
    w2 = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    b2 = tl.load(bias2_ptr + offsets, mask=mask, other=0.0)
    
    # First layer norm
    mean = tl.sum(x, axis=0) / n_elements
    diff = x - mean
    variance = tl.sum(diff * diff, axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(variance + eps)
    normalized1 = diff * rstd
    out1 = normalized1 * w1 + b1
    
    # Second layer norm (fused)
    mean2 = tl.sum(out1, axis=0) / n_elements
    diff2 = out1 - mean2
    variance2 = tl.sum(diff2 * diff2, axis=0) / n_elements
    rstd2 = 1.0 / tl.sqrt(variance2 + eps)
    normalized2 = diff2 * rstd2
    out2 = normalized2 * w2 + b2
    
    # Store outputs
    tl.store(out1_ptr + block_start + offsets, out1, mask=mask)
    tl.store(out2_ptr + block_start + offsets, out2, mask=mask)


@torch.fx.wrap
def triton_fused_layernorm(x, weight1, bias1, weight2, bias2, eps=1e-05):
    """Triton-based fused layer norms (skipping dropout)"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    
    fused_layernorm_kernel[(num_programs,)](
        x_ptr=x,
        weight1_ptr=weight1,
        bias1_ptr=bias1,
        weight2_ptr=weight2,
        bias2_ptr=bias2,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2


def replacement_func():
    return triton_fused_layernorm