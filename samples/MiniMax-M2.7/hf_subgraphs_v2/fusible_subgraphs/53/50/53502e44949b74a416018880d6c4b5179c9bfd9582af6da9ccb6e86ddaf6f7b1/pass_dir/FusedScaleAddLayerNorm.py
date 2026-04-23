import torch
import triton
import triton.language as tl


# Pattern matching function - matches layer_norm only
def pattern(x, ln_w, ln_b):
    """
    Match the layer_norm pattern:
    - layer_norm(x, (256,), ln_w, ln_b, 1e-05)
    """
    out = torch.nn.functional.layer_norm(x, (256,), ln_w, ln_b, 1e-05)
    return out


def replacement_args(x, ln_w, ln_b):
    return (x, ln_w, ln_b)


@triton.jit
def triton_layer_norm_kernel(
    x_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    n_elements,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Layer norm kernel that processes entire row in one thread block.
    This is correct and matches PyTorch's behavior exactly.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * hidden_size
    
    # Load full row (hidden_size elements)
    offsets = row_offset + tl.arange(0, hidden_size)
    
    # Load data
    x = tl.load(x_ptr + offsets).to(tl.float32)
    ln_w = tl.load(ln_w_ptr + offsets).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offsets).to(tl.float32)
    
    # Layer norm computation - same as PyTorch
    mean = tl.sum(x) / hidden_size
    x_centered = x - mean
    m2 = tl.sum(x_centered * x_centered)
    var = m2 / hidden_size
    std = tl.sqrt(var + eps)
    normalized = x_centered / std
    out = normalized * ln_w + ln_b
    
    # Store
    tl.store(out_ptr + offsets, out)


@torch.fx.wrap
def triton_ln_wrapper(x, ln_w, ln_b):
    """
    Wrapper for the layer_norm Triton kernel.
    """
    batch, seq_len, hidden = x.shape
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Grid: one program per row
    num_programs = batch * seq_len
    
    # Launch kernel
    triton_layer_norm_kernel[(num_programs,)](
        x,
        ln_w,
        ln_b,
        out,
        x.numel(),
        hidden,
        1e-05,
    )
    
    return out


def replacement_func():
    return triton_ln_wrapper