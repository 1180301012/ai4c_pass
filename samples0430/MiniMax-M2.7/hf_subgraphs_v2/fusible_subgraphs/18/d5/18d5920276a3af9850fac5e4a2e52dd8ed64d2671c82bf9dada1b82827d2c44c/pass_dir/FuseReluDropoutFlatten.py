import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Flatten kernel.
    Flatten(1, -1) on shape [B, C, 1, 1] produces [B, C].
    We compute: relu(x[i]) for all flattened elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)  # ReLU
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_relu_flatten_wrapper(x):
    """
    Wrapper for fused ReLU + Flatten operation.
    Input x: 4D tensor [B, C, H, W] with H=W=1
    Output: 2D tensor [B, C*H*W]
    """
    B, C, H, W = x.shape
    out_flat_dim = C * H * W  # Will be C for H=W=1
    
    # Compute output as [B, C*H*W] using reshape
    BLOCK_SIZE = 1024
    n_elements = B * out_flat_dim
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Reshape to [B*C*H*W] and apply ReLU
    x_flat = x.reshape(-1)
    out_flat = torch.empty_like(x_flat)
    
    if num_programs > 0:
        fused_relu_flatten_kernel[(num_programs,)](
            x_ptr=x_flat,
            out_ptr=out_flat,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Reshape to [B, C*H*W]
    return out_flat.reshape(B, out_flat_dim)


def pattern(x):
    """Match ReLU -> Dropout(p=0) -> Flatten pattern"""
    relu_out = torch.nn.functional.relu(x, inplace=False)
    dropout_out = torch.nn.functional.dropout(relu_out, 0.0, False, False)
    flatten_out = dropout_out.flatten(1, -1)
    return flatten_out


def replacement_args(x):
    """Extract input tensor for replacement function"""
    return (x,)


def replacement_func():
    """Return the fused ReLU + Flatten wrapper"""
    return fused_relu_flatten_wrapper