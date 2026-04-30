import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
    ],
    key=['n_elements_per_batch'],
)
@triton.jit
def rms_norm_kernel_4d(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    n_elements_per_batch: tl.constexpr,
    batch_size: tl.constexpr,
    scale: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel operating on 4D tensor with shape [B, C, H, W]
    Performs: output = (relu(x).flatten(2) / max(rms * scale, eps)) * weight
    
    This kernel:
    1. Loads 4D tensor elements
    2. Applies ReLU activation
    3. Computes RMS normalization across flattened spatial dims
    4. Applies scale and clamp
    5. Multiplies by weight
    """
    # One program per batch element
    batch_idx = tl.program_id(0)
    
    # Compute sum of squares for this batch
    sum_sq = 0.0
    base_offset = batch_idx * n_elements_per_batch
    
    for i in range(BLOCK_SIZE):
        offset = base_offset + i
        if offset < (batch_idx + 1) * n_elements_per_batch and offset < n_elements:
            x_val = tl.load(x_ptr + offset)
            # Apply ReLU
            x_relu = tl.maximum(x_val, 0.0)
            sum_sq += x_relu * x_relu
    
    # Compute RMS
    rms = tl.sqrt(sum_sq / n_elements_per_batch + eps)
    
    # Scale and clamp
    scaled_rms = rms * scale
    clamped_rms = tl.max(scaled_rms, tl.constexpr(eps))
    
    # Load weight
    weight_val = tl.load(weight_ptr)
    
    # Compute and store output
    for i in range(BLOCK_SIZE):
        offset = base_offset + i
        if offset < (batch_idx + 1) * n_elements_per_batch and offset < n_elements:
            x_val = tl.load(x_ptr + offset)
            # Apply ReLU
            x_relu = tl.maximum(x_val, 0.0)
            # Normalize and apply weight
            normalized = x_relu / clamped_rms
            out_val = normalized * weight_val
            tl.store(output_ptr + offset, out_val)


@torch.fx.wrap
def triton_rms_norm(in_0, in_1):
    """Fused RMSNorm with scale = 0.07216878364870322 - kernel operates on 4D input"""
    B, C, H, W = in_1.shape
    n_elements_per_batch = C * H * W
    n_elements = B * n_elements_per_batch
    
    # Allocate output (will be flattened shape [B, C*H*W])
    output = torch.empty((B, n_elements_per_batch), dtype=in_1.dtype, device=in_1.device)
    
    eps = 1e-05
    scale = 0.07216878364870322
    
    # One program per batch element
    grid = (B,)
    
    rms_norm_kernel_4d[grid](
        in_1, in_0, output,
        n_elements,
        n_elements_per_batch,
        B,
        scale,
        eps,
    )
    
    return output


def replacement_func():
    return triton_rms_norm