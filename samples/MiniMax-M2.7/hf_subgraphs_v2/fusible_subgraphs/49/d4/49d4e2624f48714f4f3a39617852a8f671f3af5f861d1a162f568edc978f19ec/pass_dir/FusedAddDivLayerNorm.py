import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
    ],
    key=['normalized_shape'],
)
@triton.jit
def fused_add_div_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    normalized_shape: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. z = (x + y) / 2
    2. LayerNorm(z)
    
    Assumes input shape is [1, normalized_shape] where we normalize over the last dim.
    """
    # Program ID corresponds to the batch dimension (assuming batch_size=1 for this pattern)
    row_idx = tl.program_id(0)
    
    # Base offsets for this row
    row_offset = row_idx * normalized_shape
    
    # Compute mean in accumulation register
    sum_vals = tl.zeros((1,), dtype=tl.float32)
    
    # First pass: compute sum of (x + y)
    for i in range(0, normalized_shape, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + normalized_shape
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_vals += tl.sum(x_vals + y_vals, axis=0)
    
    mean = sum_vals / normalized_shape
    
    # Second pass: compute variance
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, normalized_shape, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + normalized_shape
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = (x_vals + y_vals) * 0.5  # divide by 2 = multiply by 0.5
        diff = z - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    
    var = sum_sq / normalized_shape
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Third pass: compute normalized output
    for i in range(0, normalized_shape, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + normalized_shape
        
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        z = (x_vals + y_vals) * 0.5
        diff = z - mean
        norm = diff * rstd
        
        # Load weight and bias for normalized shape
        w_offsets = i + tl.arange(0, BLOCK_SIZE)
        w_mask = w_offsets < normalized_shape
        w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        
        out = norm * w + b
        # Store in original dtype (bfloat16 or float16)
        if output_ptr.dtype == tl.bfloat16:
            out = out.to(tl.bfloat16)
        else:
            out = out.to(tl.float16)
        tl.store(output_ptr + offsets, out, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_add_div_layer_norm_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused kernel.
    
    in_0: bias tensor [768]
    in_1: weight tensor [768]
    in_2: input tensor [1, 768]
    in_3: input tensor [1, 768]
    """
    n_batch, n_features = in_2.shape
    
    # Allocate output tensor
    output = torch.empty_like(in_2)
    
    # Launch kernel - one program per batch element
    grid = (n_batch,)
    
    fused_add_div_layer_norm_kernel[grid](
        x_ptr=in_2,
        y_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        n_elements=n_batch * n_features,
        normalized_shape=n_features,
        eps=1e-12,
    )
    
    return output


def replacement_func():
    return fused_add_div_layer_norm_wrapper