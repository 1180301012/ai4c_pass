import torch
import triton
import triton.language as tl


def pattern(tmp_4, tmp_3, tmp_2, eps):
    layer_norm = torch.nn.functional.layer_norm(tmp_4, (38, 1, 1), tmp_3, tmp_2, eps)
    relu = torch.nn.functional.relu(layer_norm, inplace=True)
    return relu


def replacement_args(tmp_4, tmp_3, tmp_2, eps):
    return (tmp_4, tmp_3, tmp_2, eps, 38)


@triton.jit
def layer_norm_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, eps, out_ptr,
    batch, C,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate thread indices
    batch_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    # Calculate offset for current (batch, channel)
    off = batch_idx * C + c_idx
    x = tl.load(x_ptr + off)

    # Load weight and bias for current channel
    weight = tl.load(weight_ptr + c_idx)
    bias = tl.load(bias_ptr + c_idx)

    # First, accumulate sum and sum of squares for reduction
    sh_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sh_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Reduction pass: accumulate values per channel
    for i in range(0, batch, BLOCK_SIZE):
        idx = i + c_idx
        if idx < batch:
            x_val = tl.load(x_ptr + idx)
            sh_sum[tl.program_id(0)] += x_val
            sh_sq[tl.program_id(0)] += x_val * x_val

    # Barrier to ensure all threads complete reduction
    tl.barrier()
    
    # Reduce shared memory values to get sum and sum of squares
    sum_val = tl.sum(sh_sum)
    sq_val = tl.sum(sh_sq)
    
    # Calculate mean and variance
    mean = sum_val / batch
    var = (sq_val - sum_val * mean) / batch

    # Apply LayerNorm and ReLU in one step
    y = (x - mean) / tl.sqrt(var + eps) * weight + bias
    y = y if y > 0 else 0.0

    tl.store(out_ptr + off, y)


@torch.fx.wrap
def layer_norm_relu_fused(x, weight_norm, bias_norm, eps, C):
    batch = x.shape[0]
    
    # Ensure weight/bias are flattened to [C] shape
    weight_norm = weight_norm.view(C)
    bias_norm = bias_norm.view(C)
    
    # Allocate output
    output = torch.empty((batch, C), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = (batch, C)
    layer_norm_relu_kernel[grid](
        x, weight_norm, bias_norm, eps,
        output, batch, C, BLOCK_SIZE=32
    )

    # Reshape back to original spatial dimensions [batch, C, 1, 1]
    return output.view(batch, C, 1, 1)


def replacement_func():
    return layer_norm_relu_fused