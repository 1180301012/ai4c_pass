import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern to match: in-place add + batch_norm + relu
    in_0: running_mean
    in_1: running_var
    in_2: bias
    in_3: weight
    in_4: tensor to add
    in_5: base tensor (modified in-place)
    """
    # In-place addition
    in_5 = in_5 + in_4
    tmp_4 = in_5
    # Batch normalization
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # ReLU
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_4, tmp_6)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_5, in_4, in_0, in_1, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C': 64}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def fused_add_bn_relu_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    add_out_ptr, relu_out_ptr,
    N, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel for: add + batch_norm + relu
    x, y: [N, C, H, W] input tensors
    mean, var, weight, bias: [C] batch norm parameters
    """
    # Program IDs for 2D grid over (N*HW, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Offsets
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Masks
    n_mask = n_offsets < N * HW
    c_mask = c_offsets < C
    
    # Load batch norm parameters (per channel)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Compute indices for input tensors
    # Shape: [N, C, H, W] -> need to compute correct index
    # Flatten to [N*C*H*W] then index appropriately
    n_batch = n_offsets // HW
    n_spatial = n_offsets % HW
    
    # For each spatial position and batch, process all channels
    for nc_offset in range(BLOCK_SIZE_N):
        if pid_n * BLOCK_SIZE_N + nc_offset >= N * HW:
            break
        
        n_idx = (pid_n * BLOCK_SIZE_N + nc_offset) // HW
        hw_idx = (pid_n * BLOCK_SIZE_N + nc_offset) % HW
        
        # Compute base index: n_idx * C * HW + c * HW + hw_idx
        base_idx = n_idx * C * HW + hw_idx
        
        # Load x and y for all channels at this spatial position
        idx = base_idx + c_offsets * HW
        mask = c_mask & (n_idx < N)
        
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = tl.load(y_ptr + idx, mask=mask, other=0.0)
        
        # Add
        add_result = x + y
        
        # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (add_result - mean) / tl.sqrt(var + eps)
        bn_result = normalized * weight + bias
        
        # ReLU
        relu_result = tl.maximum(bn_result, 0.0)
        
        # Store results
        tl.store(add_out_ptr + idx, add_result, mask=mask)
        tl.store(relu_out_ptr + idx, relu_result, mask=mask)


@torch.fx.wrap
def fused_add_bn_relu(x, y, mean, var, weight, bias):
    """
    Fused add + batch_norm + relu
    x, y: [N, C, H, W] tensors
    mean, var, weight, bias: [C] batch norm parameters
    """
    N, C, H, W = x.shape
    HW = H * W
    eps = 1e-05
    
    # Output tensors
    add_out = torch.empty_like(x)
    relu_out = torch.empty_like(x)
    
    # Grid configuration
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_C = 64
    grid = (
        triton.cdiv(N * HW, BLOCK_SIZE_N),
        triton.cdiv(C, BLOCK_SIZE_C),
    )
    
    fused_add_bn_relu_kernel[grid](
        x, y,
        mean, var, weight, bias,
        add_out, relu_out,
        N, C, HW,
        eps,
    )
    
    return add_out, relu_out


def replacement_func():
    return fused_add_bn_relu