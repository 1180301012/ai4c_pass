import torch
import triton
import triton.language as tl


# Optimized Triton kernel for fused cat + batch_norm + prelu
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_bn_prelu_kernel(
    in_5_ptr, in_6_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    prelu_weight_ptr,
    out_ptr,
    N, C, H, W, C_out: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Concatenate in_5 and in_6 along channel dimension (implicit via indexing)
    2. Batch normalization (using running stats)
    3. PReLU activation
    
    Grid: (N, ceil(C_out * H * W / BLOCK_SIZE))
    """
    # Get program indices
    pid_n = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    # Calculate offsets for this block
    block_start = pid_block * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, C_out * H * W)
    
    # offsets in the flat output tensor
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (C_out * H * W)
    
    # Compute (c, h, w) from flat index
    # c = offsets // (H * W)
    # rest = offsets % (H * W)
    # h = rest // W
    # w = rest % W
    
    # Load values - determine source tensor and compute actual offset
    for i in range(BLOCK_SIZE):
        offset = block_start + i
        if offset >= C_out * H * W:
            break
            
        c = offset // (H * W)
        rest = offset % (H * W)
        h = rest // W
        w = rest % W
        
        # Determine source
        if c < C:
            src_offset = pid_n * C * H * W + c * H * W + h * W + w
            val = tl.load(in_5_ptr + src_offset)
        else:
            local_c = c - C
            src_offset = pid_n * C * H * W + local_c * H * W + h * W + w
            val = tl.load(in_6_ptr + src_offset)
        
        # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        mean = tl.load(running_mean_ptr + c)
        var = tl.load(running_var_ptr + c)
        weight = tl.load(bn_weight_ptr + c)
        bias = tl.load(bn_bias_ptr + c)
        
        normalized = (val - mean) / tl.sqrt(var + eps)
        normalized = normalized * weight + bias
        
        # PReLU
        prelu_w = tl.load(prelu_weight_ptr + c)
        activated = tl.where(normalized > 0, normalized, normalized * prelu_w)
        
        # Store to output [N, C_out, H, W]
        out_offset = pid_n * C_out * H * W + c * H * W + h * W + w
        tl.store(out_ptr + out_offset, activated)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the pattern: cat -> batch_norm -> prelu -> adaptive_avg_pool2d -> view
    Returns tmp_7 (prelu output) and tmp_9 (view output)
    """
    # cat along channel dimension
    tmp_5 = torch.cat([in_5, in_6], 1)
    
    # batch_norm with running_mean, running_var, weight, bias
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    
    # prelu
    tmp_7 = torch.prelu(tmp_6, in_0)
    
    # adaptive avg pool to (1, 1)
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
    
    # view to [128, 128]
    tmp_9 = tmp_8.view(128, 128)
    
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Inputs:
    - in_0: prelu weight [C]
    - in_1: running_mean [C]
    - in_2: running_var [C]  
    - in_3: bn bias [C]
    - in_4: bn weight [C]
    - in_5: input tensor 1 [N, C, H, W]
    - in_6: input tensor 2 [N, C, H, W]
    
    Outputs:
    - tmp_7: prelu output [N, 2*C, H, W]
    - tmp_9: viewed output [N, 2*C] (or [1, 2*C] depending on N)
    """
    N = in_5.shape[0]
    C = in_5.shape[1]
    H = in_5.shape[2]
    W = in_5.shape[3]
    
    C_out = C * 2  # After concatenation
    eps = 0.001  # batch_norm eps
    
    # Allocate output tensor for fused bn + prelu
    tmp_7 = torch.empty((N, C_out, H, W), device=in_5.device, dtype=in_5.dtype)
    
    # Calculate grid
    # Total elements = N * C_out * H * W
    total_elements = N * C_out * H * W
    BLOCK_SIZE = 1024  # Will be tuned by autotune
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (N, num_blocks)
    
    # Launch Triton kernel
    fused_bn_prelu_kernel[grid](
        in_5, in_6,
        in_1, in_2, in_4, in_3,
        in_0,
        tmp_7,
        N, C, H, W, C_out,
        eps,
        BLOCK_SIZE,
    )
    
    # Compute adaptive avg pool using torch
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
    
    # View to [N, C_out]
    tmp_9 = tmp_8.view(N, C_out)
    
    return tmp_7, tmp_9


def replacement_func():
    return kernel_wrapper