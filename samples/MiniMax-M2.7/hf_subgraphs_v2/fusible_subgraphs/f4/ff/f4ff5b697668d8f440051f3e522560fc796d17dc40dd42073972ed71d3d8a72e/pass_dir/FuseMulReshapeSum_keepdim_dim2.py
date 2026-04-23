import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_reshape_sum_kernel(
    x_ptr,           # Shape [N, 17, 64, 64] - the 4D tensor after reshape
    w_ptr,           # Shape [1, 1, 1, 64] or [1, 1, 64, 1] - weight with broadcast
    out_ptr,         # Output shape [N, 17, 1, 1]
    N: tl.constexpr,  # Batch size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of output [n, j, 0, 0]
    pid_n = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Base offsets for the current (n, j) slice
    # x has shape [N, 17, 64, 64], we want to sum over k and l dimensions
    # For weight shape [1, 1, 1, 64] (broadcast on dim 0,1,2):
    #   result[n,j] = sum over k,l of x[n,j,k,l] * w[0,0,0,l]
    # For weight shape [1, 1, 64, 1] (broadcast on dim 0,1,3):
    #   result[n,j] = sum over k,l of x[n,j,k,l] * w[0,0,k,0]
    
    # We'll compute sum over both k and l dimensions
    # The 64*64 = 4096 elements are reduced to 1
    
    # Initialize accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over k and l
    for k in range(64):
        for l in range(64):
            # Offsets for x[n, j, k, l]
            x_offset = (pid_n * 17 * 64 * 64 + pid_j * 64 * 64 + k * 64 + l).to(tl.int64)
            
            # Weight offset - determine which dimension has the 64
            # We need to check the stride pattern, but here we assume:
            # If weight shape has 64 at the last dim, use w[0,0,0,l]
            # If weight shape has 64 at the third-to-last dim, use w[0,0,k,0]
            w_offset_last = l  # for [1,1,1,64]
            w_offset_third = k  # for [1,1,64,1]
            
            # Load x value
            x_val = tl.load(x_ptr + x_offset).to(tl.float32)
            
            # Load both weight possibilities (they're small, won't hurt)
            w_last = tl.load(w_ptr + w_offset_last).to(tl.float32)
            w_third = tl.load(w_ptr + w_offset_third).to(tl.float32)
            
            # Use whichever weight is non-zero/non-trivial
            # Actually, we need to detect which pattern based on w_ptr device
            # For now, sum both contributions
            # This works because one of them will be 0 (broadcasting padding)
            
            acc = acc + x_val * (w_last + w_third)
    
    # Store result
    out_offset = pid_n * 17 + pid_j
    tl.store(out_ptr + out_offset, acc)


@triton.jit
def fused_mul_reshape_sum_kernel_v2(
    x_ptr,           # Shape [N, 17, 64, 64] - the 4D tensor after reshape
    w_ptr,           # Weight pointer - will detect pattern from metadata
    out_ptr,         # Output shape [N, 17, 1, 1]
    weight_stride,   # Stride of the 64 dimension in weight
    weight_dim,      # 3 for [*,*,*,64], 2 for [*,*,64,*]
    N: tl.constexpr,  # Batch size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of output [n, j, 0, 0]
    pid_n = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Initialize accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over k and l
    for k in range(64):
        for l in range(64):
            # Offsets for x[n, j, k, l]
            x_offset = (pid_n * 17 * 64 * 64 + pid_j * 64 * 64 + k * 64 + l).to(tl.int64)
            
            # Load x value
            x_val = tl.load(x_ptr + x_offset).to(tl.float32)
            
            # Weight offset depends on weight pattern
            if weight_dim == 3:  # [1,1,1,64] -> broadcast over k
                w_offset = l
            else:  # weight_dim == 2: [1,1,64,1] -> broadcast over l
                w_offset = k * 64  # stride is 64 for the 64 dimension
            
            w_val = tl.load(w_ptr + w_offset).to(tl.float32)
            
            acc = acc + x_val * w_val
    
    # Store result
    out_offset = pid_n * 17 + pid_j
    tl.store(out_ptr + out_offset, acc)


# Better kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_mul_reshape_sum_kernel_v3(
    x_ptr,
    w_ptr,
    out_ptr,
    weight_stride,
    weight_dim,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of the flattened output
    pid = tl.program_id(0)
    
    # Calculate output indices this program handles
    output_size_per_program = (N * 17 + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * output_size_per_program
    end_idx = min(start_idx + output_size_per_program, N * 17)
    
    # Process each output element
    for idx in range(start_idx, end_idx):
        n = idx // 17
        j = idx % 17
        
        # Initialize accumulator
        acc = tl.zeros([1], dtype=tl.float32)
        
        # Iterate over k and l
        for k in range(64):
            for l in range(64):
                x_offset = (n * 17 * 64 * 64 + j * 64 * 64 + k * 64 + l).to(tl.int64)
                x_val = tl.load(x_ptr + x_offset).to(tl.float32)
                
                if weight_dim == 3:
                    w_offset = l
                else:
                    w_offset = k * 64
                
                w_val = tl.load(w_ptr + w_offset).to(tl.float32)
                acc = acc + x_val * w_val
        
        tl.store(out_ptr + idx, acc)


@torch.fx.wrap
def triton_fused_mul_reshape_sum(x, weight, output):
    """
    Fused kernel: multiply (with broadcast) + reshape + sum(dim=2, keepdim=True)
    
    Input:
        x: tensor of shape [N, 17, 64, 64]
        weight: tensor of shape [1, 1, 1, 64] or [1, 1, 64, 1]
        output: pre-allocated tensor of shape [N, 17, 1, 1]
    """
    N = x.shape[0]
    
    # Detect weight pattern
    weight_shape = weight.shape
    if weight_shape[-1] == 64:
        weight_dim = 3
        weight_stride = 1
    else:
        weight_dim = 2
        weight_stride = 64  # The 64 dimension has stride 64
    
    # Flatten output and compute total elements
    out_flat = output.view(-1)
    total_elements = N * 17
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs = max(num_programs, 1)
    
    fused_mul_reshape_sum_kernel_v3[(num_programs,)](
        x_ptr=x,
        w_ptr=weight,
        out_ptr=out_flat,
        weight_stride=weight_stride,
        weight_dim=weight_dim,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(tmp_3, in_0):
    """
    Match the pattern: mul(weight) -> reshape -> sum(dim=2, keepdim=True)
    
    The pattern needs to match:
        tmp_4 = tmp_3.mul(in_0)
        tmp_5 = tmp_4.reshape(N, 17, -1)
        tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    
    But pattern() returns a single value, so we return tmp_6 directly.
    The replacement will produce tmp_6 with the correct shape.
    """
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    return tmp_6


def replacement_args(tmp_3, in_0):
    """
    Extract arguments needed for the replacement function.
    We need:
        - The 4D tensor (tmp_3)
        - The weight tensor (in_0)
        - Information to detect weight pattern and batch size
    """
    return (tmp_3, in_0)


def replacement_func():
    """
    Return the fused implementation function.
    """
    def fused_impl(x, weight):
        # Determine output shape
        N = x.shape[0]
        out_shape = (N, 17, 1, 1)
        
        # Allocate output
        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        
        return triton_fused_mul_reshape_sum(x, weight, output)
    
    return fused_impl