import torch
import triton
import triton.language as tl


@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized AdaptiveAvgPool2d to (1, 1) kernel.
    Each program handles one (batch, channel) position.
    """
    pid = tl.program_id(0)
    
    nc_elements = N * C
    elements_per_program = (nc_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start = pid * elements_per_program
    end = min(start + elements_per_program, nc_elements)
    
    for idx in range(start, end):
        n = idx // C
        c = idx % C
        
        spatial_sum = 0.0
        for h in range(H):
            for w in range(W):
                in_idx = n * C * H * W + c * H * W + h * W + w
                val = tl.load(input_ptr + in_idx)
                spatial_sum = spatial_sum + val
        
        pooled = spatial_sum / (H * W)
        
        out_idx = n * C + c
        tl.store(output_ptr + out_idx, pooled)


@torch.fx.wrap
def adaptive_avg_pool2d_wrapper(input_tensor):
    N, C, H, W = input_tensor.shape
    
    output = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    num_programs = (N * C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    adaptive_avg_pool2d_kernel[(num_programs,)](
        input_tensor, output,
        N, C, H, W,
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_5):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    return tmp_6


def replacement_args(in_5):
    return (in_5,)


def replacement_func():
    return adaptive_avg_pool2d_wrapper