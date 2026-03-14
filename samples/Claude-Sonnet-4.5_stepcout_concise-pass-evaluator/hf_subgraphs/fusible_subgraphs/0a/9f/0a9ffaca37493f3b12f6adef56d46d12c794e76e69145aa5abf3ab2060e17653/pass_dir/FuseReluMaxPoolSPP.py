import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Match the ReLU + 3x MaxPool + Concat pattern"""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 8, 'BLOCK_SIZE_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 16, 'BLOCK_SIZE_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 64}, num_warps=8),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_relu_maxpool_spp_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Calculate channel block
    c_start = pid_c * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Calculate spatial position
    hw_start = pid_hw * BLOCK_SIZE_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
    hw_total = H * W
    hw_mask = hw_offsets < hw_total
    
    # Decode spatial positions
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W
    
    # Process each channel and spatial position
    for c_idx in range(BLOCK_SIZE_C):
        c = c_start + c_idx
        if c >= C:
            break
            
        for hw_idx in range(BLOCK_SIZE_HW):
            hw = hw_start + hw_idx
            if hw >= hw_total:
                break
                
            h = hw // W
            w = hw % W
            
            # Load input and apply ReLU
            in_offset = pid_b * C * H * W + c * H * W + h * W + w
            val = tl.load(input_ptr + in_offset)
            relu_val = tl.maximum(val, 0.0)
            
            # Write ReLU output (first C channels of output)
            out_offset_relu = pid_b * (4 * C) * H * W + c * H * W + h * W + w
            tl.store(output_ptr + out_offset_relu, relu_val)
            
            # Compute max_pool: kernel=5, stride=1, padding=2, dilation=1
            max_val = -1e30
            
            for dh in range(5):
                for dw in range(5):
                    ih = h + dh - 2  # padding=2
                    iw = w + dw - 2
                    
                    # Check bounds
                    if ih >= 0 and ih < H and iw >= 0 and iw < W:
                        pool_offset = pid_b * C * H * W + c * H * W + ih * W + iw
                        pool_val = tl.load(input_ptr + pool_offset)
                        pool_val = tl.maximum(pool_val, 0.0)
                        max_val = tl.maximum(max_val, pool_val)
            
            # Write max_pool output to three sections (channels C:2C, 2C:3C, 3C:4C)
            for rep in range(3):
                out_c = (rep + 1) * C + c
                out_offset_pool = pid_b * (4 * C) * H * W + out_c * H * W + h * W + w
                tl.store(output_ptr + out_offset_pool, max_val)


@torch.fx.wrap
def fused_relu_maxpool_spp(input_tensor):
    B, C, H, W = input_tensor.shape
    
    # Output has 4x channels (original + 3x maxpool)
    output = torch.empty(B, 4 * C, H, W, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Define grid
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_HW = 64
    grid = (
        B,
        triton.cdiv(C, BLOCK_SIZE_C),
        triton.cdiv(H * W, BLOCK_SIZE_HW)
    )
    
    fused_relu_maxpool_spp_kernel[grid](
        input_tensor,
        output,
        B, C, H, W,
    )
    
    return output


def replacement_func():
    return fused_relu_maxpool_spp