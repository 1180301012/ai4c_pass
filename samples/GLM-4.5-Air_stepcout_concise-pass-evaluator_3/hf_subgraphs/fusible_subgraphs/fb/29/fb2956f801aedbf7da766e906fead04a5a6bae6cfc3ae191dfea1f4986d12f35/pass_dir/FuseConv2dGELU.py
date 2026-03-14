import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """
    Pattern: Conv2D -> GELU fusion
    This matches the sequence of depthwise convolution followed by GELU activation
    """
    conv_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (1, 1), (1, 1), -1)
    gelu_result = torch.nn.functional.gelu(conv_result)
    return gelu_result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """
    Extract arguments for fused Conv2D + GELU kernel
    """
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def fused_conv2d_gelu_kernel(
    x_ptr,  # input tensor
    w_ptr,  # weight tensor [C, 1, 3, 3]
    b_ptr,  # bias tensor [C]
    out_ptr, # output tensor
    batch_size, C, H, W,
    BLOCK_SIZE_M: tl.constexpr,  # block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # block size for spatial dimensions
):
    """
    Fused depthwise convolution + GELU activation kernel
    Optimized for depthwise convolutions with groups=C
    """
    pid_m = tl.program_id(0)  # program id for output channels
    pid_n = tl.program_id(1)  # program id for spatial dimensions
    
    # Offset for output channels
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offset < C
    
    # Offsets for spatial dimensions
    h_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    h_mask = h_offset < H
    
    # Process for each batch
    for batch in range(batch_size):
        # Calculate base pointer for current batch
        x_base_ptr = x_ptr + batch * C * H * W
        out_base_ptr = out_ptr + batch * C * H * W
        
        # Output loop over channel blocks
        for m_block in range(0, C, BLOCK_SIZE_M):
            m_block_offset = m_block + tl.arange(0, BLOCK_SIZE_M)
            m_block_mask = m_block_offset < C
            
            # Load bias for this channel block
            bias = tl.load(b_ptr + m_block_offset, mask=m_block_mask, other=0.0)
            
            # Process spatial blocks
            for h_block in range(0, H, BLOCK_SIZE_N):
                h_block_offset = h_block + tl.arange(0, BLOCK_SIZE_N)
                h_block_mask = h_block_offset < H
                
                # Initialize output accumulation
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                
                # Depthwise convolution with 3x3 kernel
                for kh in range(-1, 2):  # kernel height
                    for kw in range(-1, 2):  # kernel width
                        # Calculate input coordinates with padding
                        ih_base = h_block + 1 + kh  # padding of 1
                        iw_base = 1 + kw  # padding of 1
                        
                        # Load input patch
                        if 0 <= ih_base < H-1 and 0 <= iw_base < H-1:  # valid region
                            for cm_idx in range(BLOCK_SIZE_M):
                                cm = m_block_offset[cm_idx]
                                ih_start = ih_base * H
                                for cn_idx in range(BLOCK_SIZE_N):
                                    ih = ih_start + h_block_offset[cn_idx]
                                    iw = iw_base * H
                                    x_offset = cm * H * W + ih
                                    
                                    # Load weight (depthwise: [C, 1, 3, 3])
                                    w_idx = cm * 9 + (kh + 1) * 3 + (kw + 1)
                                    w_val = tl.load(w_ptr + w_idx, mask=(cm < C), other=0.0)
                                    
                                    # Load input value
                                    x_val = tl.load(x_base_ptr + x_offset, mask=(ih < H*W), other=0.0)
                                    
                                    # Accumulate
                                    acc[cm_idx, cn_idx] += x_val * w_val
                
                # Add bias and apply GELU
                for cm_idx in range(BLOCK_SIZE_M):
                    for cn_idx in range(BLOCK_SIZE_N):
                        cm = m_block_offset[cm_idx]
                        h_pos = h_block_offset[cn_idx]
                        
                        if m_block_mask[cm_idx] and h_block_mask[cn_idx]:
                            val = acc[cm_idx, cn_idx] + bias[cm_idx]
                            # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                            x_gelu = val
                            x_cubed = x_gelu * x_gelu * x_gelu
                            tanh_arg = 0.7978845608028654 * (x_gelu + 0.044715 * x_cubed)
                            gelu_val = 0.5 * x_gelu * (1.0 + tl.tanh(tanh_arg))
                            
                            # Store result
                            out_offset = cm * H * W + h_pos
                            tl.store(out_base_ptr + out_offset, gelu_val)

@torch.fx.wrap
def fused_conv2d_gelu(input_tensor, weight_tensor, bias_tensor):
    """
    Fused depthwise convolution + GELU activation function
    """
    B, C, H, W = input_tensor.shape
    
    # Optimized block sizes for GPU
    BLOCK_SIZE_M = 64   # Channels per block
    BLOCK_SIZE_N = 16   # Spatial dimensions per block
    
    # Calculate grid dimensions
    grid_m = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch fused kernel
    fused_conv2d_gelu_kernel[grid](
        x_ptr=input_tensor,
        w_ptr=weight_tensor.view(-1),  # flatten to [C*9]
        b_ptr=bias_tensor,
        out_ptr=output,
        batch_size=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    """
    Return the fused Conv2D + GELU function
    """
    return fused_conv2d_gelu