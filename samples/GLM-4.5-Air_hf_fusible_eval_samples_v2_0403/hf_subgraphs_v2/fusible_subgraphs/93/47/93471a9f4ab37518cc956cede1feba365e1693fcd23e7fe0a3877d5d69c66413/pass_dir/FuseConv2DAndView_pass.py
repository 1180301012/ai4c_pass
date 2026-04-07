import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, tmp_5):
    # Match conv2d + view pattern
    conv2d = torch.conv2d(input=conv_input, weight=weight, groups=512)
    tmp_5 = conv2d.view(1, 512, 64, 64)
    return tmp_5

def replacement_args(conv_input, weight, tmp_5):
    return (conv_input, weight, tmp_5)

@triton.jit
def fused_conv2d_view_kernel(
    input_ptr, 
    weight_ptr, 
    output_ptr,
    N,           # batch size = 1
    IC,          # input channels = 512  
    IH,          # input height = 70
    IW,          # input width = 70
    OC,          # output channels = 512
    KH,          # kernel height = 7
    KW,          # kernel width = 7
    OH,          # output height = 64
    OW,          # output width = 64
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute ranges for blocks
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_range = tl.arange(0, BLOCK_SIZE_K)

    # Create masks
    m_mask = m_range < OH
    n_mask = n_range < OW
    k_mask = k_range < KH

    # Load input and weight tiles
    input_ptrs = input_ptr + m_range[:, None] * IH * IW + k_range[None, :] * IW + pid_n * BLOCK_SIZE_N[None, :]
    input_tiles = tl.load(input_ptrs, mask=m_mask[:, None, None] & k_mask[None, None, :] & pid_n * BLOCK_SIZE_N[None, None, :] + k_range[None, None, :] < IW, other=0.0)

    weight_ptrs = weight_ptr + n_range[:, None] * KH * KW + k_range[None, :]
    weight_tiles = tl.load(weight_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

    # Matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, KW, BLOCK_SIZE_K):
        k_offset = k + k_range
        k_mask_offset = k_offset < KW
        input_tiles = tl.load(input_ptrs, mask=m_mask[:, None, None] & k_mask_offset[None, None, :] & pid_n * BLOCK_SIZE_N[None, None, :] + k_offset[None, None, :] < IW, other=0.0)
        weight_tiles = tl.load(weight_ptrs, mask=n_mask[:, None] & k_mask_offset[None, :], other=0.0)
        acc += tl.dot(input_tiles.to(tl.float32), weight_tiles.to(tl.float32))
        input_ptrs += KW * IW * tl.int32(1)
        weight_ptrs += KW * tl.int32(1)

    acc = acc.to(tl.float16 if input_tiles.dtype == tl.float16 else tl.bfloat16 if input_tiles.dtype == tl.bfloat16 else tl.float32)
    
    # Store result with direct reshaping
    output_ptrs = output_ptr + m_range[:, None] * OC * OH * OW + n_range[None, :] * OC * OH + tl.arange(0, OC)[None, None, :]
    acc_output = acc.reshape((BLOCK_SIZE_M, BLOCK_SIZE_N, groups))
    tl.store(output_ptrs + pid_n * BLOCK_SIZE_N[None, None, :] * OC * OH, acc_output, mask=m_mask[:, None, None] & n_mask[None, None, :] & tl.arange(0, OC)[None, None, :] < OC)

@torch.fx.wrap
def fused_conv2d_view(conv_input, weight):
    # Handle different dtypes
    if conv_input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16:
        output_dtype = torch.bfloat16
    elif conv_input.dtype == torch.float16 or weight.dtype == torch.float16:
        output_dtype = torch.float16
    else:
        output_dtype = torch.float32

    # Output shape: [1, 512, 64, 64]
    batch_size, input_channels, input_height, input_width = conv_input.shape
    output_channels = weight.shape[0]
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    
    # Assume stride and padding are such that output is 64x64
    output_height = 64
    output_width = 64
    groups = 512
    
    output = torch.empty((batch_size, output_channels, output_height, output_width), 
                        dtype=output_dtype, device=conv_input.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 8
    
    grid = ( (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
             (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )
    
    fused_conv2d_view_kernel[grid](
        input_ptr=conv_input,
        weight_ptr=weight,
        output_ptr=output,
        N=batch_size,
        IC=input_channels,
        IH=input_height,
        IW=input_width,
        OC=output_channels,
        KH=kernel_height,
        KW=kernel_width,
        OH=output_height,
        OW=output_width,
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return fused_conv2d_view