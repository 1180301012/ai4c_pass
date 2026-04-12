import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_flatten_transpose_contiguous_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    seq_len,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """Fused kernel for GELU + flatten(2) + transpose(1,2) + contiguous
    
    Args:
        input_ptr: Pointer to input tensor [1, C, H, W]
        output_ptr: Pointer to output tensor [1, H*W, C]  
        n_batch: Batch size (always 1)
        n_channels: Number of channels
        seq_len: Sequence length (H*W)
        H: Height dimension
        W: Width dimension
        BLOCK_SIZE: Triton block size
    """
    # Compute global indices
    pid = tl.program_id(0)
    total_elements = n_channels * seq_len
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
        
    # Each program handles a chunk of output elements
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < total_elements
    
    # Convert linear output index to (seq_idx, channel_idx)
    seq_idx = output_idx // n_channels
    channel_idx = output_idx % n_channels
    
    # Convert to input coordinates [1, C, H, W]
    # input_idx = channel_idx * seq_len + seq_idx
    input_batch = 0
    
    # Load input element [1, C, H, W] -> [output_pos]
    input_offset = input_batch * (n_channels * H * W) + channel_idx * (H * W) + seq_idx
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Apply GELU activation with proper dtype handling
    if INPUT_DTYPE == tl.float32:
        gelu_x = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    else:
        # For fp16/bf16, cast to fp32 temporarily for erf calculation
        x_fp32 = tl.cast(x, tl.float32)
        gelu_x_fp32 = x_fp32 * 0.5 * (1.0 + tl.erf(x_fp32 * 0.7071067811865476))
        gelu_x = tl.cast(gelu_x_fp32, INPUT_DTYPE)
    
    # Store result directly in output layout [1, H*W, C]
    output_offset = input_batch * (seq_len * n_channels) + seq_idx * n_channels + channel_idx
    tl.store(output_ptr + output_offset, gelu_x, mask=mask)

@torch.fx.wrap
def get_tl_dtype(torch_dtype):
    """Convert torch dtype to triton dtype"""
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        return tl.float32  # fallback

def fused_gelu_flatten_transpose_contiguous(input_tensor):
    """Fused function: GELU + flatten(2) + transpose(1, 2) + contiguous
    
    Args:
        input_tensor: Input tensor [1, C, H, W]
        
    Returns:
        Output tensor [1, H*W, C]
    """
    # Get input shape
    shape = input_tensor.shape
    n_batch, n_channels, H, W = shape
    seq_len = H * W
    
    # Create output tensor
    output_shape = [n_batch, seq_len, n_channels]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set Triton launch parameters
    BLOCK_SIZE = 1024
    total_elements = n_channels * seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_gelu_flatten_transpose_contiguous_kernel[(num_programs,)](
        input_tensor,
        output,
        n_batch,
        n_channels,
        seq_len,
        H,
        W,
        BLOCK_SIZE,
        get_tl_dtype(input_tensor.dtype)
    )
    
    return output

def pattern(in_2):
    """Match the sequence: GELU -> flatten(2) -> transpose(1,2) -> contiguous"""
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    return tmp_5

def replacement_args(in_2):
    return (in_2,)

def replacement_func():
    return fused_gelu_flatten_transpose_contiguous