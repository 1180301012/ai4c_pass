import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_split_kernel(
    uv_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    # Dimensions
    batch_size,
    dim1_size,
    dim2_size,
    # Split sizes
    split_0_size,
    split_1_size,
    split_2_size,
    # Strides for uv tensor (batch, dim1, dim2)
    uv_stride_batch,
    uv_stride_dim1,
    uv_stride_dim2,
    # Output strides
    out0_stride_batch,
    out0_stride_dim1,
    out0_stride_dim2,
    out1_stride_batch,
    out1_stride_dim1,
    out1_stride_dim2,
    out2_stride_batch,
    out2_stride_dim1,
    out2_stride_dim2,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU activation and split operation.
    
    uv shape: [batch, 17, 1152]
    split sizes along dim=2: [512, 512, 128]
    
    Output 0: [batch, 17, 512] (split[0])
    Output 1: [batch, 17, 512] (split[1])
    Output 2: [batch, 17, 128] (split[2])
    """
    # Get program IDs for batch and dim1 dimensions
    batch_idx = tl.program_id(0)
    dim1_idx = tl.program_id(1)
    
    # Calculate offsets for this program
    base_uv_offset = (
        batch_idx * uv_stride_batch + 
        dim1_idx * uv_stride_dim1
    )
    
    # Load and process data in blocks
    # Process split 0 region [0, split_0_size)
    for split0_offset in range(0, split_0_size, BLOCK_SIZE):
        offsets = base_uv_offset + split0_offset + tl.arange(0, BLOCK_SIZE)
        mask = split0_offset + tl.arange(0, BLOCK_SIZE) < split_0_size
        
        # Load uv values
        x = tl.load(uv_ptr + offsets, mask=mask, other=0.0)
        
        # Apply SiLU: x * sigmoid(x)
        sigmoid_x = tl.sigmoid(x)
        out = x * sigmoid_x
        
        # Store to output 0
        out_offsets = (
            batch_idx * out0_stride_batch + 
            dim1_idx * out0_stride_dim1 + 
            split0_offset + tl.arange(0, BLOCK_SIZE)
        )
        tl.store(out0_ptr + out_offsets, out, mask=mask)
    
    # Process split 1 region [split_0_size, split_0_size + split_1_size)
    for split1_offset in range(0, split_1_size, BLOCK_SIZE):
        uv_offsets = (
            base_uv_offset + 
            split_0_size + 
            split1_offset + 
            tl.arange(0, BLOCK_SIZE)
        )
        mask = split1_offset + tl.arange(0, BLOCK_SIZE) < split_1_size
        
        # Load uv values
        x = tl.load(uv_ptr + uv_offsets, mask=mask, other=0.0)
        
        # Apply SiLU: x * sigmoid(x)
        sigmoid_x = tl.sigmoid(x)
        out = x * sigmoid_x
        
        # Store to output 1
        out_offsets = (
            batch_idx * out1_stride_batch + 
            dim1_idx * out1_stride_dim1 + 
            split1_offset + 
            tl.arange(0, BLOCK_SIZE)
        )
        tl.store(out1_ptr + out_offsets, out, mask=mask)
    
    # Process split 2 region [split_0_size + split_1_size, total)
    for split2_offset in range(0, split_2_size, BLOCK_SIZE):
        uv_offsets = (
            base_uv_offset + 
            split_0_size + 
            split_1_size + 
            split2_offset + 
            tl.arange(0, BLOCK_SIZE)
        )
        mask = split2_offset + tl.arange(0, BLOCK_SIZE) < split_2_size
        
        # Load uv values
        x = tl.load(uv_ptr + uv_offsets, mask=mask, other=0.0)
        
        # Apply SiLU: x * sigmoid(x)
        sigmoid_x = tl.sigmoid(x)
        out = x * sigmoid_x
        
        # Store to output 2
        out_offsets = (
            batch_idx * out2_stride_batch + 
            dim1_idx * out2_stride_dim1 + 
            split2_offset + 
            tl.arange(0, BLOCK_SIZE)
        )
        tl.store(out2_ptr + out_offsets, out, mask=mask)


@torch.fx.wrap
def silu_and_split_wrapper(uv):
    """
    Wrapper function for the fused SiLU + split kernel.
    
    Args:
        uv: Input tensor of shape [batch, 17, 1152], dtype bfloat16/float16/float32
        
    Returns:
        tuple of (out0, out1, out2):
        - out0: shape [batch, 17, 512]
        - out1: shape [batch, 17, 512]
        - out2: shape [batch, 17, 128]
    """
    batch_size, dim1_size, dim2_size = uv.shape
    
    # Split sizes along dim=2
    split_0_size = 512
    split_1_size = 512
    split_2_size = 128
    
    # Create output tensors
    out0 = torch.empty((batch_size, dim1_size, split_0_size), 
                       dtype=uv.dtype, device=uv.device)
    out1 = torch.empty((batch_size, dim1_size, split_1_size), 
                       dtype=uv.dtype, device=uv.device)
    out2 = torch.empty((batch_size, dim1_size, split_2_size), 
                       dtype=uv.dtype, device=uv.device)
    
    # Define grid for the kernel
    # Use batch_size * dim1_size programs, each handling one (batch, dim1) position
    grid = (batch_size, dim1_size, 1)
    
    # Block size for processing each split region
    BLOCK_SIZE = 128
    
    # Launch the kernel
    silu_and_split_kernel[grid](
        uv, out0, out1, out2,
        batch_size, dim1_size, dim2_size,
        split_0_size, split_1_size, split_2_size,
        uv.stride(0), uv.stride(1), uv.stride(2),
        out0.stride(0), out0.stride(1), out0.stride(2),
        out1.stride(0), out1.stride(1), out1.stride(2),
        out2.stride(0), out2.stride(1), out2.stride(2),
        BLOCK_SIZE
    )
    
    return out0, out1, out2


# Module-level replacement function (required by the framework)
def fused_silu_split_replacement(gamma, uv):
    """
    Fused replacement for silu + split + unsqueeze + expand.
    
    Args:
        gamma: Tensor of shape [2, 128] - the gamma weight tensor
        uv: Tensor of shape [batch, 17, 1152] - the uv input tensor
        
    Returns:
        tuple: (expanded_gamma, split0, unsqueezed_split2, split1)
    """
    # Run the fused silu + split kernel
    split0, split1, split2 = silu_and_split_wrapper(uv)
    
    # Unsqueeze the third part
    unsqueezed_split2 = split2.unsqueeze(2)
    
    # Expand gamma tensor: [2, 128] -> [1, 1, 2, 128]
    expanded_gamma = gamma[(None, None, slice(None, None, None))]
    
    return (expanded_gamma, split0, unsqueezed_split2, split1)


def pattern(in_0, in_1):
    """
    Match the silu + split + unsqueeze + expand pattern.
    
    Pattern:
        tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
        split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
        tmp_3 = split[0]
        tmp_4 = split[1]
        tmp_5 = split[2]
        tmp_6 = tmp_5.unsqueeze(2)
        tmp_7 = in_0[None, None, :]
        return (tmp_7, tmp_3, tmp_6, tmp_4)
    """
    # Apply silu inplace (but we need the result for splitting)
    # Note: The inplace=True is just a hint, split creates new tensors
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    
    # Split along dim=2
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    
    # Get the three parts
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    
    # Unsqueeze the last part
    tmp_6 = tmp_5.unsqueeze(2)
    
    # Expand the gamma tensor
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    in_0 is the gamma tensor for expansion
    in_1 is the uv tensor for silu + split
    """
    return (in_0, in_1)


def replacement_func():
    """
    Returns the replacement function that implements the fused kernel.
    Must return a module-level function reference.
    """
    return fused_silu_split_replacement