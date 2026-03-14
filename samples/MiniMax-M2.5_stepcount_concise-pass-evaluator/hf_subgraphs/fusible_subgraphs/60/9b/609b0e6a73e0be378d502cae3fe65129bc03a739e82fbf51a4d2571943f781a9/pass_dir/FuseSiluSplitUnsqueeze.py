import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: silu + split + unsqueeze
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_2 = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_2[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = tmp_0[None, None, slice(None, None, None)]
    return tmp_7, tmp_3, tmp_6, tmp_4


def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement function.
    """
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_K': 1024}, num_stages=3, num_warps=4),
    ],
    key=['batch_size', 'num_kpts'],
)
@triton.jit
def silu_split_kernel(
    input_ptr,
    output0_ptr,
    output1_ptr,
    output2_ptr,
    batch_size,
    num_kpts,
    stride_batch,
    stride_kpt,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused SiLU + split kernel.
    Processes the last dimension (1152 = 512 + 512 + 128) in chunks.
    """
    # Get position
    pid = tl.program_id(0)
    
    # Each program handles one batch * num_kpts position
    b = pid // num_kpts
    k = pid % num_kpts
    
    # Boundary check
    if b >= batch_size:
        return
    
    # Base offsets
    base_offset = b * stride_batch + k * stride_kpt
    
    # Process first split (512 elements)
    for i in tl.range(0, 512, block_size=BLOCK_SIZE_K):
        offset = base_offset + i
        x = tl.load(input_ptr + offset)
        sigmoid_x = tl.sigmoid(x)
        silu_out = x * sigmoid_x
        tl.store(output0_ptr + b * stride_batch * 512 + k * 512 + i, silu_out)
    
    # Process second split (512 elements)
    for i in tl.range(0, 512, block_size=BLOCK_SIZE_K):
        offset = base_offset + 512 + i
        x = tl.load(input_ptr + offset)
        sigmoid_x = tl.sigmoid(x)
        silu_out = x * sigmoid_x
        tl.store(output1_ptr + b * stride_batch * 512 + k * 512 + i, silu_out)
    
    # Process third split (128 elements)
    for i in tl.range(0, 128, block_size=BLOCK_SIZE_K):
        offset = base_offset + 1024 + i
        x = tl.load(input_ptr + offset)
        sigmoid_x = tl.sigmoid(x)
        silu_out = x * sigmoid_x
        tl.store(output2_ptr + b * stride_batch * 128 + k * 128 + i, silu_out)


@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    """
    Fused kernel for silu + split + unsqueeze operations.
    Uses Triton to compute SiLU and split in a single kernel launch.
    """
    # Get input shape
    batch_size = in_1.shape[0]
    num_kpts = in_1.shape[1]
    
    # Create output tensors
    # split[0]: [batch, 17, 512]
    # split[1]: [batch, 17, 512]
    # split[2]: [batch, 17, 128]
    output0 = torch.empty((batch_size, num_kpts, 512), dtype=in_1.dtype, device=in_1.device)
    output1 = torch.empty((batch_size, num_kpts, 512), dtype=in_1.dtype, device=in_1.device)
    output2 = torch.empty((batch_size, num_kpts, 128), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate strides
    stride_batch = in_1.shape[1] * in_1.shape[2]
    stride_kpt = in_1.shape[2]
    
    # Grid: (batch_size * num_kpts,)
    grid = (batch_size * num_kpts,)
    
    # Launch kernel
    silu_split_kernel[grid](
        in_1,
        output0,
        output1,
        output2,
        batch_size,
        num_kpts,
        stride_batch,
        stride_kpt,
    )
    
    # Unsqueeze the third split output to add dimension at position 2
    unsqueezed = output2.unsqueeze(2)
    
    # Add dimensions to in_0
    indexed = in_0[None, None, slice(None, None, None)]
    
    return indexed, output0, unsqueezed, output1


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_silu_split_unsqueeze