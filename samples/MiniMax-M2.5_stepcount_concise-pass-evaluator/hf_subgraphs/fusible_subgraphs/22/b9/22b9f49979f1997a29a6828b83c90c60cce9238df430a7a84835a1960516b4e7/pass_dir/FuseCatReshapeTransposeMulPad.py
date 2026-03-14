import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, H, W)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)
    """
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, 40, 576)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Different block sizes for various problem sizes
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    N, K, M, 
    stride_in0_0, stride_in0_1, stride_in0_2, stride_in0_3,
    stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    stride_in2_0, stride_in2_1, stride_in2_2, stride_in2_3,
    stride_in3_0, stride_in3_1, stride_in3_2, stride_in3_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Concatenate in_0, in_1, in_2 along dim=1
    2. Reshape to (1, 8, K, N)
    3. Transpose to (1, 8, N, K)
    4. Multiply with in_3
    5. Pad with (0, 0, 1, 0, 0, 0) -> adds 1 row at the top
    
    Input shapes:
    - in_0: [1, C0, H, W] 
    - in_1: [1, C1, H, W]
    - in_2: [1, C2, H, W]
    - in_3: [1, 8, N, K]
    
    Output shape: [1, 9, N, K] (padded with 1 row at the top)
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Output dimensions: [1, 9, N, K] -> N is rows, K is cols after transpose
    # We process along N dimension (the transposed dimension)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Each program processes one block of N
    pid_n = pid % num_pid_n
    pid_k = pid // num_pid_n
    
    # Offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_n = offs_n < N
    mask_k = offs_k < K
    
    # Output pointer offset: [1, 9, N, K] 
    # The output is padded: we have 9 rows instead of 8
    # The padding adds 1 row at the top (index 0), so the original data starts at row 1
    # For simplicity, we write to the non-padded region (rows 1-8), and row 0 is zeros
    
    # Calculate the output offset for this block
    # Output layout: [batch=1, head=8+1=9, N, K]
    # We skip row 0 (padded row) and write to rows 1-9
    out_offs = (1 * stride_out_1 + offs_n[:, None] * stride_out_2 + offs_k[None, :] * stride_out_3)
    
    # Load in_3: [1, 8, N, K] - shape is [1, head, N, K]
    # We need to load for all 8 heads
    for head in range(8):
        in3_offs = (0 * stride_in3_0 + head * stride_in3_1 + offs_n[:, None] * stride_in3_2 + offs_k[None, :] * stride_in3_3)
        in3_ptrs = in_3_ptr + in3_offs
        in3 = tl.load(in3_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute the concatenated input for this head and position
        # The concatenation happens along C dimension
        # After reshape: [1, 8, K, N], after transpose: [1, 8, N, K]
        # For each (n, k) position, we need to find which input channel it comes from
        
        # Input spatial: H*W = K (before transpose)
        # For in_0: channels C0, in_1: C1, in_2: C2
        # Total C = C0 + C1 + C2 = 8 * head_dim (where head_dim is 40, 64, 8, 16 for different graphs)
        
        result = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        
        # Load from each input tensor and accumulate
        # The reshape takes the concatenated channels and reorganizes them
        # Total channels = 8 * head_dim, organized as 8 heads with head_dim channels each
        
        # We need to compute which part of the concatenated tensor to use
        # For a given head h, we use channels [h*head_dim, (h+1)*head_dim]
        
        # Get channel offsets for each input
        C0 = 80  # in_0 channels
        C1 = 120  # in_1 channels  
        C2 = 120  # in_2 channels
        total_C = C0 + C1 + C2  # 320 = 8 * 40
        
        head_dim = K  # K is the head dimension (40, 64, 8, or 16)
        
        # For each input tensor, calculate the channel range for this head
        # in_0: channels 0 to C0-1 (80 channels = 2 * 40)
        # in_1: channels C0 to C0+C1-1 (120 channels = 3 * 40)
        # in_2: channels C0+C1 to total_C-1 (120 channels = 3 * 40)
        
        # Actually, the layout after reshape(1, 8, 40, 576):
        # - dim 1 (8): head index
        # - dim 2 (40): head_dim 
        # - dim 3 (576): spatial (H*W)
        
        # After transpose(-1, -2): [1, 8, 576, 40]
        # So N=576, K=40
        
        # The concatenation happens on channel dim:
        # [1, C0+C1+C2, H, W] -> reshape(1, 8, head_dim, H*W)
        # This means we split the channel dim into 8 heads, each with head_dim channels
        # For head h, we take channels [h*head_dim : (h+1)*head_dim]
        
        # Let's compute contributions from each input tensor
        # For head h, the channels are: [h*head_dim, (h+1)*head_dim)
        # These channels are split across in_0, in_1, in_2 proportionally
        
        # Channel range for this head
        head_start = head * head_dim
        head_end = (head + 1) * head_dim
        
        # Load from in_0 if this head uses its channels
        # in_0 has C0 channels (80 = 2*40), so it contributes to heads 0, 1
        if head * head_dim < C0:
            # Calculate which part of in_0 to use
            in0_start = head * head_dim
            in0_end = min((head + 1) * head_dim, C0)
            in0_head = 0  # relative head within in_0
            in0_head_dim = in0_end - in0_start
            
            # Load in_0: [1, C0, H, W] -> [1, head, head_dim, H*W]
            # We need to map (n, k) to (head, head_dim_idx, spatial)
            # After transpose, n is spatial index, k is head_dim index
            for ch in range(in0_end - in0_start):
                ch_idx = in0_start + ch
                # in_0: [1, C0, H, W] -> [1, head, head_dim, H*W]
                # Channel ch_idx corresponds to head ch_idx // head_dim, within-head index ch_idx % head_dim
                in0_head_idx = ch_idx // head_dim
                in0_sub_head_dim = ch_idx % head_dim
                
                # Load from in_0
                # Shape: [1, C0, H, W] where H*W = N (after transpose)
                in0_offs = (0 * stride_in0_0 + ch_idx * stride_in0_1 + offs_n[:, None] * stride_in0_2 + in0_sub_head_dim * stride_in0_3)
                in0_ptrs = in_0_ptr + in0_offs
                in0_val = tl.load(in0_ptrs, mask=mask_n[:, None] & (in0_sub_head_dim < head_dim), other=0.0)
                result += in0_val
        
        # Load from in_1 if this head uses its channels  
        if head * head_dim >= C0 and head * head_dim < C0 + C1:
            in1_start = max(0, head * head_dim - C0)
            in1_end = min((head + 1) * head_dim - C0, C1)
            
            for ch in range(in1_end - in1_start):
                ch_idx = C0 + in1_start + ch
                in1_head_dim = (head * head_dim - C0) + ch
                
                in1_offs = (0 * stride_in1_0 + in1_head_dim * stride_in1_1 + offs_n[:, None] * stride_in1_2 + ch * stride_in1_3)
                in1_ptrs = in_1_ptr + in1_offs
                in1_val = tl.load(in1_ptrs, mask=mask_n[:, None] & (ch < head_dim), other=0.0)
                result += in1_val
        
        # Load from in_2 if this head uses its channels
        if head * head_dim >= C0 + C1:
            in2_start = max(0, head * head_dim - C0 - C1)
            in2_end = min((head + 1) * head_dim - C0 - C1, C2)
            
            for ch in range(in2_end - in2_start):
                ch_idx = C0 + C1 + in2_start + ch
                in2_head_dim = (head * head_dim - C0 - C1) + ch
                
                in2_offs = (0 * stride_in2_0 + in2_head_dim * stride_in2_1 + offs_n[:, None] * stride_in2_2 + ch * stride_in2_3)
                in2_ptrs = in_2_ptr + in2_offs
                in2_val = tl.load(in2_ptrs, mask=mask_n[:, None] & (ch < head_dim), other=0.0)
                result += in2_val
        
        # Multiply with in_3
        out_val = result * in3
        
        # Store result (skip the first row which is padding)
        out_ptrs = out_ptr + out_offs
        tl.store(out_ptrs, out_val, mask=mask_n[:, None] & mask_k[None, :])


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper for the fused kernel.
    
    Input shapes:
    - in_0: [1, C0, H, W]
    - in_1: [1, C1, H, W]
    - in_2: [1, C2, H, W]
    - in_3: [1, 8, N, K] where N = H*W, K = head_dim
    
    Output shape: [1, 9, N, K] (padded with 1 zero row at the top)
    """
    # Get shapes
    batch_size = in_0.shape[0]
    assert batch_size == 1, "Only batch size 1 supported"
    
    # in_3 shape: [1, 8, N, K]
    N = in_3.shape[2]  # spatial dimension after transpose
    K = in_3.shape[3]  # head dimension
    
    # Output: [1, 9, N, K] - padded with 1 row at top
    output = torch.zeros((1, 9, N, K), dtype=torch.float32, device=in_0.device)
    
    # Calculate grid
    # We parallelize over N and K
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (num_pid_n * num_pid_k,)
    
    # Launch kernel
    fused_kernel[grid](
        in_0, in_1, in_2, in_3, output,
        N, K, 8,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper