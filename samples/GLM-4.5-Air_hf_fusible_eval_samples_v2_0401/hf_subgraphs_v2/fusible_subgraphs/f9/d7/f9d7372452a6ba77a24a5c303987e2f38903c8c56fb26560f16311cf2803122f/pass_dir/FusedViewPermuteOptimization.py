import torch
import triton
import triton.language as tl

def pattern(in_1):
    # Fused view + permute operation
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4

def replacement_args(in_1):
    return (in_1,)

def get_reshaped_shape(in_tensor):
    original_shape = in_tensor.shape
    # For shape [1, 32, 64, 48] ->_view-> [1, 32, 3072] ->permute-> [1, 3072, 32]
    # We need the final shape: [1, 64*48, 32] = [1, 3072, 32]
    batch_size = original_shape[0]
    intermediate_dim = original_shape[1]  # 32
    last_dim = original_shape[2] * original_shape[3]  # 64*48 = 3072
    return (batch_size, last_dim, intermediate_dim)

@triton.jit
def fused_view_permute_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Input: [N, C, H, W] -> Output: [N, H*W, C]
    
    # Each program handles one warp for better GPU utilization
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate output dimensions with total elements
    output_elements = N * (H * W) * C
    mask = offsets < output_elements
    
    # Extract coordinates from flattened output index [N, H*W, C]
    out_idx = offsets
    n = out_idx // ((H * W) * C)
    hw_idx = (out_idx % ((H * W) * C)) // C
    c = out_idx % C
    
    # Map (H, W) coordinates from flat HW index
    h = hw_idx // W
    w = hw_idx % W
    
    # Calculate input index [N, C, H, W] - ensure we handle all dimensions correctly
    in_idx = n * (C * H * W) + c * (H * W) + h * W + w
    
    # Load input and store output directly with memory coalescing
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_view_permute(in_1):
    original_shape = in_1.shape
    N, C, H_in, W_in = original_shape
    
    # Calculate output shape: [N, H_in*W_in, C]
    H_out = H_in * W_in
    output_shape = (N, H_out, C)
    
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Use larger block size to reduce kernel launch overhead
    n_elements = out.numel()
    BLOCK_SIZE = 4096  # Larger block for better GPU occupancy and reduced overhead
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_view_permute_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        N=N,
        C=C,
        H=H_in,
        W=W_in,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_view_permute