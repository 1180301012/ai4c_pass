import torch
import triton
import triton.language as tl


# Pattern matching function - matches mul -> silu (simpler pattern)
def pattern(in_4, in_5):
    """
    Simple pattern: in_5 * in_4 -> silu
    """
    tmp_4 = in_5 * in_4
    tmp_6 = torch.nn.functional.silu(tmp_4, inplace=True)
    return tmp_6


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # For this simple pattern, we only need in_4 and in_5
    return (in_4, in_5)


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# Fused Triton kernel: mul -> batch_norm -> silu
@triton.jit
def fused_mul_bn_silu_kernel(
    # Input pointers
    x_ptr,          # in_5: [B, N, H, W]
    weight_mul_ptr, # in_4: [B, N, 1, 1]
    # BN parameters
    mean_ptr,       # in_0: [N]
    var_ptr,        # in_1: [N]
    bias_ptr,       # in_2: [N]
    weight_ptr,     # in_3: [N]
    # Output
    output_ptr,
    # Shapes
    B: tl.constexpr,  # Batch size
    N: tl.constexpr,  # Channel dimension
    H: tl.constexpr,  # Height
    W: tl.constexpr,  # Width
    # Constants
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program and grid info
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Calculate total elements and stride
    total_elements = B * N * H * W
    nhw = H * W  # spatial size per batch
    nb = N * nhw  # elements per batch
    
    # Block starting position
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < total_elements
    
    # Compute batch, channel, h, w indices from flat index
    # flat_idx = b * N * H * W + n * H * W + h * W + w
    # But with NCHW layout: ((b * N + n) * H + h) * W + w
    # Actually PyTorch uses contiguous NCHW: ((b * N + n) * H + h) * W + w
    
    # Compute indices
    batch_idx = offsets // nb
    remainder = offsets % nb
    channel_idx = remainder // nhw
    spatial_idx = remainder % nhw
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load input x at the computed indices
    # x is [B, N, H, W] in NCHW format, flattened as [B*N*H*W]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute index for weight_mul: [B, N, 1, 1] -> flattened as [B*N]
    # Flattened index = b * N + n
    weight_mul_flat_idx = batch_idx * N + channel_idx
    weight_mul = tl.load(weight_mul_ptr + weight_mul_flat_idx, mask=mask, other=0.0)
    
    # Step 1: Multiplication with broadcasting
    mul_out = x * weight_mul
    
    # Load BN parameters using channel_idx
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # Step 2: Batch normalization (using running stats, training=False)
    std = tl.sqrt(var + EPS)
    normalized = (mul_out - mean) / std
    bn_out = normalized * weight + bias
    
    # Step 3: SiLU activation: x * sigmoid(x)
    sigmoid_val = tl.sigmoid(bn_out)
    silu_out = bn_out * sigmoid_val
    
    # Store result
    tl.store(output_ptr + offsets, silu_out, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused kernel: multiplication -> batch normalization -> SiLU
    """
    # Get shapes
    # in_5 (x): [B, N, H, W]
    # in_4 (weight_mul): [B, N, 1, 1] 
    # in_0 (mean): [N]
    # in_1 (var): [N]
    # in_2 (bias): [N]
    # in_3 (weight): [N]
    
    B, N, H, W = in_5.shape
    total_elements = in_5.numel()
    
    # Allocate output
    output = torch.empty_like(in_5)
    
    # Determine block size - need to handle various tensor sizes
    BLOCK_SIZE = 4096
    
    # Calculate number of programs
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Limit the number of programs to avoid too many small kernels
    # Use at least 32 programs for good GPU utilization
    num_programs = max(num_programs, 32)
    # But cap at 8192 to avoid excessive kernel count
    num_programs = min(num_programs, 8192)
    
    # Launch kernel
    fused_mul_bn_silu_kernel[(num_programs,)](
        x_ptr=in_5,
        weight_mul_ptr=in_4,
        mean_ptr=in_0,
        var_ptr=in_1,
        bias_ptr=in_2,
        weight_ptr=in_3,
        output_ptr=output,
        B=B,
        N=N,
        H=H,
        W=W,
        EPS=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_kernel_wrapper