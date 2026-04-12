import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    in_0_ptr,  # [1, 512] - sigmoid input
    in_1_ptr,  # [1, 512, 64, 64] - main input tensor
    out_ptr,   # [1, 512, 64, 64] - output tensor
    
    N, C, H, W,  # 1, 512, 64, 64
    one_val: tl.constexpr,  # Scalar value 1.0
    dropout_scale: tl.constexpr,  # Dropout scale factor
    
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    num_programs = (N * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if pid >= num_programs:
        return
    
    # Compute offsets for this program's block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * H * W)
    
    # Compute channel index for each offset
    idx = offsets
    c_idx = (idx // (H * W)) % C  # Channel index (0-511)
    
    # Load sigmoid values for each channel (preserves original dtype)
    sigmoid_vals = tl.load(in_0_ptr + c_idx, mask=c_idx < C)
    
    # Load input tensor values (preserves original dtype)
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 for precise computation
    sigmoid_f32 = sigmoid_vals.to(tl.float32)
    x_f32 = x.to(tl.float32)
    one_f32 = one_val
    
    # Perform the fused computation: x * (1 + sigmoid(x_channel))
    # This should exactly match the PyTorch computation
    computation = x_f32 * (one_f32 + sigmoid_f32)
    result = tl.maximum(computation, 0.0)
    
    # Apply dropout scaling (eval mode: scale by (1 - dropout_rate))
    result_scaled = result * dropout_scale
    
    # Convert back to original dtype for storage
    result_original = result_scaled.to(x.dtype)
    
    # Store result
    tl.store(out_ptr + offsets, result_original, mask=mask)

@torch.fx.wrap
def fused_forward(in_0, in_1):
    # Get tensor shapes
    N, C = in_0.shape
    _, C_in, H, W = in_1.shape
    
    # Verify shapes are as expected
    assert N == 1, f"Expected N=1, got {N}"
    assert C == C_in, f"Channel mismatch: {C} vs {C_in}"
    assert H == 64 and W == 64, f"Expected H=64, W=64, got H={H}, W={W}"
    
    # Create output tensor
    out = torch.empty((1, C, H, W), dtype=in_1.dtype, device=in_1.device)
    
    # Set up Triton kernel launch with larger block size for better performance
    total_elements = N * C * H * W
    BLOCK_SIZE = 2048  # Increased block size for better occupancy
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Compute shapes for kernel
    N_shape, C_shape, H_shape, W_shape = N, C, H, W
    
    # Launch kernel with one_val as float32 - type will be cast automatically in kernel
    one_val = 1.0
    dropout_scale = 0.9  # (1 - 0.1) for dropout rate 0.1 in eval mode
    
    # Launch kernel with dtype info
    fused_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N=N_shape,
        C=C_shape,
        H=H_shape,
        W=W_shape,
        one_val=one_val,
        dropout_scale=dropout_scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_forward