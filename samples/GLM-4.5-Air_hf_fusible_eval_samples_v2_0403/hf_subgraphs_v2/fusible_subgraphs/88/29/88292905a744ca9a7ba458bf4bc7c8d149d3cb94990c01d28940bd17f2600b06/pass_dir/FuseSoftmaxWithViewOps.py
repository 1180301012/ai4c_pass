import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(in_1.shape[0], -1)
    tmp_2 = tmp_1.view(in_1.shape[0], -1, 1, 1)
    tmp_3 = tmp_2.view(in_1.shape[0], 2, -1, 1, 1)
    return tmp_3

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_softmax_view_kernel(
    in_1_ptr, 
    out_ptr,
    N_batch: tl.constexpr,
    N_channel_0: tl.constexpr,
    N_channel_1: tl.constexpr,
    N_channel_2: tl.constexpr,
    N_total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position across all batches and channels
    pid = tl.program_id(0)
    
    if pid >= N_total_elements:
        return
    
    # Process this spatial position across all batches
    for batch_idx in range(N_batch):
        # Load the two feature values at this spatial position for both channels
        in_base = batch_idx * 2 * N_total_elements + pid
        feat0 = tl.load(in_1_ptr + in_base, mask=in_base < N_batch * 2 * N_total_elements)
        feat1 = tl.load(in_1_ptr + in_base + N_total_elements, mask=(in_base + N_total_elements) < N_batch * 2 * N_total_elements)
        
        # Compute softmax across the 2 channels
        max_val = tl.maximum(feat0, feat1)
        exp0 = tl.exp(feat0 - max_val)
        exp1 = tl.exp(feat1 - max_val)
        sum_exp = exp0 + exp1
        softmax0 = exp0 / sum_exp
        softmax1 = exp1 / sum_exp
        
        # Store output - direct reshape to [N_batch, 2, N_total_elements, 1, 1] layout
        # This corresponds to: reshape(8, -1) -> view(8, -1, 1, 1) -> view(8, 2, -1, 1, 1)
        # where -1 becomes N_total_elements and we store in a flat layout
        out_base = batch_idx * 2 * N_total_elements
        tl.store(out_ptr + out_base + pid, softmax0)
        tl.store(out_ptr + out_base + pid + N_total_elements, softmax1)

@torch.fx.wrap
def fused_softmax_view(in_1):
    N_batch = in_1.shape[0]
    N_channel_0 = in_1.shape[1]  # This should be 2
    N_channel_1 = in_1.shape[2]  # This should be 1
    N_channel_2 = in_1.shape[3]  # This varies (128, etc.)
    
    # Total spatial elements to process (excluding the 2-channel dim)
    N_total_elements = N_channel_1 * N_channel_2
    if len(in_1.shape) > 4:
        N_total_elements *= in_1.shape[4]
    
    BLOCK_SIZE = 1024
    
    # Output shape is [N_batch, 2, N_total_elements, 1, 1]
    output_shape = [N_batch, 2, N_total_elements, 1, 1]
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    num_total_elements = N_total_elements
    fused_softmax_view_kernel[(num_total_elements,)](
        in_1_ptr=in_1,
        out_ptr=out,
        N_batch=N_batch,
        N_channel_0=N_channel_0,
        N_channel_1=N_channel_1,
        N_channel_2=N_channel_2,
        N_total_elements=N_total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_softmax_view