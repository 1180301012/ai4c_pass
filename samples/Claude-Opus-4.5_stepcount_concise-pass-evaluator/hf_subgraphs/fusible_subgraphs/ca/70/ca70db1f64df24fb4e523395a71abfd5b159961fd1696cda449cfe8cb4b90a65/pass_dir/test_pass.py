import torch
import triton
import triton.language as tl

# Pattern: Match the second linear operation with slicing
# Includes the reshape operation to match from graph inputs
# in_4.reshape(300, -1, 256) -> tmp_9
# torch.nn.functional.linear(tmp_9, in_3, in_2) -> tmp_10
# tmp_10[Ellipsis, slice(None, 256, None)] -> tmp_11
# tmp_10[Ellipsis, slice(-256, None, None)] -> tmp_12

def pattern(in_4, in_3, in_2):
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),
    ],
    key=['K'],
)
@triton.jit
def fused_linear_slice_kernel(
    in_ptr, weight_ptr, bias_ptr, 
    out1_ptr, out2_ptr,
    M, K, N,
    stride_in, stride_w, stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute: out = in @ weight.T + bias
    # in: (M, 1, K), weight: (N, K), bias: (N,)
    # Output: (M, 1, N) but we only compute first and last N/2
    # We compute both halves in one pass
    
    pid = tl.program_id(0)
    row_offset = pid * stride_out
    
    # Compute for both slices
    # Slice 1: cols 0:256, Slice 2: cols 256:512
    
    # Accumulator for first half
    acc1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # Accumulator for second half
    acc2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Input is (M, 1, K), load the (1, K) row
    for k in range(K):
        # Load input element
        inp_val = tl.load(in_ptr + pid * stride_in + k)
        
        # Load weight column k (row k of weight)
        # First half: cols 0-255
        w1_ptrs = weight_ptr + k * stride_w + tl.arange(0, BLOCK_SIZE)
        w1_vals = tl.load(w1_ptrs, mask=tl.arange(0, BLOCK_SIZE) < 256)
        
        # Second half: cols 256-511
        w2_ptrs = weight_ptr + k * stride_w + 256 + tl.arange(0, BLOCK_SIZE)
        w2_vals = tl.load(w2_ptrs, mask=tl.arange(0, BLOCK_SIZE) < 256)
        
        acc1 += inp_val * w1_vals
        acc2 += inp_val * w2_vals
    
    # Add bias
    bias_offs = tl.arange(0, BLOCK_SIZE)
    bias_mask = bias_offs < 256
    
    bias1 = tl.load(bias_ptr + bias_offs, mask=bias_mask)
    bias2 = tl.load(bias_ptr + 256 + bias_offs, mask=bias_mask)
    
    result1 = acc1 + bias1
    result2 = acc2 + bias2
    
    # Store outputs - each output is (1, 256) but we store as (256,)
    # We need to handle the 3D output structure
    out1_ptrs = out1_ptr + pid * 256 + bias_offs  # (M, 1, 256) flattened
    out2_ptrs = out2_ptr + pid * 256 + bias_offs
    
    tl.store(out1_ptrs, result1, mask=bias_mask)
    tl.store(out2_ptrs, result2, mask=bias_mask)


@torch.fx.wrap
def fused_linear_slice_wrapper(in_4, in_3, in_2):
    # in_4: (1, 150, 1, 512) - proposal_feat
    # in_3: (N, K) - weight (512, 512)  
    # in_2: (N,) - bias (512,)
    # Output: (M, 1, 256), (M, 1, 256)
    
    # Reshape in_4 to (M, 1, K)
    tmp_9 = in_4.reshape(300, -1, 256)  # (300, 1, 512)
    
    # Handle 3D input: squeeze to 2D, compute, then unsqueeze
    if tmp_9.dim() == 3:
        input_2d = tmp_9.squeeze(1)  # (M, K)
        squeeze_output = True
    else:
        input_2d = tmp_9
        squeeze_output = False
    
    M, K = input_2d.shape
    N = in_3.shape[0]  # 512
    
    BLOCK_SIZE = 256
    
    # Allocate outputs as 2D first
    out1 = torch.empty((M, 256), dtype=torch.float32, device=in_4.device)
    out2 = torch.empty((M, 256), dtype=torch.float32, device=in_4.device)
    
    grid = (M,)
    
    fused_linear_slice_kernel[grid](
        in_ptr=input_2d,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out1_ptr=out1,
        out2_ptr=out2,
        M=M,
        K=K,
        N=N,
        stride_in=K,
        stride_w=K,
        stride_out=256,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to 3D if needed
    if squeeze_output:
        out1 = out1.unsqueeze(1)  # (M, 1, 256)
        out2 = out2.unsqueeze(1)  # (M, 1, 256)
    
    return out1, out2


def replacement_func():
    return fused_linear_slice_wrapper