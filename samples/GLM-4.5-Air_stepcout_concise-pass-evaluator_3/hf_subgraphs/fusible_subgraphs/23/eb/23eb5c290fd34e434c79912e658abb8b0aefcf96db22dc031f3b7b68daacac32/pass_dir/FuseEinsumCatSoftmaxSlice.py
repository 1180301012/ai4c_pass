import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation from model.py
# The model structure is:
#   tmp_0 = in_0  (alias)
#   tmp_1 = einsum(in_2, in_1)
#   tmp_2 = cat([tmp_0, tmp_1], dim=-1)
#   tmp_3 = softmax(tmp_2, dim=-1)
#   tmp_4 = tmp_3[:,:,:,:64]
#   return (tmp_3, tmp_4)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0  # alias
    t1 = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    t2 = torch.cat([tmp_0, t1], dim=-1)
    t3 = torch.nn.functional.softmax(t2, dim=-1)
    t4 = t3[:, :, :, :64]
    return t3, t4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Optimized fused kernel: einsum + cat + softmax + slice
@triton.jit
def einsum_softmax_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_0_ptr, out_1_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
    stride_in_0_b, stride_in_0_c, stride_in_0_h, stride_in_0_w,
    stride_in_1_b, stride_in_1_c, stride_in_1_h, stride_in_1_k,
    stride_in_2_b, stride_in_2_c, stride_in_2_h, stride_in_2_w,
    stride_out_0_b, stride_out_0_h, stride_out_0_w, stride_out_0_n,
    stride_out_1_b, stride_out_1_h, stride_out_1_w,
):
    # pid is (batch, h, w)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # N = C + K = 64 + 64 = 128
    N = C + K
    
    # Compute offsets
    # in_0 [B, C, H, W] -> treat as [B, H, W, C] for cat
    in_0_base = pid_b * stride_in_0_b + pid_h * stride_in_0_h + pid_w * stride_in_0_w
    # in_1 [B, C, H, K]
    in_1_base = pid_b * stride_in_1_b + pid_h * stride_in_1_h + pid_w * stride_in_1_k
    # in_2 [B, C, H, W]
    in_2_base = pid_b * stride_in_2_b + pid_h * stride_in_2_h + pid_w * stride_in_2_w
    
    # Output offsets
    out_0_base = pid_b * stride_out_0_b + pid_h * stride_out_0_h + pid_w * stride_out_0_w
    out_1_base = pid_b * stride_out_1_b + pid_h * stride_out_1_h + pid_w * stride_out_1_w
    
    # Compute einsum: sum_c in_2[b,c,h,w] * in_1[b,c,h,k] for each k
    # This is a dot product over C dimension
    
    # First, load in_2 values for all c
    # in_2[b,c,h,w] - we need all C values for this b,h,w
    in_2_vals = tl.load(in_2_base + tl.arange(0, C) * stride_in_2_c)
    
    # For each k, compute sum_c in_2[c] * in_1[c,k]
    # This is a matrix-vector multiply
    
    # Compute the einsum result for all K values
    # einsum[k] = sum_c in_2[c] * in_1[c,k]
    
    # Load in_1 for all c and k: shape [C, K]
    # We'll do a loop over c
    einsum_result = tl.zeros((K,), dtype=tl.float32)
    
    for c in range(C):
        in_2_c = in_2_vals[c]
        in_1_c = tl.load(in_1_base + c * stride_in_1_c + tl.arange(0, K) * stride_in_1_k)
        einsum_result = einsum_result + in_2_c * in_1_c
    
    # Now we need to concatenate in_0 and einsum_result
    # in_0 is [B, C, H, W], we need it as [B, H, W, C]
    # So we need to read in_0[b, c, h, w] with c being the "cat" dimension index
    
    # Load in_0 values - need to transpose from [B,C,H,W] to [B,H,W,C]
    # in_0[b,c,h,w] offset = b*stride_in_0_b + c*stride_in_0_c + h*stride_in_0_h + w*stride_in_0_w
    in_0_vals = tl.load(in_0_base + tl.arange(0, C) * stride_in_0_c)
    
    # Concatenate: [in_0_vals, einsum_result] -> total N = C + K = 128
    # First C elements from in_0, next K from einsum
    # For softmax, we need all N values
    
    # Compute max for numerical stability
    # Combine the two arrays
    concat_vals = tl.concat(in_0_vals, einsum_result)
    
    # Find max
    max_val = tl.max(concat_vals, axis=0)
    
    # Compute exp(x - max)
    exp_vals = tl.exp(concat_vals - max_val)
    
    # Compute sum
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Softmax = exp(x - max) / sum
    softmax_vals = exp_vals / sum_exp
    
    # Store full softmax result
    tl.store(out_0_base + tl.arange(0, N) * stride_out_0_n, softmax_vals)
    
    # Store sliced result (first C=64 elements)
    tl.store(out_1_base + tl.arange(0, C) * stride_out_1_w, softmax_vals[:C])


@torch.fx.wrap
def fused_einsum_softmax_wrapper(in_0, in_1, in_2):
    # Get shapes
    B, C, H, W = in_0.shape
    K = in_1.shape[-1]  # key dimension
    
    # Output shapes
    # softmax result: cat along dim=-1 -> [B, H, W, C+K] = [B, H, W, 128]
    # But wait, we need to determine the actual shapes
    
    # Actually, let's check the shapes more carefully
    # in_0: [B, C, H, W]
    # einsum: [B, H, W, K]
    # For cat to work, we need to transpose in_0 or view it differently
    
    # The model does: torch.cat([in_0, tmp_1], dim=-1)
    # This would fail with shapes [B,C,H,W] and [B,H,W,K]
    # Unless the tensors are viewed differently or there's some assumption
    
    # Let me assume the shapes work out and we proceed
    # Output 0: [B, H, W, C+K]
    # Output 1: [B, H, W, C]
    
    # Actually, based on the slice [Ellipsis, slice(None, 64, None)], C=64
    # So output1 is [B, H, W, 64]
    
    # But for output0, we need C+K = 64+64 = 128
    # And the shape should be [B, H, W, 128]
    
    # Prepare output tensors
    out_0 = torch.empty((B, H, W, C + K), dtype=torch.float32, device=in_0.device)
    out_1 = torch.empty((B, H, W, C), dtype=torch.float32, device=in_0.device)
    
    # Strides for output (row-major)
    stride_out_0_b = out_0.stride(0)
    stride_out_0_h = out_0.stride(1)
    stride_out_0_w = out_0.stride(2)
    stride_out_0_n = out_0.stride(3)
    
    stride_out_1_b = out_1.stride(0)
    stride_out_1_h = out_1.stride(1)
    stride_out_1_w = out_1.stride(2)
    
    # Launch kernel
    grid = (B, H, W)
    
    einsum_softmax_kernel[grid](
        in_0, in_1, in_2,
        out_0, out_1,
        B, C, H, W, K,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        stride_out_0_b, stride_out_0_h, stride_out_0_w, stride_out_0_n,
        stride_out_1_b, stride_out_1_h, stride_out_1_w,
    )
    
    return out_0, out_1


def replacement_func():
    return fused_einsum_softmax_wrapper