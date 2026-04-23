import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the einsum + add + mul + add + contiguous pattern.
    
    einsum: bchj,bhwj->bchw over dimension j
    in_0: scalar (scale factor)
    in_1: [B, 64, 64, 64]
    in_2: [B, 512, 64, 64]
    in_3: [B, 512, 64, 64] - accumulated with einsum result
    in_4: [B, 512, 64, 64]
    """
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1);  in_4 = in_1 = None
    in_3 += einsum;  in_5 = in_3;  in_3 = einsum = None
    tmp_3 = in_5 * in_0;  in_5 = in_0 = None
    tmp_4 = tmp_3 + in_2;  tmp_3 = in_2 = None
    tmp_5 = tmp_4.contiguous();  tmp_4 = None
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_einsum_bmm_kernel(
    # in_4 ptr: [B, C, H, W] where C=512, H=W=64
    in_4_ptr, in_4_batch_stride, in_4_channel_stride, in_4_h_stride, in_4_w_stride,
    # in_1 ptr: [B, H, W, W]  
    in_1_ptr, in_1_batch_stride, in_1_h_stride, in_1_w_stride,
    # in_2 ptr: [B, C, H, W]
    in_2_ptr, in_2_batch_stride, in_2_channel_stride, in_2_h_stride, in_2_w_stride,
    # in_3 ptr: [B, C, H, W] (accumulation)
    in_3_ptr, in_3_batch_stride, in_3_channel_stride, in_3_h_stride, in_3_w_stride,
    # scalar scale
    scale,
    # output ptr
    out_ptr, out_batch_stride, out_channel_stride, out_h_stride, out_w_stride,
    # sizes
    B, C, H, W,
    # block dimensions
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel for einsum(bchj,bhwj->bchw) + add(gamma) + mul(scale) + add(bias) + contiguous
    
    The einsum computes: out[b,c,h,w] = sum_j(in_4[b,c,h,j] * in_1[b,h,w,j])
    This is equivalent to: out[b,c,h,:] = in_4[b,c,h,:] @ in_1[b,h,:,:]
    """
    # Get batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Get pointers to the start of this batch and channel
    # in_4[b, c, h, w] = in_4_ptr + b*in_4_batch_stride + c*in_4_channel_stride + h*in_4_h_stride + w*in_4_w_stride
    in_4_base = in_4_ptr + batch_idx * in_4_batch_stride + channel_idx * in_4_channel_stride
    
    # in_1[b, h, w, j] - we need row h for all w,j
    # in_1_ptr + b*in_1_batch_stride + h*in_1_h_stride + w*in_1_w_stride + j
    
    # Result accumulator
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    
    # Compute einsum: out[b,c,h,w] = sum_j(in_4[b,c,h,j] * in_1[b,h,w,j])
    # For each h, we compute a dot product of in_4[b,c,h,:] with in_1[b,h,w,:]
    
    for h_idx in range(H):
        # Load in_4[b, c, h_idx, :] - the query vector [W]
        in_4_offsets = h_idx * in_4_h_stride + tl.arange(0, BLOCK_W)
        in_4_mask = in_4_offsets < W
        in_4_vals = tl.load(in_4_base + in_4_offsets, mask=in_4_mask, other=0.0)
        
        # For each output position (h_idx, w), compute dot product with in_1[b, h_idx, w, :]
        for w_idx in range(BLOCK_W):
            w_actual = w_idx
            if w_actual < W:
                # in_1[b, h_idx, w_actual, :] - the key vector [W]
                # Layout: [b, h, w, j] -> stride: [H*W*W, W*W, W, 1]
                in_1_base_w = in_1_ptr + batch_idx * in_1_batch_stride + h_idx * in_1_h_stride + w_actual * in_1_w_stride
                in_1_offsets = tl.arange(0, BLOCK_W)
                in_1_mask = in_1_offsets < W
                in_1_vals = tl.load(in_1_base_w + in_1_offsets, mask=in_1_mask, other=0.0)
                
                # Dot product
                prod = in_4_vals * in_1_vals
                acc[h_idx, w_actual] += tl.sum(prod)
    
    # Load in_3[b, c, h, w] for accumulation
    in_3_base = in_3_ptr + batch_idx * in_3_batch_stride + channel_idx * in_3_channel_stride
    
    # Apply scale and add in_2
    for h_idx in range(BLOCK_H):
        for w_idx in range(BLOCK_W):
            if h_idx < H and w_idx < W:
                # Load accumulated einsum result
                einsum_val = acc[h_idx, w_idx]
                
                # Load in_3 for accumulation
                offset = h_idx * in_3_h_stride + w_idx * in_3_w_stride
                in_3_val = tl.load(in_3_base + offset)
                
                # Apply scale and accumulate
                scaled = einsum_val * scale
                final_val = scaled + in_3_val
                
                # Load and add in_2
                in_2_base = in_2_ptr + batch_idx * in_2_batch_stride + channel_idx * in_2_channel_stride
                in_2_val = tl.load(in_2_base + offset)
                final_val = final_val + in_2_val
                
                # Store to output
                out_base = out_ptr + batch_idx * out_batch_stride + channel_idx * out_channel_stride
                tl.store(out_base + offset, final_val)


@torch.fx.wrap
def fused_einsum_bmm(in_0, in_1, in_2, in_3, in_4):
    """
    Fused implementation using bmm for the einsum operation.
    
    The einsum 'bchj,bhwj->bchw' contracts over j:
    out[b,c,h,w] = sum_j(in_4[b,c,h,j] * in_1[b,h,w,j])
    
    This can be expressed as bmm:
    For each b,c: out[b,c,h,:] = in_4[b,c,h,:] @ in_1[b,h,:,:]
    """
    B, C, H, W = in_4.shape
    W2 = in_1.shape[-1]  # Should equal W (64)
    
    # Reshape for bmm:
    # in_4: [B, C, H, W] -> [B, C*H, W]
    in_4_reshaped = in_4.permute(0, 2, 1, 3).reshape(B * H, C, W)
    # in_1: [B, H, W, W] -> [B*H, W, W]
    in_1_reshaped = in_1.permute(0, 1, 3, 2).reshape(B * H, W, W2)
    
    # bmm: [B*H, C, W] @ [B*H, W, W] -> [B*H, C, W]
    bmm_out = torch.bmm(in_4_reshaped, in_1_reshaped)
    
    # Reshape back: [B*H, C, W] -> [B, H, C, W] -> [B, C, H, W]
    bmm_out = bmm_out.reshape(B, H, C, W).permute(0, 2, 1, 3)
    
    # Now apply the remaining operations
    # in_3 += einsum
    result = bmm_out + in_3
    
    # tmp_3 = in_5 * in_0 (scale)
    if in_0.numel() == 1:
        scale = in_0.item()
        result = result * scale
    
    # tmp_4 = tmp_3 + in_2
    result = result + in_2
    
    # Make contiguous
    return result.contiguous()


def replacement_func():
    return fused_einsum_bmm