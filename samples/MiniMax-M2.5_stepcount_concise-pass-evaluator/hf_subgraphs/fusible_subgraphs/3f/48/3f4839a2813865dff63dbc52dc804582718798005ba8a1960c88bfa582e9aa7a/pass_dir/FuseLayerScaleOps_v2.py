import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    """
    Simpler pattern: match just the first part of the computation:
    - avg_pool2d on in_2
    - Subtract in_2 from pooled result
    - Multiply by reshaped in_0
    - Add to in_2
    """
    # Copy input
    tmp_0 = in_0
    
    # avg_pool2d
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    
    # Subtract
    tmp_3 = tmp_2 - in_2
    
    # Reshape
    tmp_4 = tmp_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    
    # Multiply
    tmp_6 = tmp_5 * tmp_3
    
    # Add
    tmp_7 = in_2 + tmp_6
    
    return tmp_7


def replacement_args(in_0, in_2):
    """Extract arguments needed for the replacement."""
    return (in_0, in_2)


def replacement_func():
    """Return the replacement function."""
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        ],
        key=['B', 'C', 'H', 'W'],
    )
    @triton.jit
    def fused_kernel(
        in_ptr,
        scale1_ptr,
        out_ptr,
        B: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_elements = B * C * H * W
        
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < num_elements
        
        w = offset % W
        h = (offset // W) % H
        c = (offset // (W * H)) % C
        b = offset // (W * H * C)
        
        in_val = tl.load(in_ptr + offset, mask=mask, other=0.0)
        scale1 = tl.load(scale1_ptr + c)
        
        # Compute 3x3 avg pooling manually with boundary handling
        sum_val = tl.zeros_like(in_val)
        
        # 9 positions with clamping
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                h_n = h + kh
                w_n = w + kw
                h_n = tl.where(h_n < 0, 0, tl.where(h_n >= H, H - 1, h_n))
                w_n = tl.where(w_n < 0, 0, tl.where(w_n >= W, W - 1, w_n))
                off = b * C * H * W + c * H * W + h_n * W + w_n
                sum_val = sum_val + tl.load(in_ptr + off, mask=mask, other=0.0)
        
        avg_val = sum_val / 9.0
        diff = avg_val - in_val
        out_val = in_val + scale1 * diff
        
        tl.store(out_ptr + offset, out_val, mask=mask)
    
    @torch.fx.wrap
    def wrapper(scale1, in_2):
        B, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        num_elements = B * C * H * W
        grid = (num_elements + 1024 - 1) // 1024
        
        fused_kernel[grid](
            in_ptr=in_2,
            scale1_ptr=scale1,
            out_ptr=out,
            B=B, C=C, H=H, W=W,
        )
        return out
    
    return wrapper