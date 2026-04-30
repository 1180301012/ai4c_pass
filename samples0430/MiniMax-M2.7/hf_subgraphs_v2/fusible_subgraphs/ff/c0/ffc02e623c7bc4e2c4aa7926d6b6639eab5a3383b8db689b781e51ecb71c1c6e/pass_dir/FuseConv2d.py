import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_3x3_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H, W,
    K, R, S,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    out_H: tl.constexpr, out_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Conv2D kernel for 3x3 kernels with stride=1, padding=1.
    Grid: (N * K) programs, each computes one output channel for one batch element.
    """
    pid = tl.program_id(0)
    
    elements_per_program = (N * K + BLOCK_SIZE - 1) // BLOCK_SIZE
    start = pid * elements_per_program
    end = min(start + elements_per_program, N * K)
    
    for idx in range(start, end):
        n = idx // K
        out_c = idx % K
        
        for h in range(out_H):
            for w in range(out_W):
                in_h = h - pad_h
                in_w = w - pad_w
                
                acc = 0.0
                for c in range(C_in):
                    for r in range(R):
                        for s in range(S):
                            ih = in_h + r
                            iw = in_w + s
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                in_idx = n * C_in * H * W + c * H * W + ih * W + iw
                                w_idx = out_c * C_in * R * S + c * R * S + r * S + s
                                x = tl.load(input_ptr + in_idx)
                                w_val = tl.load(weight_ptr + w_idx)
                                acc = acc + x * w_val
                
                out_idx = n * K * out_H * out_W + out_c * out_H * out_W + h * out_W + w
                tl.store(output_ptr + out_idx, acc)


@torch.fx.wrap
def conv2d_wrapper(input_tensor, weight_tensor):
    N, C_in, H, W = input_tensor.shape
    K, C_w, R, S = weight_tensor.shape
    
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    out_H = (H + 2 * pad_h - R) // stride_h + 1
    out_W = (W + 2 * pad_w - S) // stride_w + 1
    
    output = torch.empty((N, K, out_H, out_W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 256
    num_programs = (N * K + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv2d_3x3_kernel[(num_programs,)](
        input_tensor, weight_tensor, output,
        N, C_in, H, W,
        K, R, S,
        stride_h, stride_w,
        pad_h, pad_w,
        out_H, out_W,
        BLOCK_SIZE,
    )
    
    return output


def pattern(in_6, in_0):
    conv2d = torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    return conv2d


def replacement_args(in_6, in_0):
    return (in_6, in_0)


def replacement_func():
    return conv2d_wrapper