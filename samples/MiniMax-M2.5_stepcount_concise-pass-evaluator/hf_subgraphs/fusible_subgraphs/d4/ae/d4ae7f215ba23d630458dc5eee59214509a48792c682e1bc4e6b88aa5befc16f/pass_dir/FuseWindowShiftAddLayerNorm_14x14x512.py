import torch
import triton
import triton.language as tl


@triton.jit
def triton_layernorm_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                            N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Triton layer norm kernel - one program per sequence position"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(input_ptr + pid * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    variance = tl.sum(diff * diff, axis=0) / N
    std = tl.sqrt(variance + 1e-5)
    normalized = diff / std
    
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    result = normalized * w + b
    
    tl.store(output_ptr + pid * N + offs, result, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """Match full pattern from graph with 14x14x512"""
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 14, 14, 512)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 196, 512)
    tmp_6 = in_2 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (512,), in_1, in_0, 1e-05)
    return tmp_6, tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def triton_layernorm_wrapper(in_0, in_1, in_2, in_3):
    """Triton layer norm - just replace layer_norm with Triton"""
    # For the add result, we can't recompute due to torch.roll being blocked
    # So we'll return the input as-is and compute only layer_norm with Triton
    # The graph will have the add result passed through
    
    # This is tricky - we need the add result from somewhere
    # Since we can't call torch.roll, we'll need to accept the input as-is
    # and only apply Triton to the layer_norm part
    
    # Actually, let's just compute layer_norm with Triton
    N = 512
    # We need the input tensor for layer_norm - it's the result of in_2 + (rolled view)
    # But we can't compute that without torch.roll
    
    # For now, just use PyTorch's layer_norm and return both
    # This won't give speedup but will let us verify the pattern
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 14, 14, 512)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 196, 512)
    tmp_6 = in_2 + tmp_5
    
    # Use Triton for layer_norm only
    output = torch.empty_like(tmp_6)
    SEQ_LEN = 196
    grid = (SEQ_LEN,)
    triton_layernorm_kernel[grid](tmp_6, in_1, in_0, output, N, N)
    
    return tmp_6, output


def replacement_func():
    return triton_layernorm_wrapper