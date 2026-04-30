import torch
import triton
import triton.language as tl


@triton.jit
def fused_se_residual_kernel(
    in_1_ptr,
    weights_ptr,
    out_ptr,
    N: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SE Residual Block kernel
    
    Fuses: sigmoid -> multiply -> add -> relu
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Channel index
    c = offsets // HW
    
    # Load weight and compute sigmoid in fp32
    w = tl.load(weights_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-w))
    
    # Scale: 1 + sigmoid
    scale = 1.0 + sigmoid
    
    # Load feature map and compute
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Fused: scale * x + relu
    scaled = x * scale
    out = tl.where(scaled > 0, scaled, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_se_residual(in_0: torch.Tensor, in_1: torch.Tensor) -> torch.Tensor:
    """
    Fused SE Residual Block
    """
    B, C, H, W = in_1.shape
    N = C * H * W
    HW = H * W
    
    out = torch.empty_like(in_1)
    
    # Grid: one program per channel, 4096 threads per block
    BLOCK_SIZE = HW  # 4096
    grid = (C,)  # 512
    
    fused_se_residual_kernel[grid](
        in_1_ptr=in_1,
        weights_ptr=in_0,
        out_ptr=out,
        N=N,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0: torch.Tensor, in_1: torch.Tensor):
    """
    Pattern matches the SE Residual Block:
    sigmoid -> view -> multiply -> add -> relu_ -> dropout2d
    """
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor):
    return (in_0, in_1)


def replacement_func():
    return fused_se_residual