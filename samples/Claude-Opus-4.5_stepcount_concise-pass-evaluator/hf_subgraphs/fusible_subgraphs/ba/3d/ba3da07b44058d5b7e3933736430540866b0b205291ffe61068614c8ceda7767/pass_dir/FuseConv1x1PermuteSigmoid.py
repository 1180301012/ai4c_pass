import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern for conv2d -> permute -> reshape
    Match without sigmoid - sigmoid will be applied by the model afterwards
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(24, -1, 36)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    Cin,
    H,
    W,
    Cout,
    HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv + permute + reshape (NO sigmoid).
    
    Input layout: [B, Cin, H, W] - NCHW
    Weight layout: [Cout, Cin] - flattened from [Cout, Cin, 1, 1]
    Output layout: [B, H*W, Cout] - after permute & reshape
    """
    pid = tl.program_id(0)
    num_hw_blocks = tl.cdiv(HW, BLOCK_HW)
    
    batch_id = pid // num_hw_blocks
    hw_block_id = pid % num_hw_blocks
    
    hw_start = hw_block_id * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW
    
    cout_offs = tl.arange(0, BLOCK_COUT)
    cout_mask = cout_offs < Cout
    
    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_HW, BLOCK_COUT), dtype=tl.float32)
    bias = tl.load(bias_ptr + cout_offs, mask=cout_mask, other=0.0)
    acc += bias[None, :]
    
    # Matrix multiplication over input channels
    for k_start in range(0, Cin, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Cin
        
        # Load input block [BLOCK_HW, BLOCK_K]
        inp_idx = batch_id * Cin * HW + k_offs[None, :] * HW + hw_offs[:, None]
        inp = tl.load(input_ptr + inp_idx, mask=hw_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight transposed [BLOCK_K, BLOCK_COUT]
        wgt_idx = cout_offs[None, :] * Cin + k_offs[:, None]
        wgt = tl.load(weight_ptr + wgt_idx, mask=k_mask[:, None] & cout_mask[None, :], other=0.0)
        
        # Matrix multiply: [BLOCK_HW, BLOCK_K] @ [BLOCK_K, BLOCK_COUT]
        acc += tl.dot(inp, wgt)
    
    # NO sigmoid - output directly
    result = acc
    
    # Store output: output[batch, hw, c] at output_ptr[batch * HW * Cout + hw * Cout + c]
    out_idx = batch_id * HW * Cout + hw_offs[:, None] * Cout + cout_offs[None, :]
    out_mask = hw_mask[:, None] & cout_mask[None, :]
    tl.store(output_ptr + out_idx, result, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1(in_0, in_1, in_2):
    """
    Replacement function for conv2d + permute + reshape.
    in_0: bias [Cout]
    in_1: weight [Cout, Cin, 1, 1]
    in_2: input [B, Cin, H, W]
    """
    bias = in_0
    weight = in_1
    input_tensor = in_2
    
    B, Cin, H, W = input_tensor.shape
    Cout = weight.shape[0]
    HW = H * W
    
    # Allocate output tensor with shape [B, H*W, Cout]
    output = torch.empty((B, HW, Cout), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten weight from [Cout, Cin, 1, 1] to [Cout, Cin]
    weight_flat = weight.view(Cout, Cin)
    
    # Kernel block sizes
    BLOCK_HW = 128
    BLOCK_K = 32
    BLOCK_COUT = 64
    
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (B * num_hw_blocks,)
    
    fused_conv1x1_kernel[grid](
        input_tensor,
        weight_flat,
        bias,
        output,
        B, Cin, H, W, Cout, HW,
        BLOCK_HW=BLOCK_HW,
        BLOCK_K=BLOCK_K,
        BLOCK_COUT=BLOCK_COUT,
    )
    
    return output


def replacement_func():
    return fused_conv1x1