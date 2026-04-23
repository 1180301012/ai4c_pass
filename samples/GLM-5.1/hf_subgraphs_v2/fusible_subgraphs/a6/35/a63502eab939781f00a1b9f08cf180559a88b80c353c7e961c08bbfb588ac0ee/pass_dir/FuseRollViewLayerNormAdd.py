import torch
import triton
import triton.language as tl

# Pattern matching - must mirror model.py exactly (without cleanup statements)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_roll_layernorm_add_kernel(
    in_ptr,        # in_3: input tensor [1, H*2, W*2, H*2, W*2, C]
    bias_ptr,      # in_0: bias [C]
    weight_ptr,    # in_1: weight [C]
    add_ptr,       # in_2: residual [1, N, C]
    out_ptr,       # output [1, N, C]
    H: tl.constexpr,   # window height (32 or 64)
    W: tl.constexpr,   # window width (32 or 64)
    C: tl.constexpr,   # channels (768 or 384)
    H2: tl.constexpr,  # 2*H
    W2: tl.constexpr,  # 2*W
    SHIFT_H: tl.constexpr,  # shift in H dim (4)
    SHIFT_W: tl.constexpr,  # shift in W dim (4)
    N: tl.constexpr,   # H * W
    BLOCK_C: tl.constexpr,  # block size for C dimension
):
    # Each program handles one row (N dimension)
    row_idx = tl.program_id(0)
    
    # Compute rolled indices for this row
    h = row_idx // W
    w = row_idx % W
    
    rolled_h = (h + SHIFT_H) % H
    rolled_w = (w + SHIFT_W) % W
    
    # Input tensor layout: [1, H2, W2, H2, W2, C]
    # For row_idx = h*W + w, we need h_idx = h // (W2//W) = h // 2 for 64x64 case
    # Actually, H=32 -> H2=4*2=8? No...
    # Wait: in_3 shape is [1, H2, W2, H2, W2, C] where the view maps to [-1, H, W, C]
    # For the 32x32 case: in_3 = [1, 4, 8, 4, 8, 768], view -> [-1, 32, 32, 768]
    # So -1 = 1*4*8*4*8 / (32*32*768) = 1
    # The mapping: dim0=1, dim1=4, dim2=8, dim3=4, dim4=8, dim5=768 -> view to [1, 32, 32, 768]
    # This means: h_out = idx1 * (8*4*8) + idx2 * (4*8) + idx3 * 8 + idx4
    # Hmm, actually let me think about this differently.
    
    # The input tensor in_3 has shape [1, H2, W2, H2, W2, C]
    # After view(-1, H, W, C), we get [1, H, W, C]  
    # The view reshapes dims [1,2,3,4] -> [H, W]
    # For 32 case: [4, 8, 4, 8] -> [32, 32]
    # For 64 case: [8, 8, 8, 8] -> [64, 64]
    
    # So for output (h, w), we need to find the corresponding indices in the 5D input
    # h = d1 * (W2 * H2 * W2) + d2 * (H2 * W2) + d3 * W2 + d4
    # where d1 ranges over [0, H2), d2 over [0, W2), d3 over [0, H2), d4 over [0, W2)
    
    # After roll with shifts (SHIFT_H, SHIFT_W) on dims (1,2) of [1, H, W, C]:
    # rolled[h, w, c] = original[(h + SHIFT_H) % H, (w + SHIFT_W) % W, c]
    
    # rolled_h and rolled_w are the indices into the 3D tensor [H, W, C]
    # We need to map (rolled_h, rolled_w) back to the 5D input indices
    
    # For the 32 case: rolled_h = d1*8 + d3*2 + d4_offset? No...
    # Actually [4, 8, 4, 8] -> [32, 32]
    # h = d1 * (8*4*8) + d2 * (4*8) + d3 * 8 + d4? No, that gives max = 3*256 + 7*32 + 3*8 + 7 = 768+224+24+7 = 1023 ≠ 31
    
    # Let me reconsider. view(-1, H, W, C) just flattens dims 0-3 and reshapes.
    # in_3 shape: [1, H2, W2, H2, W2, C]
    # After contiguous, we have the same shape but contiguous memory.
    # view(-1, H, W, C) reshapes [1, H2, W2, H2, W2, C] -> [1, H, W, C]
    # The -1 resolves to 1 since 1*H2*W2*H2*W2*C = H*W*C
    
    # The mapping from (h, w) to (d1, d2, d3, d4):
    # h is the row in [H], w is the col in [W]
    # Since the view merges [H2, W2, H2, W2] into [H, W]:
    # The merged index h*w_stride + w = d1*(W2*H2*W2) + d2*(H2*W2) + d3*W2 + d4
    # Wait, in row-major: the flat index over dims 1-4 is:
    # flat = d1 * (W2 * H2 * W2) + d2 * (H2 * W2) + d3 * W2 + d4
    # And h * W + w = flat ... but that can't work directly since H*W = H2*W2*H2*W2
    
    # Hmm, actually the view just reinterpret the contiguous memory.
    # The contiguous memory layout for in_3 is:
    # element at (d0, d1, d2, d3, d4, c) is at offset:
    # d0 * (H2*W2*H2*W2*C) + d1*(W2*H2*W2*C) + d2*(H2*W2*C) + d3*(W2*C) + d4*C + c
    
    # After view to [1, H, W, C], element at (0, h, w, c) is at offset:
    # 0 * (H*W*C) + h*(W*C) + w*C + c
    
    # These must be equal for the same physical element.
    # So h*(W*C) + w*C + c = d1*(W2*H2*W2*C) + d2*(H2*W2*C) + d3*(W2*C) + d4*C + c
    # h*W + w = d1*(W2*H2*W2) + d2*(H2*W2) + d3*W2 + d4
    
    # So the mapping from (h, w) to (d1, d2, d3, d4) is:
    # Given h and w, compute flat = h * W + w
    # d1 = flat // (W2 * H2 * W2)
    # remainder = flat % (W2 * H2 * W2)
    # d2 = remainder // (H2 * W2)
    # remainder2 = remainder % (H2 * W2)
    # d3 = remainder2 // W2
    # d4 = remainder2 % W2
    
    # For 32 case: W=32, W2=8, H2=4
    # flat = h*32 + w
    # d1 = flat // (8*4*8) = flat // 256
    # d2 = (flat % 256) // (4*8) = (flat % 256) // 32
    # d3 = (flat % 32) // 8
    # d4 = (flat % 32) % 8
    
    # Hmm but that gives d1 up to 31*32+31=1023, 1023//256=3, so d1 in [0,3] ✓
    # d2 up to (1023%256)//32 = 255//32 = 7, so d2 in [0,7] ✓
    # d3 up to (31%32)//8... wait flat%32 for h=0,w=31 is 31, 31//8=3 ✓
    # d4 up to 7 ✓
    
    # For 64 case: W=64, W2=8, H2=8
    # flat = h*64 + w, max = 63*64+63=4095
    # d1 = flat // (8*8*8) = flat // 512, max=4095//512=7 ✓
    # d2 = (flat%512) // (8*8) = (flat%512)//64, max=511//64=7 ✓
    # d3 = (flat%64)//8, max=63//8=7 ✓
    # d4 = (flat%64)%8, max=7 ✓ 
    
    # Now, after roll with shifts (SHIFT_H, SHIFT_W) on dims (1, 2) of [1, H, W, C]:
    # The roll shifts along dim 1 (H) and dim 2 (W)
    # rolled[h, w] = original[(h+SHIFT_H)%H, (w+SHIFT_W)%W]
    
    # So we need to read from the input at position corresponding to
    # rolled_h = (h + SHIFT_H) % H, rolled_w = (w + SHIFT_W) % W
    
    # The input offset for (rolled_h, rolled_w, c) is:
    # rolled_flat = rolled_h * W + rolled_w
    # offset = rolled_flat * C + c  (since in_3 is contiguous after .contiguous())
    
    # Actually wait - in_3 is [1, H2, W2, H2, W2, C]. After contiguous(), it's the same shape but contiguous.
    # But the view to [-1, H, W, C] means the memory is already contiguous,
    # so element at logical (0, h, w, c) is at physical offset h*W*C + w*C + c.
    # And this equals the offset for the original 6D tensor at (0, d1, d2, d3, d4, c).
    
    # So for reading the rolled element, we just compute:
    # rolled_flat = rolled_h * W + rolled_w
    # base_offset = rolled_flat * C
    
    # Now for the layernorm computation:
    # For each row (h, w), we compute layernorm over the C elements
    # mean = sum(x_c) / C
    # var = sum((x_c - mean)^2) / C  
    # y_c = (x_c - mean) / sqrt(var + eps) * weight_c + bias_c
    # Then add residual: out_c = add_c + y_c
    
    # Compute offsets for the rolled input row
    rolled_flat = rolled_h * W + rolled_w
    
    # Load the rolled input row
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    in_offsets = rolled_flat * C + c_offsets
    x = tl.load(in_ptr + in_offsets, mask=c_mask, other=0.0).to(tl.float32)
    
    # Load bias and weight
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    
    # Load residual (add)
    add_offsets = row_idx * C + c_offsets
    add_val = tl.load(add_ptr + add_offsets, mask=c_mask, other=0.0).to(tl.float32)
    
    # Compute layernorm
    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    eps = 1e-05
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply weight/bias
    y = x_centered * rstd * weight + bias
    
    # Add residual
    out = add_val + y
    
    # Store output
    out_offsets = row_idx * C + c_offsets
    tl.store(out_ptr + out_offsets, out.to(in_ptr.dtype.element_ty), mask=c_mask)

@torch.fx.wrap
def fused_roll_layernorm_add(in_0, in_1, in_2, in_3):
    # in_3: [1, H2, W2, H2, W2, C]
    # Compute dimensions from input shape
    shape_3 = in_3.shape  # [1, H2, W2, H2, W2, C]
    H2 = shape_3[1]
    W2 = shape_3[2]
    C = shape_3[5]
    
    # H = H2 * W2 * H2 * W2... no
    # The view is to (-1, H, W, C) where H and W are determined by the specific pattern
    # For the 32x32 case: in_3 = [1, 4, 8, 4, 8, 768], view -> [-1, 32, 32, 768]
    # H = 32 = 4 * 8, W = 32 = 4 * 8? No, 4*8=32 but that's H2*W2
    # Actually H*W = H2*W2*H2*W2 = 4*8*4*8 = 1024, and H=W=32
    # So H = sqrt(H2*W2*H2*W2)? No, that's not general.
    
    # For the 64x64 case: in_3 = [1, 8, 8, 8, 8, 384], view -> [-1, 64, 64, 384]
    # H = 64, W = 64, H2*W2*H2*W2 = 8*8*8*8 = 4096 = 64*64 ✓
    
    # The H and W are hardcoded in the pattern, so we know them:
    # This pass is for the 32x32,768 case
    H = 32
    W = 32
    C_val = 768
    N = H * W  # 1024
    
    # Ensure input is contiguous
    in_3_cont = in_3.contiguous()
    
    BLOCK_C = 1024  # Must be >= C
    
    # Output shape: [1, N, C] = [1, 1024, 768]
    out = torch.empty((1, N, C_val), dtype=in_3.dtype, device=in_3.device)
    
    grid = (N,)
    
    fused_roll_layernorm_add_kernel[grid](
        in_ptr=in_3_cont,
        bias_ptr=in_0,
        weight_ptr=in_1,
        add_ptr=in_2,
        out_ptr=out,
        H=H, W=W, C=C_val,
        H2=H2, W2=W2,
        SHIFT_H=4, SHIFT_W=4,
        N=N,
        BLOCK_C=BLOCK_C,
    )
    
    return (out,)

def replacement_func():
    return fused_roll_layernorm_add