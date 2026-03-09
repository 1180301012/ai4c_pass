import torch
import triton
import triton.language as tl

# Pattern matching function for division + type conversion + multiplication fusion
def pattern(tmp_0, in_4, in_3):
    # tmp_3 = in_4 / in_3
    tmp_3 = in_4 / in_3
    tmp_4 = tmp_3.to(torch.float32)
    # tmp_5 = tmp_0.unsqueeze(-1)
    tmp_5 = tmp_0.unsqueeze(-1)
    # tmp_6 = tmp_4 * tmp_5
    tmp_6 = tmp_4 * tmp_5
    # tmp_7 = tmp_6.to(torch.float32)
    tmp_7 = tmp_6.to(torch.float32)
    return tmp_7

# Argument extraction function
def replacement_args(tmp_0, in_4, in_3):
    return (tmp_0, in_4, in_3)

# Optimized kernel fused operation
@triton.jit
def fused_div_mul_kernel(
    x_ptr,           # tmp_0 (attention mask)
    y_ptr,           # in_4 (dividend) 
    z_ptr,           # in_3 (divisor)
    out_ptr,         # output result
    batch,           # batch dimension
    seq,             # seq dimension  
    features,        # feature dimension (320)
    BLOCK_SIZE_N: tl.constexpr,
):
    # Create program IDs for 2D grid (batch, seq) and process features in blocks
    m = tl.program_id(0)
    n = tl.program_id(1)
    p = tl.program_id(2)
    
    # Calculate start index for this program
    x_offset = m * seq + n  # attention mask is [batch, seq]
    z_offset = m * seq + n  # divisor is [batch, seq, 1] - load the scalar
    y_offset = m * seq * features + n * features  # start of this batch,seq slice
    out_offset = m * seq * features + n * features  # start of output for this batch,seq
    
    # Create offsets within the feature block
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < features
    
    # Load attention mask value (broadcast to all features)
    x = tl.load(x_ptr + x_offset)
    
    # Load divisor value (should be [batch, seq, 1])
    z = tl.load(z_ptr + z_offset)
    
    # Load feature data for this block
    y_ptr_block = y_ptr + y_offset + p * BLOCK_SIZE_N
    y = tl.load(y_ptr_block + offsets, mask=mask, other=0.0)
    
    # Calculate output pointer
    out_ptr_block = out_ptr + out_offset + p * BLOCK_SIZE_N
    
    # Perform fused operations: y / z * x.unsqueeze(-1)
    # Handle division by zero
    z_safe = tl.where(z == 0.0, 1.0, z)
    result = (y / z_safe) * x
    
    # Store result
    tl.store(out_ptr_block + offsets, result, mask=mask)

@torch.fx.wrap
def fused_div_mul_triton(tmp_0, in_4, in_3):
    # Get tensor shapes from weight meta analysis
    # tmp_0: attention mask [batch, seq]
    # in_4: tensor [batch, seq, features] 
    # in_3: divisor [batch, seq, 1]
    
    batch, seq = tmp_0.shape
    features = in_4.shape[2]
    
    # Allocate output
    output = torch.empty((batch, seq, features), dtype=torch.float32, device=tmp_0.device)
    
    # Set up grid: (batch, seq, num_feature_blocks)
    BLOCK_SIZE_N = 64  # features per block
    num_blocks_features = (features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (batch, seq, num_blocks_features)
    
    fused_div_mul_kernel[grid](
        tmp_0,
        in_4,
        in_3,
        output,
        batch,
        seq,
        features,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_div_mul_triton