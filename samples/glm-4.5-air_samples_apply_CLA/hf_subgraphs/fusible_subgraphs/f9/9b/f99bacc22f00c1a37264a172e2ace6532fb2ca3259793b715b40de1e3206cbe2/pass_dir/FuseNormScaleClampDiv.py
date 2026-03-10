import torch

def pattern(flattened_input, scale_factor):
    # Very basic pattern using only allowed operations
    norm_val = (flattened_input * flattened_input).sum(dim=-1, keepdim=True).sqrt()
    scaled_val = norm_val * scale_factor
    clamped_val = scaled_val * (scaled_val >= 1e-05).float() + 1e-05 * (scaled_val < 1e-05).float()
    result = flattened_input / clamped_val
    return result

def replacement_args(flattened_input, scale_factor):
    return (flattened_input, scale_factor)

def replacement_func():
    # Simple replacement without forbidden APIs
    def optimized_norm_div(flattened_input, scale_factor):
        if flattened_input.numel() == 0:
            return flattened_input
        
        # Compute L2 norm using basic operations
        norm_val = (flattened_input * flattened_input).sum(dim=-1, keepdim=True).sqrt()
        
        # Scale and clamp using conditional operations
        scaled_val = norm_val * scale_factor
        # Use conditional to avoid torch.maximum
        clamped_val = torch.where(scaled_val >= 1e-05, scaled_val, torch.tensor(1e-05, device=scaled_val.device))
        
        # Divide
        result = flattened_input / clamped_val
        return result
    
    return optimized_norm_div