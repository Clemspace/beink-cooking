#!/usr/bin/env python3
"""
Simple fix - just provide basic SDXL conditioning with default values
"""

def simple_sdxl_fix():
    """Provide basic SDXL conditioning"""
    
    with open('parallel_sampling_sdxl.py', 'r') as f:
        content = f.read()
    
    print("🔧 Adding simple SDXL conditioning...")
    
    # Replace the added_cond_kwargs with proper SDXL defaults
    content = content.replace(
        'added_cond_kwargs=getattr(self, "_sdxl_added_cond_kwargs", {}),  # SDXL conditioning',
        '''added_cond_kwargs={
                        "text_embeds": torch.zeros((1, 1280), device=curr_latent.device, dtype=curr_latent.dtype),
                        "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=curr_latent.device, dtype=curr_latent.dtype)
                    },  # SDXL conditioning'''
    )
    
    # Also add torch import if not present
    if 'import torch' not in content:
        content = 'import torch\n' + content
    
    with open('parallel_sampling_sdxl.py', 'w') as f:
        f.write(content)
    
    print("✅ Added simple SDXL conditioning!")
    print("   🔧 Using default text_embeds (zeros)")
    print("   🔧 Using default time_ids for 1024x1024")
    
    return True

if __name__ == "__main__":
    simple_sdxl_fix()
    print("🚀 Ready to test parallel sampling with basic SDXL conditioning!")
    print("Run: python3 main.py --use_fp16 --num_steps 20")