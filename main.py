# main.py
import os
import torch
import argparse
import time
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# Import our custom modules
#from carbon_tracking import CarbonTracker
from simple_carbon_tracking import SimpleCarbonTracker as CarbonTracker


# Import lightweight SDXL UNet (assuming you've placed it in your project directory)
import sys
sys.path.append(".")
# Try to import the lightweight SDXL UNet implementation
try:
    from sdxl_rewrite import UNet2DConditionModel as LightweightUNet
    LIGHTWEIGHT_UNET_AVAILABLE = True
    print("✅ Lightweight SDXL UNet implementation found!")
except ImportError:
    LIGHTWEIGHT_UNET_AVAILABLE = False
    print("⚠️ Lightweight SDXL UNet implementation not found. Using standard diffusers UNet.")

# Import our parallel sampling implementation
from parallel_sampling_sdxl import ParallelSDXLPipeline


def load_model(model_id, use_lightweight=True, torch_dtype=torch.float16):
    """
    Load SDXL model with optional lightweight UNet
    
    Args:
        model_id: HuggingFace model ID
        use_lightweight: Whether to use the lightweight UNet implementation
        torch_dtype: Torch data type to use
        
    Returns:
        Loaded pipeline
    """
    print(f"Loading model: {model_id}")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    
    if use_lightweight and LIGHTWEIGHT_UNET_AVAILABLE:
        print("Switching to lightweight UNet implementation...")
        lightweight_unet = LightweightUNet().cuda().to(torch_dtype)
        
        # Load weights from original UNet
        lightweight_unet.load_state_dict(pipeline.unet.state_dict())
        
        # Replace UNet in pipeline
        pipeline.unet = lightweight_unet
        
    return pipeline


def run_benchmarks(args):
    """
    Run benchmarks comparing standard vs parallel sampling
    
    Args:
        args: Command-line arguments
    """
    # Initialize carbon tracker
    tracker = CarbonTracker(
        project_name=args.project_name,
        output_dir=args.output_dir
    )
    
    # Load model
    pipeline = load_model(
        args.model_id,
        use_lightweight=args.use_lightweight,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32
    )
    
    # Apply optimizations
    if args.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled")
        except:
            print("⚠️ xFormers not available")
    
    if args.compile_unet and torch.__version__ >= "2.0.0":
        try:
            pipeline.unet = torch.compile(
                pipeline.unet, 
                mode="reduce-overhead", 
                fullgraph=True
            )
            print("✅ torch.compile applied to UNet")
        except:
            print("⚠️ Failed to compile UNet")
    
    # Enable model CPU offloading if requested
    if args.enable_offload:
        pipeline.enable_model_cpu_offload()
        print("✅ Model CPU offloading enabled")
    
    # Create parallel pipeline
    parallel_pipeline = ParallelSDXLPipeline(pipeline)
    parallel_pipeline.configure(
        num_blocks=args.num_blocks,
        picard_iterations=args.picard_iterations
    )
    
    # Define test prompts
    if args.prompts:
        test_prompts = args.prompts
    else:
        test_prompts = [
            "A serene landscape with mountains and a lake at sunset",
            "A futuristic city with flying cars and tall skyscrapers",
            "A portrait of a woman with long hair in renaissance style",
            "An astronaut riding a horse on Mars, highly detailed",
            "A macro photograph of a snowflake on a dark background"
        ]
    
    # Run standard inference
    @tracker.track_inference("standard")
    def run_standard_inference(prompt):
        return pipeline(
            prompt=prompt, 
            num_inference_steps=args.num_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale
        ).images[0]
    
    # Run parallel inference
    @tracker.track_inference("parallel")
    def run_parallel_inference(prompt):
        return parallel_pipeline(
            prompt=prompt,
            num_inference_steps=args.num_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale
        ).images[0]
    
    # Create output directory for images
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Run benchmarks
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt}")
        
        # Run standard inference
        print("\nRunning standard inference...")
        standard_image = run_standard_inference(prompt)
        standard_image.save(os.path.join(images_dir, f"standard_{i}.png"))
        
        # Run parallel inference
        print("\nRunning parallel inference...")
        parallel_image = run_parallel_inference(prompt)
        parallel_image.save(os.path.join(images_dir, f"parallel_{i}.png"))
        
        # Create comparison image
        comparison = Image.new('RGB', (standard_image.width * 2, standard_image.height))
        comparison.paste(standard_image, (0, 0))
        comparison.paste(parallel_image, (standard_image.width, 0))
        comparison.save(os.path.join(images_dir, f"comparison_{i}.png"))
    
    # Generate report and plots
    tracker.save_results()
    tracker.generate_report()
    tracker.plot_comparison()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="SDXL Optimization with Parallel Sampling")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="HuggingFace model ID")
    parser.add_argument("--use_lightweight", action="store_true", 
                        help="Use lightweight UNet implementation")
    
    # Sampling configuration
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of parallel sampling blocks")
    parser.add_argument("--picard_iterations", type=int, default=3,
                        help="Number of Picard iterations per block")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    
    # Image configuration
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    
    # Optimization configuration
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use FP16 precision")
    parser.add_argument("--enable_xformers", action="store_true",
                        help="Enable xFormers memory efficient attention")
    parser.add_argument("--compile_unet", action="store_true",
                        help="Apply torch.compile to UNet (requires PyTorch 2.0+)")
    parser.add_argument("--enable_offload", action="store_true",
                        help="Enable model CPU offloading")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for output files")
    parser.add_argument("--project_name", type=str, default="sdxl-parallel-sampling",
                        help="Project name for tracking")
    
    # Test prompts
    parser.add_argument("--prompts", type=str, nargs="+",
                        help="Test prompts to use for benchmarking")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(args)

    print("\n== 🎉 Benchmarks Complete! ==")
    print(f"Results saved to {args.output_dir}")
    print(f"Check {args.output_dir}/images for generated images")
    print(f"Check {args.output_dir}/{args.project_name}_report.md for performance report")