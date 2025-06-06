# main.py
import os
import torch
import argparse
import time
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# Import our enhanced carbon tracking
from carbon_tracking import EnhancedCarbonTracker

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

# Import our parallel sampling implementation (use the fixed version)
#from parallel_sampling_sdxl import ParallelSDXLPipeline
from savvy_sampling import ParallelSDXLPipeline



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
    # Initialize enhanced carbon tracker
    tracker = EnhancedCarbonTracker(
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
    
    # Define test prompts - curated for good visual comparison
    if args.prompts:
        test_prompts = args.prompts
    else:
        test_prompts = [
            "A majestic mountain landscape at golden hour with crystal clear lake reflections",
            "Portrait of a cyberpunk character with neon lights and detailed facial features",
            "A fantastical dragon soaring above medieval castle towers, highly detailed",
            "Macro photography of a dewdrop on a flower petal, bokeh background",
            "An astronaut floating in space with Earth visible in the background, photorealistic"
        ]
    
    # Limit to requested number of test images
    if args.max_images:
        test_prompts = test_prompts[:args.max_images]
    
    print(f"\n🚀 Starting benchmark with {len(test_prompts)} test prompts")
    print(f"📋 Configuration:")
    print(f"   • Steps: {args.num_steps}")
    print(f"   • Blocks: {args.num_blocks}")
    print(f"   • Picard iterations: {args.picard_iterations}")
    print(f"   • Guidance scale: {args.guidance_scale}")
    print(f"   • Resolution: {args.width}x{args.height}")
    
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
        )
    
    # Create output directory for images
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Run benchmarks
    print(f"\n📸 Generating images...")
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"🎨 Prompt {i+1}/{len(test_prompts)}")
        print(f"💬 {prompt}")
        print(f"{'='*60}")
        
        # Run standard inference
        print("\n🔄 Running standard inference...")
        try:
            standard_image = run_standard_inference(prompt)
            standard_image.save(os.path.join(images_dir, f"standard_{i}.png"))
            print("✅ Standard generation complete")
        except Exception as e:
            print(f"❌ Standard generation failed: {e}")
            continue
        
        # Small delay to separate the inference runs
        time.sleep(1)
        
        # Run parallel inference
        print("\n⚡ Running parallel inference...")
        try:
            parallel_image = run_parallel_inference(prompt)
            parallel_image.save(os.path.join(images_dir, f"parallel_{i}.png"))
            print("✅ Parallel generation complete")
        except Exception as e:
            print(f"❌ Parallel generation failed: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("📊 Generating analysis and comparisons...")
    print(f"{'='*60}")
    
    # Generate all visual comparisons and analysis
    try:
        # Save raw results
        tracker.save_results()
        
        # Generate comprehensive report
        tracker.generate_report()
        
        # Create all visual comparisons
        comparison_results = tracker.generate_all_comparisons()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"\n📋 Generated files:")
        print(f"   • Individual images: {images_dir}/")
        if comparison_results.get("individual_comparisons"):
            print(f"   • Side-by-side comparisons: {len(comparison_results['individual_comparisons'])} files")
        if comparison_results.get("dashboard"):
            print(f"   • Performance dashboard: {comparison_results['dashboard']}")
        if comparison_results.get("grid"):
            print(f"   • Comparison grid: {comparison_results['grid']}")
        
        # Print summary for presentation
        print(f"\n🎯 QUICK PRESENTATION SUMMARY:")
        if tracker.results["standard"] and tracker.results["parallel"]:
            std_avg_time = sum(r["execution_time"] for r in tracker.results["standard"]) / len(tracker.results["standard"])
            par_avg_time = sum(r["execution_time"] for r in tracker.results["parallel"]) / len(tracker.results["parallel"])
            time_improvement = (std_avg_time - par_avg_time) / std_avg_time * 100
            
            std_avg_co2 = sum(r["emissions"] for r in tracker.results["standard"]) / len(tracker.results["standard"])
            par_avg_co2 = sum(r["emissions"] for r in tracker.results["parallel"]) / len(tracker.results["parallel"])
            co2_improvement = (std_avg_co2 - par_avg_co2) / std_avg_co2 * 100
            
            print(f"   • Average speedup: {time_improvement:+.1f}%")
            print(f"   • Carbon reduction: {co2_improvement:+.1f}%")
            print(f"   • Images generated: {len(tracker.results['standard'])} pairs")
            
    except Exception as e:
        print(f"❌ Analysis generation failed: {e}")
        import traceback
        traceback.print_exc()


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
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of test images to generate")
    
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
    
    print("🚀 SDXL Parallel Sampling Benchmark")
    print("=" * 50)
    
    try:
        run_benchmarks(args)
        print(f"\n🎉 Benchmark Complete!")
        print(f"📁 Check {args.output_dir} for all results")
        print(f"📊 Dashboard: {args.output_dir}/{args.project_name}_dashboard.png")
        print(f"📝 Report: {args.output_dir}/{args.project_name}_report.md")
        print(f"🖼️  Comparisons: {args.output_dir}/comparisons/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()