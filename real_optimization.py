#!/usr/bin/env python3
# Real optimization demo with multiple working optimizations
import os
import torch
import argparse
import time
import json
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import psutil
import GPUtil

class RealOptimizationTracker:
    """Performance tracker with real optimizations"""
    
    def __init__(self, output_dir="./real_optimization_results"):
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    def track_generation(self, method_name, prompt, func, *args, **kwargs):
        """Track a single image generation"""
        print(f"   ⏱️  Starting {method_name}...")
        
        # Get initial GPU memory
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        initial_gpu_mem = gpu.memoryUsed if gpu else 0
        
        # Track generation
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - start_time
        final_gpu_mem = gpu.memoryUsed if gpu else 0
        gpu_memory_used = final_gpu_mem
        
        # Estimate carbon
        estimated_carbon = (300 * execution_time / 3600) * 0.0004
        
        metrics = {
            "method": method_name,
            "prompt": prompt,
            "execution_time": execution_time,
            "gpu_memory_mb": gpu_memory_used,
            "estimated_carbon_kg": estimated_carbon,
            "timestamp": time.time()
        }
        
        self.results.append(metrics)
        
        print(f"   ✅ {method_name}: {execution_time:.2f}s, {gpu_memory_used:.0f}MB GPU, {estimated_carbon:.6f}kg CO2")
        
        return result
    
    def save_results(self):
        with open(os.path.join(self.output_dir, "real_benchmark_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
    
    def print_comparison(self):
        if len(self.results) < 2:
            return
            
        # Group results by method
        methods = {}
        for result in self.results:
            method = result["method"]
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        print("\n" + "="*70)
        print("🎯 REAL OPTIMIZATION COMPARISON")
        print("="*70)
        
        for method_name, results in methods.items():
            avg_time = sum(r["execution_time"] for r in results) / len(results)
            total_carbon = sum(r["estimated_carbon_kg"] for r in results)
            avg_memory = sum(r["gpu_memory_mb"] for r in results) / len(results)
            
            print(f"\n📊 {method_name.upper()}:")
            print(f"   ⚡ Average time: {avg_time:.2f} seconds/image")
            print(f"   🌱 Total carbon: {total_carbon:.6f} kg CO2eq")
            print(f"   💾 GPU memory: {avg_memory:.0f} MB")
        
        # Calculate improvement
        if len(methods) == 2:
            method_names = list(methods.keys())
            baseline_results = methods[method_names[0]]
            optimized_results = methods[method_names[1]]
            
            baseline_avg = sum(r["execution_time"] for r in baseline_results) / len(baseline_results)
            optimized_avg = sum(r["execution_time"] for r in optimized_results) / len(optimized_results)
            
            if baseline_avg > optimized_avg:
                speedup = baseline_avg / optimized_avg
                improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
                print(f"\n🚀 REAL PERFORMANCE GAIN:")
                print(f"   📈 Speedup: {speedup:.2f}x")
                print(f"   💡 Time reduction: {improvement:.1f}%")
                print(f"   ⏱️  Time saved: {baseline_avg - optimized_avg:.2f}s per image")

def load_baseline_model(model_id):
    """Load baseline SDXL model"""
    print("🔄 Loading BASELINE model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use FP32 for baseline
        use_safetensors=True,
    ).to("cuda")
    print("   📋 Baseline: FP32, no optimizations")
    return pipeline

def load_optimized_model(model_id):
    """Load optimized SDXL model with working optimizations"""
    print("🔄 Loading OPTIMIZED model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # FP16 optimization
        use_safetensors=True,
        variant="fp16",  # Use FP16 variant
    ).to("cuda")
    
    optimizations = []
    
    # Try xFormers
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        optimizations.append("xFormers")
        print("   ✅ xFormers enabled")
    except Exception as e:
        print(f"   ⚠️ xFormers failed: {e}")
    
    # Skip torch.compile (can hang on complex models)
    print("   ⚠️ torch.compile skipped (can hang on SDXL)")
    
    # Enable memory efficient attention (always works)
    try:
        pipeline.enable_attention_slicing(1)
        optimizations.append("attention_slicing")
        print("   ✅ Attention slicing enabled")
    except:
        pass
    
    # VAE slicing for memory efficiency
    try:
        pipeline.enable_vae_slicing()
        optimizations.append("vae_slicing")
        print("   ✅ VAE slicing enabled")
    except:
        pass
    
    print(f"   🛠️  Applied optimizations: {', '.join(optimizations) if optimizations else 'FP16 only'}")
    return pipeline, optimizations

def run_real_comparison(args):
    """Run real optimization comparison"""
    
    tracker = RealOptimizationTracker(args.output_dir)
    
    prompts = args.prompts or [
        "A futuristic city with flying cars at sunset",
        "A magical forest with glowing mushrooms",
        "An astronaut riding a horse on Mars"
    ]
    
    print(f"🎯 REAL OPTIMIZATION COMPARISON")
    print(f"📝 Testing {len(prompts)} prompts with {args.num_steps} steps")
    print(f"🖼️  Output resolution: {args.width}x{args.height}")
    
    # Load both models
    baseline_pipeline = load_baseline_model(args.model_id)
    optimized_pipeline, applied_opts = load_optimized_model(args.model_id)
    
    print(f"\n🚀 Starting real performance comparison...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n🖼️  Image {i+1}/{len(prompts)}: '{prompt}'")
        
        # Baseline generation (FP32)
        baseline_image = tracker.track_generation(
            "baseline_fp32", prompt,
            lambda p: baseline_pipeline(
                prompt=p,
                num_inference_steps=args.num_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale
            ).images[0],
            prompt
        )
        baseline_image.save(os.path.join(tracker.output_dir, "images", f"baseline_{i}.png"))
        
        # Optimized generation (FP16 + optimizations)
        optimized_image = tracker.track_generation(
            "optimized_fp16", prompt,
            lambda p: optimized_pipeline(
                prompt=p,
                num_inference_steps=args.num_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale
            ).images[0],
            prompt
        )
        optimized_image.save(os.path.join(tracker.output_dir, "images", f"optimized_{i}.png"))
        
        # Create comparison
        comparison = Image.new('RGB', (baseline_image.width * 2, baseline_image.height))
        comparison.paste(baseline_image, (0, 0))
        comparison.paste(optimized_image, (baseline_image.width, 0))
        comparison.save(os.path.join(tracker.output_dir, "images", f"comparison_{i}.png"))
    
    # Save and display results
    tracker.save_results()
    tracker.print_comparison()
    
    print(f"\n📁 All results saved to: {args.output_dir}")
    print("🎉 Real optimization comparison complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Real SDXL Optimization Comparison")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", default="./real_optimization_results")
    parser.add_argument("--prompts", nargs="+")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_real_comparison(args)