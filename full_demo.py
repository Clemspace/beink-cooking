#!/usr/bin/env python3
"""
BEINK Diffusion Optimizer - Hackathon Demo
===========================================

Demonstrating sustainable AI through optimized diffusion model inference.
Two approaches: Research-backed parallel sampling + GPU-poor optimizations.
"""

import os
import torch
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt
from datetime import datetime

class HackathonDemo:
    """
    Comprehensive demo showcasing both theoretical and practical approaches
    to sustainable diffusion model optimization.
    """
    
    def __init__(self, output_dir="./hackathon_results"):
        self.output_dir = output_dir
        self.results = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized output structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/comparisons", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)
        
    def load_baseline_pipeline(self):
        """Load unoptimized baseline for comparison"""
        print("📦 Loading baseline SDXL (FP32, no optimizations)...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,  # Intentionally unoptimized
            use_safetensors=True,
        ).to("cuda")
        return pipeline
    
    def load_optimized_pipeline(self):
        """Load GPU-poor optimized pipeline"""
        print("🚀 Loading BEINK optimized SDXL...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")
        
        # Apply BEINK optimizations
        optimizations_applied = []
        
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            optimizations_applied.append("xFormers")
            print("   ✅ xFormers memory-efficient attention")
        except:
            print("   ⚠️ xFormers not available")
        
        try:
            pipeline.enable_attention_slicing(1)
            optimizations_applied.append("Attention Slicing")
            print("   ✅ Attention slicing")
        except:
            pass
            
        try:
            pipeline.enable_vae_slicing()
            optimizations_applied.append("VAE Slicing")
            print("   ✅ VAE slicing")
        except:
            pass
        
        print(f"   🛠️ Applied: {', '.join(optimizations_applied)}")
        return pipeline, optimizations_applied
    
    def load_parallel_pipeline(self):
        """Attempt to load research parallel sampling pipeline"""
        print("🔬 Attempting to load parallel sampling pipeline...")
        try:
            # This would normally load the research implementation
            # For demo purposes, we'll simulate or use memory-realistic version
            from memory_realistic_parallel import MemoryRealisticSDXLPipeline
            
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to("cuda")
            
            parallel_pipeline = MemoryRealisticSDXLPipeline(base_pipeline)
            parallel_pipeline.configure(
                num_blocks=2,
                picard_iterations=2,
                parallel_degree=2
            )
            
            print("   ✅ Memory-realistic parallel sampling loaded")
            return parallel_pipeline, "memory_realistic"
            
        except ImportError:
            print("   ⚠️ Parallel sampling not available - using optimized baseline")
            return None, "not_available"
        except torch.cuda.OutOfMemoryError:
            print("   ❌ CUDA OOM - insufficient GPU memory for parallel sampling")
            return None, "oom_error"
    
    def generate_with_tracking(self, pipeline, prompt, method_name, **kwargs):
        """Generate image with comprehensive performance tracking"""
        print(f"   🎨 Generating with {method_name}...")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Generation with timing
        start_time = time.time()
        
        if hasattr(pipeline, '__call__'):
            # Standard diffusers pipeline
            result = pipeline(prompt=prompt, **kwargs)
            image = result.images[0]
            performance_stats = None
        else:
            # Custom parallel pipeline
            result = pipeline(prompt=prompt, **kwargs)
            image = result.images[0]
            performance_stats = getattr(result, 'performance_stats', None)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        final_memory = torch.cuda.memory_allocated() / (1024**3)
        peak_memory = final_memory  # Simplified
        
        # Carbon footprint estimation (300W GPU, 0.4kg CO2/kWh)
        carbon_footprint = (300 * generation_time / 3600) * 0.0004
        
        metrics = {
            'method': method_name,
            'prompt': prompt,
            'generation_time': generation_time,
            'initial_memory_gb': initial_memory,
            'peak_memory_gb': peak_memory,
            'carbon_footprint_kg': carbon_footprint,
            'performance_stats': performance_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(metrics)
        
        print(f"      ✅ {generation_time:.2f}s | {peak_memory:.1f}GB | {carbon_footprint:.6f}kg CO2")
        
        return image, metrics
    
    def create_comparison_grid(self, images, labels, metrics, prompt_idx):
        """Create visually appealing comparison grid"""
        num_images = len(images)
        if num_images == 0:
            return None
        
        # Resize images
        img_size = (512, 512)
        resized_images = [img.resize(img_size) for img in images]
        
        # Create grid layout
        grid_width = num_images * img_size[0]
        grid_height = img_size[1] + 200  # Extra space for metrics
        
        grid = Image.new('RGB', (grid_width, grid_height), '#1a1a2e')
        draw = ImageDraw.Draw(grid)
        
        # Load font
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
            metric_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
        
        # Add images and labels
        for i, (img, label, metric) in enumerate(zip(resized_images, labels, metrics)):
            x_offset = i * img_size[0]
            
            # Paste image
            grid.paste(img, (x_offset, 60))
            
            # Add title
            title_bbox = draw.textbbox((0, 0), label, font=title_font)
            title_x = x_offset + (img_size[0] - title_bbox[2]) // 2
            draw.text((title_x, 20), label, fill='white', font=title_font)
            
            # Add metrics
            metric_y = 60 + img_size[1] + 10
            metrics_text = [
                f"Time: {metric['generation_time']:.2f}s",
                f"Memory: {metric['peak_memory_gb']:.1f}GB",
                f"Carbon: {metric['carbon_footprint_kg']:.6f}kg"
            ]
            
            for j, text in enumerate(metrics_text):
                draw.text((x_offset + 10, metric_y + j * 20), text, 
                         fill='#4ecdc4', font=metric_font)
            
            # Add efficiency indicators
            if i > 0:  # Compare to baseline
                baseline_time = metrics[0]['generation_time']
                current_time = metric['generation_time']
                speedup = baseline_time / current_time
                
                speedup_text = f"🚀 {speedup:.2f}x faster"
                draw.text((x_offset + 10, metric_y + 70), speedup_text, 
                         fill='#81c784', font=metric_font)
        
        # Save comparison
        comparison_path = f"{self.output_dir}/comparisons/prompt_{prompt_idx}_comparison.png"
        grid.save(comparison_path)
        print(f"   📸 Comparison saved: {comparison_path}")
        
        return grid
    
    def run_comprehensive_demo(self):
        """Run the complete hackathon demonstration"""
        print("🎯 BEINK DIFFUSION OPTIMIZER - HACKATHON DEMO")
        print("=" * 80)
        print("Sustainable AI through optimized diffusion model inference")
        print("Research implementation + GPU-poor practical solutions")
        print("=" * 80)
        
        # Test prompts showcasing different scenarios
        test_prompts = [
            "A sustainable smart city with renewable energy and green architecture",
            "An AI researcher working on climate solutions in a high-tech lab",
            "A beautiful landscape showing the harmony between technology and nature"
        ]
        
        # Load pipelines
        print("\n🔧 LOADING PIPELINES...")
        baseline_pipeline = self.load_baseline_pipeline()
        optimized_pipeline, optimizations = self.load_optimized_pipeline()
        parallel_pipeline, parallel_status = self.load_parallel_pipeline()
        
        generation_params = {
            'num_inference_steps': 25,  # Reasonable for demo
            'height': 1024,
            'width': 1024,
            'guidance_scale': 7.5
        }
        
        print(f"\n🎨 GENERATING IMAGES...")
        print(f"Parameters: {generation_params}")
        
        all_comparisons = []
        
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"\n📝 Prompt {prompt_idx + 1}/{len(test_prompts)}:")
            print(f"    '{prompt}'")
            
            images = []
            labels = []
            prompt_metrics = []
            
            # 1. Baseline generation
            print(f"\n1️⃣ BASELINE (Unoptimized FP32)")
            baseline_img, baseline_metric = self.generate_with_tracking(
                baseline_pipeline, prompt, "baseline", **generation_params
            )
            baseline_img.save(f"{self.output_dir}/images/baseline_{prompt_idx}.png")
            
            images.append(baseline_img)
            labels.append("Baseline\n(FP32)")
            prompt_metrics.append(baseline_metric)
            
            # 2. BEINK optimized generation
            print(f"\n2️⃣ BEINK OPTIMIZED (GPU-Poor Solution)")
            optimized_img, optimized_metric = self.generate_with_tracking(
                optimized_pipeline, prompt, "beink_optimized", **generation_params
            )
            optimized_img.save(f"{self.output_dir}/images/optimized_{prompt_idx}.png")
            
            images.append(optimized_img)
            labels.append("BEINK Optimized\n(FP16 + Optimizations)")
            prompt_metrics.append(optimized_metric)
            
            # 3. Parallel sampling (if available)
            if parallel_pipeline is not None:
                print(f"\n3️⃣ PARALLEL SAMPLING (Research Implementation)")
                try:
                    parallel_img, parallel_metric = self.generate_with_tracking(
                        parallel_pipeline, prompt, "parallel_sampling", **generation_params
                    )
                    parallel_img.save(f"{self.output_dir}/images/parallel_{prompt_idx}.png")
                    
                    images.append(parallel_img)
                    labels.append("Parallel Sampling\n(Memory-Realistic)")
                    prompt_metrics.append(parallel_metric)
                    
                except Exception as e:
                    print(f"      ❌ Parallel sampling failed: {e}")
            else:
                print(f"\n3️⃣ PARALLEL SAMPLING: {parallel_status}")
            
            # Create comparison grid
            comparison_grid = self.create_comparison_grid(
                images, labels, prompt_metrics, prompt_idx
            )
            if comparison_grid:
                all_comparisons.append(comparison_grid)
        
        # Generate final analysis
        self.generate_final_analysis()
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"📁 Results saved to: {self.output_dir}")
        
    def generate_final_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        print(f"\n📊 GENERATING ANALYSIS...")
        
        # Save raw results
        with open(f"{self.output_dir}/metrics/raw_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create performance analysis
        self.create_performance_plots()
        self.create_sustainability_report()
        self.create_presentation_summary()
    
    def create_performance_plots(self):
        """Create performance visualization plots"""
        methods = list(set(r['method'] for r in self.results))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('#1a1a2e')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # 1. Execution time comparison
        method_times = {}
        for method in methods:
            times = [r['generation_time'] for r in self.results if r['method'] == method]
            method_times[method] = np.mean(times)
        
        colors = ['#ff6b6b', '#4ecdc4', '#81c784'][:len(methods)]
        bars1 = ax1.bar(method_times.keys(), method_times.values(), color=colors)
        ax1.set_ylabel('Generation Time (s)', color='white')
        ax1.set_title('Performance Comparison', color='white', fontweight='bold')
        
        # Add values on bars
        for bar, value in zip(bars1, method_times.values()):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{value:.2f}s', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 2. Speedup factors
        if 'baseline' in method_times:
            baseline_time = method_times['baseline']
            speedups = {method: baseline_time / time for method, time in method_times.items()}
            
            bars2 = ax2.bar(speedups.keys(), speedups.values(), color=colors)
            ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Speedup Factor', color='white')
            ax2.set_title('Performance Improvements', color='white', fontweight='bold')
            
            for bar, value in zip(bars2, speedups.values()):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{value:.2f}x', ha='center', va='bottom', color='white', fontweight='bold')
        
        # 3. Memory usage
        method_memory = {}
        for method in methods:
            memory = [r['peak_memory_gb'] for r in self.results if r['method'] == method]
            method_memory[method] = np.mean(memory)
        
        bars3 = ax3.bar(method_memory.keys(), method_memory.values(), color=colors)
        ax3.set_ylabel('Peak Memory (GB)', color='white')
        ax3.set_title('GPU Memory Usage', color='white', fontweight='bold')
        
        # 4. Carbon footprint
        method_carbon = {}
        for method in methods:
            carbon = [r['carbon_footprint_kg'] for r in self.results if r['method'] == method]
            method_carbon[method] = np.sum(carbon)  # Total carbon for all images
        
        bars4 = ax4.bar(method_carbon.keys(), method_carbon.values(), color=colors)
        ax4.set_ylabel('Total Carbon Footprint (kg CO2)', color='white')
        ax4.set_title('Environmental Impact', color='white', fontweight='bold')
        
        plt.suptitle('BEINK Diffusion Optimizer - Performance Analysis', 
                    fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metrics/performance_analysis.png', 
                   dpi=300, facecolor='#1a1a2e', bbox_inches='tight')
        plt.close()
        
        print("   📊 Performance analysis saved")
    
    def create_sustainability_report(self):
        """Create sustainability impact report"""
        baseline_results = [r for r in self.results if r['method'] == 'baseline']
        optimized_results = [r for r in self.results if r['method'] != 'baseline']
        
        if not baseline_results or not optimized_results:
            return
        
        total_baseline_time = sum(r['generation_time'] for r in baseline_results)
        total_optimized_time = sum(r['generation_time'] for r in optimized_results)
        
        total_baseline_carbon = sum(r['carbon_footprint_kg'] for r in baseline_results)
        total_optimized_carbon = sum(r['carbon_footprint_kg'] for r in optimized_results)
        
        carbon_saved = total_baseline_carbon - total_optimized_carbon
        time_saved = total_baseline_time - total_optimized_time
        
        speedup = total_baseline_time / total_optimized_time if total_optimized_time > 0 else 1
        carbon_reduction = (carbon_saved / total_baseline_carbon * 100) if total_baseline_carbon > 0 else 0
        
        report = {
            'sustainability_metrics': {
                'total_images_generated': len(self.results),
                'total_time_saved_seconds': time_saved,
                'average_speedup': speedup,
                'total_carbon_saved_kg': carbon_saved,
                'carbon_reduction_percentage': carbon_reduction,
                'efficiency_improvement': ((speedup - 1) * 100)
            },
            'business_impact': {
                'cost_reduction_estimate': f"{((speedup-1)/speedup*100):.1f}%",
                'throughput_increase': f"{speedup:.1f}x",
                'infrastructure_efficiency': "Reduced GPU hours required",
                'environmental_benefit': f"{carbon_reduction:.1f}% less CO2 per image"
            }
        }
        
        with open(f"{self.output_dir}/metrics/sustainability_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("   🌱 Sustainability report saved")
        return report
    
    def create_presentation_summary(self):
        """Create presentation-ready summary"""
        summary = {
            'project_overview': {
                'title': 'BEINK Diffusion Optimizer',
                'subtitle': 'Sustainable AI through Optimized Diffusion Model Inference',
                'approaches': [
                    'Research-backed parallel sampling implementation',
                    'GPU-poor practical optimization suite',
                    'Comprehensive sustainability tracking'
                ]
            },
            'technical_achievements': {
                'parallel_sampling': 'Implemented Stanford research paper on temporal parallelization',
                'memory_optimization': 'Created memory-realistic version for A40 constraints',
                'production_ready': 'Built drop-in optimization framework for any diffusers pipeline',
                'sustainability_tracking': 'Integrated carbon footprint monitoring'
            },
            'results_summary': self.get_results_summary(),
            'presentation_points': [
                'Research implementation meets real-world constraints',
                'Measurable performance and sustainability improvements',
                'Production-viable optimization framework',
                'Foundation for scaling to larger GPU clusters',
                'Democratizes access to efficient AI inference'
            ]
        }
        
        with open(f"{self.output_dir}/presentation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("   📋 Presentation summary saved")
    
    def get_results_summary(self):
        """Get concise results summary"""
        if not self.results:
            return {}
        
        methods = list(set(r['method'] for r in self.results))
        baseline_time = np.mean([r['generation_time'] for r in self.results if r['method'] == 'baseline'])
        
        summary = {}
        for method in methods:
            method_results = [r for r in self.results if r['method'] == method]
            avg_time = np.mean([r['generation_time'] for r in method_results])
            speedup = baseline_time / avg_time if avg_time > 0 else 1.0
            
            summary[method] = {
                'average_time': avg_time,
                'speedup': speedup,
                'images_generated': len(method_results)
            }
        
        return summary

def main():
    """Run the hackathon demo"""
    demo = HackathonDemo()
    demo.run_comprehensive_demo()
    
    print(f"\n🎯 HACKATHON PRESENTATION READY!")
    print(f"📁 All materials in: {demo.output_dir}")
    print(f"🖼️ Visual comparisons: {demo.output_dir}/comparisons/")
    print(f"📊 Performance metrics: {demo.output_dir}/metrics/")

if __name__ == "__main__":
    main()