# enhanced_carbon_tracking.py
import time
import os
import json
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import torch
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

class EnhancedCarbonTracker:
    """
    Enhanced utility class to track carbon emissions, execution time, memory usage,
    and generate visual comparisons for different sampling methods.
    """
    
    def __init__(self, project_name="sdxl-optimization", output_dir="./results"):
        self.project_name = project_name
        self.output_dir = output_dir
        self.results = {
            "standard": [],
            "parallel": [],
        }
        self.generated_images = {
            "standard": [],
            "parallel": [],
        }
        self.prompts_used = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
        
        # Initialize CodeCarbon tracker
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            log_level="warning",
            save_to_file=True,
        )
    
    def track_inference(self, method_name):
        """
        Decorator to track carbon emissions, execution time, memory usage,
        and store generated images for comparison.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start GPU memory tracking
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                # Start carbon tracking
                self.tracker.start()
                
                # Record start time
                start_time = time.time()
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Record end time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Stop carbon tracking
                emissions = self.tracker.stop()
                
                # Get peak memory usage
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - start_memory
                
                # Store the generated image and prompt
                if hasattr(result, 'save') or isinstance(result, Image.Image):
                    # If result is a PIL Image
                    image = result
                elif hasattr(result, '__iter__') and len(result) > 0:
                    # If result is a list of images
                    image = result[0]
                else:
                    image = None
                
                if image:
                    self.generated_images[method_name].append(image)
                    # Extract prompt from kwargs or args
                    prompt = kwargs.get('prompt', args[0] if args else "Unknown prompt")
                    if len(self.prompts_used) <= len(self.generated_images[method_name]) - 1:
                        self.prompts_used.append(prompt)
                
                # Record metrics
                metrics = {
                    "execution_time": execution_time,
                    "emissions": emissions,
                    "memory_used": memory_used,
                    "timestamp": time.time(),
                    "parameters": kwargs,
                    "image_index": len(self.generated_images[method_name]) - 1,
                }
                
                self.results[method_name].append(metrics)
                
                print(f"Method: {method_name}")
                print(f"  Execution Time: {execution_time:.2f} s")
                print(f"  Carbon Emissions: {emissions:.6f} kg CO2eq")
                print(f"  Memory Used: {memory_used / 1024**2:.2f} MB")
                
                return result
            
            return wrapper
        
        return decorator
    
    def create_side_by_side_comparison(self, image_index=None, save_individual=True):
        """
        Create side-by-side comparisons of generated images.
        
        Args:
            image_index: Specific image index to compare, or None for all
            save_individual: Whether to save individual comparison images
            
        Returns:
            List of comparison image paths
        """
        comparison_paths = []
        
        if image_index is not None:
            indices = [image_index]
        else:
            # Compare all available image pairs
            max_images = min(len(self.generated_images["standard"]), 
                           len(self.generated_images["parallel"]))
            indices = range(max_images)
        
        for idx in indices:
            if (idx >= len(self.generated_images["standard"]) or 
                idx >= len(self.generated_images["parallel"])):
                continue
                
            standard_img = self.generated_images["standard"][idx]
            parallel_img = self.generated_images["parallel"][idx]
            
            # Create comparison image
            comparison = self._create_comparison_image(
                standard_img, parallel_img, 
                prompt=self.prompts_used[idx] if idx < len(self.prompts_used) else f"Image {idx}",
                image_index=idx
            )
            
            # Save comparison
            if save_individual:
                comparison_path = os.path.join(
                    self.output_dir, "comparisons", f"comparison_{idx}.png"
                )
                comparison.save(comparison_path)
                comparison_paths.append(comparison_path)
                print(f"Comparison {idx} saved to {comparison_path}")
        
        return comparison_paths
    
    def _create_comparison_image(self, img1, img2, prompt="", image_index=0):
        """Create a side-by-side comparison with labels and metrics."""
        
        # Resize images to same height if needed
        target_height = min(img1.height, img2.height, 800)  # Max height for presentation
        
        img1_resized = img1.resize((int(img1.width * target_height / img1.height), target_height))
        img2_resized = img2.resize((int(img2.width * target_height / img2.height), target_height))
        
        # Get metrics for this image
        std_metrics = self.results["standard"][image_index] if image_index < len(self.results["standard"]) else {}
        par_metrics = self.results["parallel"][image_index] if image_index < len(self.results["parallel"]) else {}
        
        # Create comparison canvas
        padding = 20
        header_height = 120
        footer_height = 150
        
        total_width = img1_resized.width + img2_resized.width + padding * 3
        total_height = target_height + header_height + footer_height + padding * 2
        
        comparison = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(comparison)
        
        # Try to load a font (fallback to default if not available)
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 18)
            metric_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
        
        # Draw title
        title = f"Image {image_index + 1}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
        draw.text((padding, padding), title, fill='black', font=title_font)
        
        # Paste images
        y_offset = header_height + padding
        comparison.paste(img1_resized, (padding, y_offset))
        comparison.paste(img2_resized, (padding * 2 + img1_resized.width, y_offset))
        
        # Draw labels
        draw.text((padding + img1_resized.width // 2 - 30, y_offset - 30), 
                 "Standard", fill='blue', font=label_font)
        draw.text((padding * 2 + img1_resized.width + img2_resized.width // 2 - 30, y_offset - 30), 
                 "Parallel", fill='green', font=label_font)
        
        # Draw metrics
        metrics_y = y_offset + target_height + padding
        
        if std_metrics and par_metrics:
            std_time = std_metrics.get('execution_time', 0)
            par_time = par_metrics.get('execution_time', 0)
            std_emissions = std_metrics.get('emissions', 0)
            par_emissions = par_metrics.get('emissions', 0)
            std_memory = std_metrics.get('memory_used', 0) / 1024**2
            par_memory = par_metrics.get('memory_used', 0) / 1024**2
            
            time_improvement = ((std_time - par_time) / std_time * 100) if std_time > 0 else 0
            emissions_improvement = ((std_emissions - par_emissions) / std_emissions * 100) if std_emissions > 0 else 0
            
            # Standard metrics
            draw.text((padding, metrics_y), 
                     f"Time: {std_time:.2f}s", fill='blue', font=metric_font)
            draw.text((padding, metrics_y + 20), 
                     f"CO2: {std_emissions:.6f}kg", fill='blue', font=metric_font)
            draw.text((padding, metrics_y + 40), 
                     f"Memory: {std_memory:.1f}MB", fill='blue', font=metric_font)
            
            # Parallel metrics
            draw.text((padding * 2 + img1_resized.width, metrics_y), 
                     f"Time: {par_time:.2f}s", fill='green', font=metric_font)
            draw.text((padding * 2 + img1_resized.width, metrics_y + 20), 
                     f"CO2: {par_emissions:.6f}kg", fill='green', font=metric_font)
            draw.text((padding * 2 + img1_resized.width, metrics_y + 40), 
                     f"Memory: {par_memory:.1f}MB", fill='green', font=metric_font)
            
            # Improvements
            improvement_x = total_width // 2 - 100
            draw.text((improvement_x, metrics_y), 
                     f"⚡ Time: {time_improvement:+.1f}%", 
                     fill='red' if time_improvement < 0 else 'darkgreen', font=metric_font)
            draw.text((improvement_x, metrics_y + 20), 
                     f"🌱 Carbon: {emissions_improvement:+.1f}%", 
                     fill='red' if emissions_improvement < 0 else 'darkgreen', font=metric_font)
        
        return comparison
    
    def create_presentation_dashboard(self):
        """Create a comprehensive dashboard for presentation."""
        
        # Calculate summary metrics
        if not self.results["standard"] or not self.results["parallel"]:
            print("Not enough data to create dashboard")
            return None
            
        avg_metrics = {}
        for method in ["standard", "parallel"]:
            avg_metrics[method] = {
                "execution_time": np.mean([r["execution_time"] for r in self.results[method]]),
                "emissions": np.mean([r["emissions"] for r in self.results[method]]),
                "memory_used": np.mean([r["memory_used"] for r in self.results[method]]) / 1024**2,
            }
        
        # Calculate improvements
        std = avg_metrics["standard"]
        par = avg_metrics["parallel"]
        improvements = {
            "execution_time": (std["execution_time"] - par["execution_time"]) / std["execution_time"] * 100,
            "emissions": (std["emissions"] - par["emissions"]) / std["emissions"] * 100,
            "memory_used": (std["memory_used"] - par["memory_used"]) / std["memory_used"] * 100,
        }
        
        # Create dashboard figure
        fig = plt.figure(figsize=(16, 10))
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'{self.project_name.upper()}: Performance Analysis Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # 1. Performance bars
        ax1 = fig.add_subplot(gs[0, :2])
        methods = ['Standard', 'Parallel']
        metrics_data = {
            'Execution Time (s)': [std["execution_time"], par["execution_time"]],
            'CO2 Emissions (kg)': [std["emissions"] * 1000, par["emissions"] * 1000],  # Convert to mg for visibility
            'Memory Usage (MB)': [std["memory_used"], par["memory_used"]]
        }
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            offset = (i - 1) * width
            bars = ax1.bar(x + offset, values, width, label=metric)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}' if val < 1 else f'{val:.1f}',
                        ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Method')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        
        # 2. Improvement percentages
        ax2 = fig.add_subplot(gs[0, 2:])
        improvement_names = ['Time', 'CO2', 'Memory']
        improvement_values = [improvements[k] for k in improvements.keys()]
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        
        bars = ax2.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Improvement with Parallel Sampling (%)')
        ax2.set_ylabel('Improvement (%)')
        
        # Add value labels
        for bar, val in zip(bars, improvement_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:+.1f}%', ha='center', 
                    va='bottom' if val > 0 else 'top', fontweight='bold')
        
        # 3. Time series plot
        if len(self.results["standard"]) > 1:
            ax3 = fig.add_subplot(gs[1, :2])
            
            std_times = [r["execution_time"] for r in self.results["standard"]]
            par_times = [r["execution_time"] for r in self.results["parallel"]]
            
            x_axis = range(1, len(std_times) + 1)
            ax3.plot(x_axis, std_times, 'o-', label='Standard', color='blue', linewidth=2)
            ax3.plot(x_axis, par_times, 's-', label='Parallel', color='green', linewidth=2)
            ax3.set_xlabel('Image Generation #')
            ax3.set_ylabel('Execution Time (s)')
            ax3.set_title('Execution Time Across Generations')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Carbon footprint comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        
        total_std_emissions = sum([r["emissions"] for r in self.results["standard"]])
        total_par_emissions = sum([r["emissions"] for r in self.results["parallel"]])
        
        pie_data = [total_std_emissions, total_par_emissions]
        pie_labels = [f'Standard\n{total_std_emissions:.6f} kg', f'Parallel\n{total_par_emissions:.6f} kg']
        colors = ['lightcoral', 'lightgreen']
        
        wedges, texts, autotexts = ax4.pie(pie_data, labels=pie_labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Total Carbon Footprint Distribution')
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
        🚀 PERFORMANCE SUMMARY:
        • Generated {len(self.results['standard'])} images with each method
        • Average speedup: {improvements['execution_time']:.1f}% faster
        • Carbon reduction: {improvements['emissions']:.1f}% less CO2
        • Memory efficiency: {improvements['memory_used']:.1f}% difference
        
        💰 COST IMPLICATIONS:
        • Standard method: {std['execution_time']:.2f}s per image
        • Parallel method: {par['execution_time']:.2f}s per image
        • Time saved per image: {std['execution_time'] - par['execution_time']:.2f}s
        
        🌱 ENVIRONMENTAL IMPACT:
        • CO2 saved per image: {(std['emissions'] - par['emissions'])*1000:.3f} mg CO2eq
        • Equivalent to {len(self.results['standard']) * (std['emissions'] - par['emissions'])*1000:.2f} mg CO2eq total savings
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, f"{self.project_name}_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def generate_all_comparisons(self):
        """Generate all visual comparisons for presentation."""
        
        print("Generating visual comparisons...")
        
        # Create side-by-side comparisons
        comparison_paths = self.create_side_by_side_comparison()
        
        # Create presentation dashboard
        dashboard_path = self.create_presentation_dashboard()
        
        # Create a summary grid of all comparisons
        if comparison_paths:
            self._create_comparison_grid(comparison_paths)
        
        return {
            "individual_comparisons": comparison_paths,
            "dashboard": dashboard_path,
            "grid": os.path.join(self.output_dir, f"{self.project_name}_comparison_grid.png")
        }
    
    def _create_comparison_grid(self, comparison_paths):
        """Create a grid showing all comparisons."""
        
        if not comparison_paths:
            return
            
        # Load all comparison images
        comparisons = [Image.open(path) for path in comparison_paths]
        
        # Calculate grid dimensions
        n_images = len(comparisons)
        cols = min(2, n_images)  # Max 2 columns for readability
        rows = (n_images + cols - 1) // cols
        
        # Get max dimensions
        max_width = max(img.width for img in comparisons)
        max_height = max(img.height for img in comparisons)
        
        # Create grid
        grid_width = cols * max_width
        grid_height = rows * max_height
        
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        
        for i, img in enumerate(comparisons):
            row = i // cols
            col = i % cols
            x = col * max_width
            y = row * max_height
            grid.paste(img, (x, y))
        
        # Save grid
        grid_path = os.path.join(self.output_dir, f"{self.project_name}_comparison_grid.png")
        grid.save(grid_path)
        print(f"Comparison grid saved to {grid_path}")
        
        return grid_path
    
    # Keep all the original methods
    def save_results(self, filename=None):
        """Save results to a JSON file"""
        if filename is None:
            filename = f"{self.project_name}_results.json"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def generate_report(self, output_format="markdown"):
        """Generate a report comparing the different methods."""
        # ... (keeping the original implementation)
        # Calculate average metrics
        avg_metrics = {}
        for method in self.results:
            if not self.results[method]:
                continue
                
            avg_metrics[method] = {
                "execution_time": np.mean([r["execution_time"] for r in self.results[method]]),
                "emissions": np.mean([r["emissions"] for r in self.results[method]]),
                "memory_used": np.mean([r["memory_used"] for r in self.results[method]]),
            }
        
        if len(avg_metrics) < 2:
            return "Not enough data to generate a comparison report."
        
        # Calculate improvements
        improvements = {}
        if "standard" in avg_metrics and "parallel" in avg_metrics:
            std = avg_metrics["standard"]
            par = avg_metrics["parallel"]
            
            improvements = {
                "execution_time": (std["execution_time"] - par["execution_time"]) / std["execution_time"] * 100,
                "emissions": (std["emissions"] - par["emissions"]) / std["emissions"] * 100,
                "memory_used": (std["memory_used"] - par["memory_used"]) / std["memory_used"] * 100,
            }
        
        # Generate report
        if output_format == "markdown":
            report = "# SDXL Parallel Sampling Performance Report\n\n"
            report += "## Executive Summary\n\n"
            report += f"This report analyzes the performance impact of implementing parallel sampling for SDXL image generation across {len(self.results['standard'])} test images.\n\n"
            
            report += "## Key Findings\n\n"
            report += f"- **Speed Improvement**: {improvements['execution_time']:.1f}% faster generation\n"
            report += f"- **Carbon Reduction**: {improvements['emissions']:.1f}% lower CO2 emissions\n"
            report += f"- **Memory Impact**: {improvements['memory_used']:.1f}% memory difference\n\n"
            
            report += "## Detailed Performance Comparison\n\n"
            report += "| Metric | Standard | Parallel | Improvement |\n"
            report += "|--------|----------|----------|-------------|\n"
            
            for metric in ["execution_time", "emissions", "memory_used"]:
                metric_name = {
                    "execution_time": "Execution Time (s)",
                    "emissions": "Carbon Emissions (kg CO2eq)",
                    "memory_used": "Memory Used (MB)",
                }[metric]
                
                std_val = avg_metrics["standard"][metric]
                par_val = avg_metrics["parallel"][metric]
                
                if metric == "memory_used":
                    std_val /= 1024**2
                    par_val /= 1024**2
                
                imp_val = improvements[metric]
                
                report += f"| {metric_name} | {std_val:.3f} | {par_val:.3f} | {imp_val:+.1f}% |\n"
            
            report += "\n## Environmental Impact\n\n"
            total_std_emissions = sum([r["emissions"] for r in self.results["standard"]])
            total_par_emissions = sum([r["emissions"] for r in self.results["parallel"]])
            total_savings = (total_std_emissions - total_par_emissions) * 1000  # Convert to mg
            
            report += f"- **Total CO2 saved**: {total_savings:.2f} mg CO2eq across all generations\n"
            report += f"- **Per-image savings**: {(total_std_emissions - total_par_emissions) * 1000 / len(self.results['standard']):.3f} mg CO2eq\n"
            report += f"- **Percentage reduction**: {improvements['emissions']:.1f}% lower carbon footprint"
            
            report += "## Technical Implementation\n\n"
            report += "The parallel sampling approach implements Picard iterations across timestep blocks, "
            report += "allowing multiple denoising steps to be processed simultaneously while maintaining image quality.\n\n"
            
            report += "## Recommendations\n\n"
            if improvements['execution_time'] > 0:
                report += "✅ **Recommended for adoption**: Parallel sampling shows measurable performance improvements.\n"
            else:
                report += "⚠️ **Requires optimization**: Current implementation shows performance regression.\n"
            
            if improvements['emissions'] > 0:
                report += "✅ **Environmental benefit**: Reduced carbon footprint makes this approach more sustainable.\n"
            
        else:  # HTML version would go here
            report = f"<h1>SDXL Parallel Sampling Report</h1><p>Performance analysis across {len(self.results['standard'])} images...</p>"
        
        # Save the report
        report_filename = f"{self.project_name}_report.{'md' if output_format == 'markdown' else 'html'}"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        return report