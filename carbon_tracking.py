# carbon_tracking.py
import time
import os
import json
from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
import torch

class CarbonTracker:
    """
    Utility class to track carbon emissions, execution time, and memory usage
    for different sampling methods.
    """
    
    def __init__(self, project_name="sdxl-optimization", output_dir="./results"):
        self.project_name = project_name
        self.output_dir = output_dir
        self.results = {
            "standard": [],
            "parallel": [],
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CodeCarbon tracker
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            log_level="warning",
            save_to_file=True,
        )
    
    def track_inference(self, method_name):
        """
        Decorator to track carbon emissions, execution time, and memory usage
        for a specific inference method.
        
        Args:
            method_name: Name of the method to track (e.g., "standard" or "parallel")
            
        Returns:
            Decorated function
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
                
                # Record metrics
                metrics = {
                    "execution_time": execution_time,
                    "emissions": emissions,
                    "memory_used": memory_used,
                    "timestamp": time.time(),
                    "parameters": kwargs,
                }
                
                self.results[method_name].append(metrics)
                
                print(f"Method: {method_name}")
                print(f"  Execution Time: {execution_time:.2f} s")
                print(f"  Carbon Emissions: {emissions:.6f} kg CO2eq")
                print(f"  Memory Used: {memory_used / 1024**2:.2f} MB")
                
                return result
            
            return wrapper
        
        return decorator
    
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
        """
        Generate a report comparing the different methods.
        
        Args:
            output_format: Format of the report ("markdown" or "html")
            
        Returns:
            Report string in the specified format
        """
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
            report = "# SDXL Optimization Report\n\n"
            report += "## Performance Comparison\n\n"
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
                
                report += f"| {metric_name} | {std_val:.2f} | {par_val:.2f} | {imp_val:.2f}% |\n"
            
            report += "\n## Carbon Footprint Reduction\n\n"
            report += f"By using parallel sampling, the carbon footprint is reduced by {improvements['emissions']:.2f}%.\n"
            report += f"This is equivalent to saving {avg_metrics['standard']['emissions'] - avg_metrics['parallel']['emissions']:.6f} kg CO2eq per inference.\n"
            
        else:  # HTML
            # Simple HTML report
            report = "<h1>SDXL Optimization Report</h1>"
            report += "<h2>Performance Comparison</h2>"
            report += "<table border='1'>"
            report += "<tr><th>Metric</th><th>Standard</th><th>Parallel</th><th>Improvement</th></tr>"
            
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
                
                report += f"<tr><td>{metric_name}</td><td>{std_val:.2f}</td><td>{par_val:.2f}</td><td>{imp_val:.2f}%</td></tr>"
            
            report += "</table>"
            report += "<h2>Carbon Footprint Reduction</h2>"
            report += f"<p>By using parallel sampling, the carbon footprint is reduced by {improvements['emissions']:.2f}%.</p>"
            report += f"<p>This is equivalent to saving {avg_metrics['standard']['emissions'] - avg_metrics['parallel']['emissions']:.6f} kg CO2eq per inference.</p>"
        
        # Save the report
        report_filename = f"{self.project_name}_report.{'md' if output_format == 'markdown' else 'html'}"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        
        return report
    
    def plot_comparison(self, save_path=None):
        """
        Generate comparison plots for the different methods.
        
        Args:
            save_path: Path to save the plots
            
        Returns:
            Path to the saved plots
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{self.project_name}_plots.png")
        
        # Prepare data
        methods = []
        exec_times = []
        emissions = []
        memory_used = []
        
        for method in self.results:
            if not self.results[method]:
                continue
                
            methods.append(method)
            exec_times.append(np.mean([r["execution_time"] for r in self.results[method]]))
            emissions.append(np.mean([r["emissions"] for r in self.results[method]]))
            memory_used.append(np.mean([r["memory_used"] for r in self.results[method]]) / 1024**2)  # Convert to MB
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot execution time
        axes[0].bar(methods, exec_times, color=['blue', 'green'])
        axes[0].set_title('Execution Time (s)')
        axes[0].set_ylabel('Seconds')
        
        # Plot carbon emissions
        axes[1].bar(methods, emissions, color=['blue', 'green'])
        axes[1].set_title('Carbon Emissions (kg CO2eq)')
        axes[1].set_ylabel('kg CO2eq')
        
        # Plot memory usage
        axes[2].bar(methods, memory_used, color=['blue', 'green'])
        axes[2].set_title('Memory Usage (MB)')
        axes[2].set_ylabel('MB')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Plots saved to {save_path}")
        
        return save_path


# Example usage:
def example_usage():
    from diffusers import StableDiffusionXLPipeline
    import torch
    
    # Initialize tracker
    tracker = CarbonTracker(project_name="sdxl-parallel-sampling")
    
    # Load standard pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    
    # Define test prompt
    prompt = "A detailed portrait of an astronaut in space, with Earth visible in the background"
    
    # Track standard inference
    @tracker.track_inference("standard")
    def run_standard_inference(prompt):
        return pipeline(prompt=prompt).images[0]
    
    # Run standard inference
    standard_image = run_standard_inference(prompt)
    
    # Now, let's say we have parallel pipeline (you would need to implement this)
    from parallel_sampling_sdxl import ParallelSDXLPipeline
    parallel_pipeline = ParallelSDXLPipeline(pipeline)
    
    # Track parallel inference
    @tracker.track_inference("parallel")
    def run_parallel_inference(prompt):
        return parallel_pipeline(prompt=prompt).images[0]
    
    # Run parallel inference
    parallel_image = run_parallel_inference(prompt)
    
    # Generate report and plots
    tracker.save_results()
    tracker.generate_report()
    tracker.plot_comparison()
    
    return standard_image, parallel_image

if __name__ == "__main__":
    example_usage()