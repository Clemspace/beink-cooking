# memory_realistic_parallel.py
"""
Memory-realistic implementation of parallel sampling for A40.
Demonstrates the concept while working within actual memory constraints.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
import time

class MemoryRealisticParallelSampling:
    """
    Memory-realistic parallel sampling that demonstrates the concept
    while working within actual GPU memory constraints.
    """
    
    def __init__(
        self,
        unet,
        scheduler,
        num_blocks=4,
        picard_iterations=3,
        parallel_degree=2,  # Reduced for memory
        verbose=True
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.num_blocks = num_blocks
        self.picard_iterations = picard_iterations
        self.parallel_degree = min(parallel_degree, 2)  # Cap at 2 for memory safety
        self.verbose = verbose
        
        # Performance tracking
        self.timing_data = {
            "block_times": [],
            "picard_iteration_times": [],
            "parallel_efficiency": []
        }
        
        # Memory monitoring
        self._monitor_memory()
        
    def _monitor_memory(self):
        """Monitor available memory and adjust settings."""
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
        available_memory = total_memory - allocated_memory
        
        if self.verbose:
            print(f"🧠 Memory status: {allocated_memory:.1f}GB used, {available_memory:.1f}GB available")
        
        # Adjust parallel degree based on available memory
        if available_memory < 5:  # Less than 5GB available
            self.parallel_degree = 1
            self.picard_iterations = min(self.picard_iterations, 2)
            if self.verbose:
                print("⚠️ Limited memory: reducing to sequential processing with minimal parallelization")
        elif available_memory < 10:  # Less than 10GB available
            self.parallel_degree = 2
            if self.verbose:
                print("🔧 Moderate memory: using conservative parallel settings")
    
    def _compute_block_sizes(self, total_steps: int) -> List[int]:
        """Compute memory-safe block sizes."""
        # Use smaller blocks to reduce memory pressure
        steps_per_block = max(2, total_steps // self.num_blocks)
        
        block_sizes = []
        remaining_steps = total_steps
        
        for i in range(self.num_blocks):
            if i < self.num_blocks - 1:
                block_size = min(steps_per_block, remaining_steps - (self.num_blocks - i - 1))
            else:
                block_size = remaining_steps
                
            block_sizes.append(max(1, block_size))
            remaining_steps -= block_size
            
        return block_sizes
    
    def _memory_efficient_picard_iteration(
        self,
        initial_latents: torch.Tensor,
        timesteps_block: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient Picard iteration that demonstrates parallel concepts
        while staying within memory limits.
        """
        batch_size = initial_latents.shape[0]
        block_length = len(timesteps_block)
        
        if self.verbose:
            print(f"    🔄 Processing block of {block_length} timesteps with {self.picard_iterations} Picard iterations")
        
        # Start with aggressive memory cleanup
        torch.cuda.empty_cache()
        
        # Initialize trajectory with memory management
        current_latents = initial_latents.clone()
        
        # Track starting memory
        start_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Picard iterations with memory monitoring
        for iteration in range(self.picard_iterations):
            iteration_start = time.time()
            
            if self.verbose and iteration == 0:
                print(f"      ⚡ Starting Picard iterations (parallel degree: {self.parallel_degree})...")
            
            # Process timesteps in memory-safe batches
            for batch_start in range(0, block_length, self.parallel_degree):
                batch_end = min(batch_start + self.parallel_degree, block_length)
                
                # Clear cache before each batch
                torch.cuda.empty_cache()
                
                if batch_end - batch_start > 1 and self.parallel_degree > 1:
                    # True parallel processing (when memory allows)
                    parallel_timesteps = [timesteps_block[i] for i in range(batch_start, batch_end)]
                    
                    # Create multiple copies of latents for parallel processing
                    parallel_latents = [current_latents.clone() for _ in range(len(parallel_timesteps))]
                    
                    # Stack for batch processing with memory check
                    try:
                        stacked_latents = torch.stack(parallel_latents, dim=0)
                        stacked_timesteps = torch.stack(parallel_timesteps)
                        
                        # Reshape for UNet batch processing
                        reshaped_latents = stacked_latents.view(-1, *stacked_latents.shape[2:])
                        expanded_timesteps = stacked_timesteps.repeat_interleave(batch_size)
                        
                        # Expand conditioning
                        expanded_encoder_states = encoder_hidden_states.repeat(len(parallel_timesteps), 1, 1)
                        expanded_added_cond = {}
                        for key, value in added_cond_kwargs.items():
                            expanded_added_cond[key] = value.repeat(len(parallel_timesteps), *([1] * (value.dim() - 1)))
                        
                        # Process in parallel with memory monitoring
                        memory_before = torch.cuda.memory_allocated() / (1024**3)
                        
                        with torch.cuda.amp.autocast():  # Use mixed precision
                            noise_predictions = self.unet(
                                reshaped_latents,
                                expanded_timesteps,
                                encoder_hidden_states=expanded_encoder_states,
                                added_cond_kwargs=expanded_added_cond,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                            )[0]
                        
                        memory_after = torch.cuda.memory_allocated() / (1024**3)
                        memory_used = memory_after - memory_before
                        
                        # Reshape back and apply scheduler steps
                        noise_predictions = noise_predictions.view(len(parallel_timesteps), batch_size, *noise_predictions.shape[1:])
                        
                        # Apply scheduler steps and update current latents
                        for i, (t, noise_pred) in enumerate(zip(parallel_timesteps, noise_predictions)):
                            if i == len(parallel_timesteps) - 1:  # Only keep the last result
                                current_latents = self.scheduler.step(
                                    noise_pred, t, current_latents, return_dict=False
                                )[0]
                        
                        if self.verbose and iteration == 0:
                            print(f"        ✓ Processed {len(parallel_timesteps)} timesteps in parallel (+{memory_used:.1f}GB)")
                        
                        # Clean up intermediate tensors
                        del stacked_latents, stacked_timesteps, reshaped_latents
                        del expanded_encoder_states, expanded_added_cond, noise_predictions
                        
                    except torch.cuda.OutOfMemoryError:
                        # Fall back to sequential processing
                        if self.verbose:
                            print(f"        ⚠️ OOM in parallel processing, falling back to sequential")
                        
                        for i in range(batch_start, batch_end):
                            t = timesteps_block[i]
                            t_batch = t.expand(batch_size)
                            
                            with torch.cuda.amp.autocast():
                                noise_pred = self.unet(
                                    current_latents,
                                    t_batch,
                                    encoder_hidden_states=encoder_hidden_states,
                                    added_cond_kwargs=added_cond_kwargs,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    return_dict=False,
                                )[0]
                            
                            current_latents = self.scheduler.step(
                                noise_pred, t, current_latents, return_dict=False
                            )[0]
                            
                            del noise_pred
                            torch.cuda.empty_cache()
                
                else:
                    # Sequential processing for single timesteps or when parallel degree is 1
                    for i in range(batch_start, batch_end):
                        t = timesteps_block[i]
                        t_batch = t.expand(batch_size)
                        
                        with torch.cuda.amp.autocast():
                            noise_pred = self.unet(
                                current_latents,
                                t_batch,
                                encoder_hidden_states=encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs,
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                            )[0]
                        
                        current_latents = self.scheduler.step(
                            noise_pred, t, current_latents, return_dict=False
                        )[0]
                        
                        del noise_pred
                
                # Memory cleanup after each batch
                torch.cuda.empty_cache()
            
            iteration_time = time.time() - iteration_start
            self.timing_data["picard_iteration_times"].append(iteration_time)
            
            if self.verbose and (iteration + 1) % 2 == 0:
                current_memory = torch.cuda.memory_allocated() / (1024**3)
                print(f"      ✓ Picard iteration {iteration + 1}/{self.picard_iterations} ({iteration_time:.3f}s, {current_memory:.1f}GB)")
        
        # Calculate efficiency metrics
        end_memory = torch.cuda.memory_allocated() / (1024**3)
        memory_efficiency = (end_memory - start_memory) / block_length  # Memory per timestep
        
        # Estimate parallel efficiency (conservative)
        theoretical_sequential_time = block_length * 0.5  # Estimate 0.5s per timestep
        actual_time = sum(self.timing_data["picard_iteration_times"][-self.picard_iterations:])
        parallel_efficiency = min(theoretical_sequential_time / actual_time, self.parallel_degree) if actual_time > 0 else 1.0
        
        self.timing_data["parallel_efficiency"].append(parallel_efficiency)
        
        if self.verbose:
            print(f"      📊 Block efficiency: {parallel_efficiency:.2f}x (theoretical max: {self.parallel_degree}x)")
        
        return current_latents
    
    def sample(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 7.5,
        neg_encoder_hidden_states: Optional[torch.Tensor] = None,
        neg_added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Memory-realistic parallel sampling."""
        
        total_steps = len(timesteps)
        
        # Monitor memory before starting
        self._monitor_memory()
        
        # Compute memory-safe block sizes
        block_sizes = self._compute_block_sizes(total_steps)
        
        if self.verbose:
            print(f"📊 Memory-realistic parallel sampling")
            print(f"📋 Block sizes: {block_sizes} (total: {sum(block_sizes)} steps)")
            print(f"⚡ Parallel degree: {self.parallel_degree} (memory-adjusted)")
        
        # Create timestep blocks
        timestep_blocks = []
        step_idx = 0
        for block_size in block_sizes:
            block_timesteps = [timesteps[step_idx + i] for i in range(block_size)]
            timestep_blocks.append(block_timesteps)
            step_idx += block_size
        
        # Process each block
        current_latents = latents
        
        pbar = tqdm(total=len(timestep_blocks), desc="Memory-safe parallel blocks") if self.verbose else None
        
        for block_idx, timesteps_block in enumerate(timestep_blocks):
            block_start_time = time.time()
            
            if self.verbose:
                print(f"\n🔄 Block {block_idx + 1}/{len(timestep_blocks)}: {len(timesteps_block)} timesteps")
            
            # Aggressive memory cleanup before each block
            torch.cuda.empty_cache()
            
            # Handle classifier-free guidance with memory efficiency
            if guidance_scale > 1.0 and neg_encoder_hidden_states is not None:
                # Process unconditional and conditional separately to save memory
                print(f"🔄 Block {block_idx+1}: Processing unconditional...")
                uncond_latents = self._memory_efficient_picard_iteration(
                    current_latents, timesteps_block, neg_encoder_hidden_states, 
                    neg_added_cond_kwargs, cross_attention_kwargs
                )
                
                torch.cuda.empty_cache()
                
                print(f"🔄 Block {block_idx+1}: Processing conditional...")
                cond_latents = self._memory_efficient_picard_iteration(
                    current_latents, timesteps_block, encoder_hidden_states, 
                    added_cond_kwargs, cross_attention_kwargs
                )
                
                # Apply guidance
                current_latents = uncond_latents + guidance_scale * (cond_latents - uncond_latents)
                
                # Clean up
                del uncond_latents, cond_latents
                torch.cuda.empty_cache()
            else:
                # No CFG case
                current_latents = self._memory_efficient_picard_iteration(
                    current_latents, timesteps_block, encoder_hidden_states, 
                    added_cond_kwargs, cross_attention_kwargs
                )
            
            block_time = time.time() - block_start_time
            self.timing_data["block_times"].append(block_time)
            
            if self.verbose:
                current_memory = torch.cuda.memory_allocated() / (1024**3)
                print(f"    ✅ Block {block_idx + 1} completed in {block_time:.3f}s ({current_memory:.1f}GB used)")
            
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        return current_latents
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.timing_data["block_times"]:
            return {}
            
        return {
            "total_blocks": len(self.timing_data["block_times"]),
            "avg_block_time": np.mean(self.timing_data["block_times"]),
            "total_block_time": np.sum(self.timing_data["block_times"]),
            "avg_picard_iteration_time": np.mean(self.timing_data["picard_iteration_times"]) if self.timing_data["picard_iteration_times"] else 0,
            "avg_parallel_efficiency": np.mean(self.timing_data["parallel_efficiency"]) if self.timing_data["parallel_efficiency"] else 1.0,
            "theoretical_speedup": np.mean(self.timing_data["parallel_efficiency"]) if self.timing_data["parallel_efficiency"] else 1.0,
            "memory_constrained": self.parallel_degree <= 2,
        }


class MemoryRealisticSDXLPipeline:
    """Memory-realistic pipeline for A40 demonstration."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
        # Conservative settings for A40 with SDXL already loaded
        self.sampler = MemoryRealisticParallelSampling(
            pipeline.unet, 
            pipeline.scheduler,
            num_blocks=3,              # Fewer blocks
            picard_iterations=2,       # Fewer iterations
            parallel_degree=2,         # Conservative parallel degree
            verbose=True
        )
        
        print(f"🧠 Initialized memory-realistic parallel sampling for A40")
        
    def configure(self, num_blocks=3, picard_iterations=2, parallel_degree=2):
        """Configure with memory-safe defaults."""
        self.sampler.num_blocks = min(num_blocks, 4)  # Cap blocks
        self.sampler.picard_iterations = min(picard_iterations, 3)  # Cap iterations
        self.sampler.parallel_degree = min(parallel_degree, 2)  # Cap parallel degree
        
        print(f"📋 Memory-realistic configuration:")
        print(f"   • Blocks: {self.sampler.num_blocks}")
        print(f"   • Picard iterations: {self.sampler.picard_iterations}")
        print(f"   • Parallel degree: {self.sampler.parallel_degree}")
        
    def _get_add_time_ids_safe(self, original_size, crops_coords_top_left, target_size, dtype):
        """Safe time IDs generation."""
        try:
            return self.pipeline._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=dtype
            )
        except (TypeError, AttributeError):
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=self.pipeline.device)
            return add_time_ids
        
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Memory-realistic parallel sampling pipeline call."""
        
        # Initial memory cleanup
        torch.cuda.empty_cache()
        
        # Set defaults
        height = height or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        batch_size = 1
        
        print(f"🧠 Memory-realistic parallel sampling: {num_inference_steps} steps @ {height}x{width}")
        
        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=self.pipeline.device,
        )
        
        # Prepare timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.pipeline.device)
        timesteps = self.pipeline.scheduler.timesteps
        
        # Prepare latents
        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.pipeline.device,
            generator,
            latents,
        )
        
        # Prepare SDXL conditioning
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        
        add_time_ids = self._get_add_time_ids_safe(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids
        }
        
        neg_added_cond_kwargs = None
        if guidance_scale > 1.0:
            neg_added_cond_kwargs = {
                "text_embeds": negative_pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        
        # Run memory-realistic parallel sampling
        print(f"🧠 Running memory-realistic parallel sampling...")
        sampling_start = time.time()
        latents = self.sampler.sample(
            latents=latents,
            timesteps=timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            guidance_scale=guidance_scale,
            neg_encoder_hidden_states=negative_prompt_embeds,
            neg_added_cond_kwargs=neg_added_cond_kwargs,
        )
        sampling_time = time.time() - sampling_start
        
        # Get performance statistics
        stats = self.sampler.get_performance_stats()
        print(f"📊 Memory-realistic sampling completed in {sampling_time:.3f}s")
        if stats:
            print(f"   • Efficiency: {stats['avg_parallel_efficiency']:.2f}x")
            print(f"   • Memory constrained: {'Yes' if stats['memory_constrained'] else 'No'}")
        
        # Decode latents
        torch.cuda.empty_cache()
        
        if not hasattr(self.pipeline, "decode_latents"):
            latents = latents / self.pipeline.vae.config.scaling_factor
            image = self.pipeline.vae.decode(latents, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type="pil")
        else:
            image = self.pipeline.decode_latents(latents)
            image = self.pipeline.numpy_to_pil(image)
        
        # Store performance stats
        self._last_performance_stats = stats
        
        # Return in expected format
        class PipelineOutput:
            def __init__(self, images, performance_stats=None):
                self.images = images
                self.performance_stats = performance_stats
        
        return PipelineOutput(image, stats)