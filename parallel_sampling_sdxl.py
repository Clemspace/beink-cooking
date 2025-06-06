# parallel_sampling_sdxl_fixed.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm

class ParallelSamplingSDXL:
    """
    Integrates parallel sampling technique with lightweight SDXL UNet.
    Implements the parallel Picard iterations approach from the paper.
    """
    
    def __init__(
        self,
        unet,
        scheduler,
        num_blocks=4,
        picard_iterations=3,
        verbose=True
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.num_blocks = num_blocks
        self.picard_iterations = picard_iterations
        self.verbose = verbose
        
    def _picard_block_iteration(
        self,
        latents: torch.Tensor,
        timesteps: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Dict[str, Any],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Perform Picard iterations for a block of timesteps.
        """
        batch_size = latents.shape[0]
        num_steps = len(timesteps)
        
        # Create initial guesses for all timesteps in the block
        latent_sequence = [latents.clone() for _ in range(num_steps)]
        
        # Perform multiple Picard iterations
        for iteration in range(self.picard_iterations):
            updated_sequence = []
            
            # Process all timesteps in parallel
            for i, (t, curr_latent) in enumerate(zip(timesteps, latent_sequence)):
                # Expand timestep for batch dimension
                t_batch = t.expand(batch_size)
                
                # Get model prediction (noise residual)
                noise_pred = self.unet(
                    curr_latent,
                    t_batch,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                
                # Apply scheduler step using the prediction
                scheduler_output = self.scheduler.step(
                    noise_pred, t, curr_latent, return_dict=False
                )
                
                # Update sequence with new prediction
                updated_sequence.append(scheduler_output[0])  # prev_sample
            
            # Update sequence for next iteration
            latent_sequence = updated_sequence
        
        # Return the final latent from this block
        return latent_sequence[-1]
    
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
        """
        Main sampling function using parallel block processing.
        """
        # Total number of denoising steps
        num_steps = len(timesteps)
        
        # Calculate steps per block
        steps_per_block = max(1, num_steps // self.num_blocks)
        
        # Create blocks of timesteps
        timestep_blocks = []
        for i in range(0, num_steps, steps_per_block):
            end_idx = min(i + steps_per_block, num_steps)
            timestep_blocks.append(timesteps[i:end_idx])
        
        # Process each block sequentially
        pbar = tqdm(total=len(timestep_blocks), desc="Parallel blocks") if self.verbose else None
        
        for block_idx, block_timesteps in enumerate(timestep_blocks):
            # Perform classifier-free guidance if requested
            if guidance_scale > 1.0 and neg_encoder_hidden_states is not None:
                # Combine conditional and unconditional in batch for efficiency
                combined_latents = torch.cat([latents, latents])
                combined_encoder_states = torch.cat([neg_encoder_hidden_states, encoder_hidden_states])
                
                # Combine added conditioning
                combined_added_cond = {}
                for key in added_cond_kwargs:
                    neg_val = neg_added_cond_kwargs.get(key, added_cond_kwargs[key])
                    combined_added_cond[key] = torch.cat([neg_val, added_cond_kwargs[key]])
                
                # Process combined batch
                combined_result = self._picard_block_iteration(
                    combined_latents, 
                    block_timesteps, 
                    combined_encoder_states, 
                    combined_added_cond,
                    cross_attention_kwargs
                )
                
                # Split and apply guidance
                uncond_latents, cond_latents = combined_result.chunk(2)
                latents = uncond_latents + guidance_scale * (cond_latents - uncond_latents)
            else:
                # No guidance, just process the conditional branch
                latents = self._picard_block_iteration(
                    latents, block_timesteps, encoder_hidden_states, 
                    added_cond_kwargs, cross_attention_kwargs
                )
            
            if pbar is not None:
                pbar.update(1)
                
        if pbar is not None:
            pbar.close()
            
        return latents


class ParallelSDXLPipeline:
    """
    Pipeline adapter to integrate parallel sampling with existing diffusers pipelines.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.sampler = ParallelSamplingSDXL(
            pipeline.unet, 
            pipeline.scheduler,
            num_blocks=4,
            picard_iterations=3
        )
        
    def configure(self, num_blocks=4, picard_iterations=3):
        """Update sampler configuration"""
        self.sampler.num_blocks = num_blocks
        self.sampler.picard_iterations = picard_iterations
    
    def _get_add_time_ids_safe(self, original_size, crops_coords_top_left, target_size, dtype):
        """
        Safely get add_time_ids, handling cases where the pipeline method might fail.
        """
        try:
            # Try the pipeline's method first
            return self.pipeline._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=dtype
            )
        except (TypeError, AttributeError) as e:
            # Fallback: create time IDs manually
            print(f"⚠️ Pipeline _get_add_time_ids failed: {e}")
            print("🔧 Using fallback time IDs generation")
            
            # Create time IDs manually - this is what SDXL expects
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
        """Override pipeline call method to use parallel sampling"""
        
        # Set defaults
        height = height or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        batch_size = 1
        
        # Encode prompts
        try:
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
        except Exception as e:
            print(f"❌ Prompt encoding failed: {e}")
            raise e
        
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
        
        # Prepare added conditioning for SDXL
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        
        # Use safe method to get time IDs
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
        
        # Run parallel sampling
        try:
            latents = self.sampler.sample(
                latents=latents,
                timesteps=timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                guidance_scale=guidance_scale,
                neg_encoder_hidden_states=negative_prompt_embeds,
                neg_added_cond_kwargs=neg_added_cond_kwargs,
            )
        except Exception as e:
            print(f"❌ Parallel sampling failed: {e}")
            raise e
        
        # Decode latents
        try:
            if not hasattr(self.pipeline, "decode_latents"):
                # Fallback for newer diffusers versions
                latents = latents / self.pipeline.vae.config.scaling_factor
                image = self.pipeline.vae.decode(latents, return_dict=False)[0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")
            else:
                image = self.pipeline.decode_latents(latents)
                image = self.pipeline.numpy_to_pil(image)
        except Exception as e:
            print(f"❌ Latent decoding failed: {e}")
            raise e
        
        # Return in expected format
        class PipelineOutput:
            def __init__(self, images):
                self.images = images
        
        return PipelineOutput(image)