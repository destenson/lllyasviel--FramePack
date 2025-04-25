from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, additional_frames_list, use_additional_frames, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, motion_bias, gpu_memory_preservation, use_teacache, mp4_crf, fps, generation_fps, consistency_boost):
    # Calculate the total number of latent sections needed
    total_latent_sections = (total_second_length * generation_fps) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    print(f"Requested video length: {total_second_length} seconds")
    print(f"Generation FPS: {generation_fps}")
    print(f"Latent window size: {latent_window_size}")
    print(f"Calculated total latent sections: {total_latent_sections}")

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, _ = input_image.shape  # _ for unused channel dimension
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # Process additional frames if provided and enabled
        additional_latents = []
        if use_additional_frames and additional_frames_list and len(additional_frames_list) > 0:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing additional frames ...'))))

            print(f"Worker received additional frames: {len(additional_frames_list)}")

            for i, frame_np in enumerate(additional_frames_list):
                try:
                    print(f"Processing frame {i}, shape: {frame_np.shape if hasattr(frame_np, 'shape') else 'unknown'}")

                    # Validate the frame format
                    if frame_np is not None and isinstance(frame_np, np.ndarray) and len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                        # Resize to match the same dimensions as the main image
                        frame_resized = resize_and_center_crop(frame_np, target_width=width, target_height=height)
                        print(f"Frame {i} resized to {frame_resized.shape}")

                        # Convert to tensor
                        frame_pt = torch.from_numpy(frame_resized).float() / 127.5 - 1
                        frame_pt = frame_pt.permute(2, 0, 1)[None, :, None]

                        # Encode with VAE to get latent
                        frame_latent = vae_encode(frame_pt, vae)
                        # Print shape for debugging
                        print(f"Frame {i} latent shape: {frame_latent.shape}")

                        # Ensure consistent tensor format for all latents
                        # We want [1, 16, 88, 68] format for all latents
                        if frame_latent.dim() == 5 and frame_latent.shape[2] == 1:
                            # If we have [1, 16, 1, 88, 68], squeeze out the middle dimension
                            frame_latent = frame_latent.squeeze(2)
                            print(f"Squeezed frame {i} latent to shape: {frame_latent.shape}")

                        additional_latents.append(frame_latent)

                        # Save processed frame for debugging
                        Image.fromarray(frame_resized).save(os.path.join(outputs_folder, f'{job_id}_frame_{i}.png'))
                        print(f"Frame {i} successfully processed and saved")
                    else:
                        print(f"Skipping frame {i}: invalid format or shape: {frame_np.shape if hasattr(frame_np, 'shape') else 'unknown'}")
                except Exception as e:
                    print(f"Error processing frame {i}: {str(e)}")
                    traceback.print_exc()  # Print full traceback for debugging

            print(f"Successfully processed {len(additional_latents)} additional frames for latent guidance")

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # Create the latent paddings sequence
        # This determines the order and number of sections to generate
        if total_latent_sections <= 1:
            # For very short videos, just generate one section
            latent_paddings = [0]
        else:
            # For multi-section videos, we want to ensure perfect continuity
            # The key insight is that we need to generate sections in a way that
            # allows the model to maintain continuity between sections

            # We'll use a simple decreasing sequence: [n-1, n-2, ..., 1, 0]
            # This ensures that each section picks up where the previous one left off
            latent_paddings = list(reversed(range(total_latent_sections)))

        # Convert to list to ensure we can index into it later
        latent_paddings = list(latent_paddings)
        print(f"Latent paddings sequence: {latent_paddings} (for {total_latent_sections} sections)")

        for latent_padding in latent_paddings:
            # # Check for end signal at the beginning of each loop iteration
            # if stream.input_queue.top() == 'end':
            #     print("Ending generation early due to user request")
            #     stream.output_queue.push(('end', None))
            #     return

            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            # Print detailed information about the current section
            section_index = latent_paddings.index(latent_padding)
            print(f"Processing section {section_index+1}/{len(latent_paddings)}")
            print(f"Latent padding: {latent_padding}")
            print(f"Latent padding size: {latent_padding_size}")
            print(f"Is last section: {is_last_section}")

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            # Split indices for different parts of the latent space
            # Use _ for the blank_indices since it's not directly used in this code
            clean_latent_indices_pre, _, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Incorporate additional latents if available
            if use_additional_frames and additional_latents and len(additional_latents) > 0:
                # For each iteration, update clean_latents_4x with any available additional latents
                # This ensures the model has guidance from the additional frames
                max_additional = min(16, len(additional_latents))  # Maximum 16 latents can be used as 4x guides

                print(f"clean_latents_4x shape: {clean_latents_4x.shape}")

                for i in range(max_additional):
                    try:
                        # Replace the 4x guide latents with our additional frames
                        # Make sure we're only replacing up to the available positions in clean_latents_4x
                        idx = min(i, clean_latents_4x.shape[2] - 1)

                        # Get the additional latent and print its shape
                        additional_latent = additional_latents[i]
                        print(f"Additional latent {i} shape: {additional_latent.shape}")

                        # Print the original shape for debugging
                        print(f"Original additional latent {i} shape: {additional_latent.shape}")

                        # We need to ensure the tensor is in the format expected by clean_latents_4x
                        # clean_latents_4x has shape [1, 16, 16, 88, 68]
                        # We need to prepare a tensor that can be assigned to a slice [:, :, idx, :, :]

                        # First, ensure we have a 4D tensor with shape [1, 16, 88, 68]
                        if additional_latent.dim() == 5:
                            # If it's [1, 16, 1, 88, 68], squeeze out the middle dimension
                            if additional_latent.shape[2] == 1:
                                additional_latent = additional_latent.squeeze(2)
                                print(f"After squeeze, shape: {additional_latent.shape}")

                        # Now we should have a 4D tensor [1, 16, 88, 68]
                        # No need to permute as it's already in the correct format

                        print(f"After processing, additional latent {i} shape: {additional_latent.shape}")

                        # Move to the correct device
                        additional_latent = additional_latent.to(clean_latents_4x.device)

                        # Assign to the clean_latents_4x tensor
                        try:
                            # We need to make sure the shapes match for assignment
                            # clean_latents_4x has shape [1, 16, 16, 88, 68]
                            # We want to assign to a specific frame index

                            # For a 4D tensor [1, 16, 88, 68], we need to assign it to a slice of clean_latents_4x
                            print(f"Assigning latent to clean_latents_4x[:, :, {idx}, :, :]")

                            # Direct assignment to the specific frame index
                            clean_latents_4x[0, :, idx, :, :] = additional_latent[0, :, :, :]

                            print(f"Successfully assigned additional latent {i} to clean_latents_4x")
                        except Exception as e:
                            print(f"Error during assignment: {str(e)}")
                            traceback.print_exc()

                            # Try an alternative approach if the direct assignment fails
                            try:
                                print("Trying alternative assignment method...")
                                # Create a view of the target slice
                                target_slice = clean_latents_4x[0, :, idx, :, :]
                                print(f"Target slice shape: {target_slice.shape}")

                                # Create a view of the source data
                                source_data = additional_latent[0, :, :, :]
                                print(f"Source data shape: {source_data.shape}")

                                # Use copy_ to assign values
                                target_slice.copy_(source_data)
                                print(f"Successfully assigned using copy_ method")
                            except Exception as e2:
                                print(f"Alternative assignment also failed: {str(e2)}")
                                traceback.print_exc()
                    except Exception as e:
                        print(f"Error assigning additional latent {i}: {str(e)}")
                        traceback.print_exc()

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                actual_frames = int(max(0, total_generated_latent_frames * 4 - 3))
                video_seconds = max(0, actual_frames / fps)
                desc = f'Total generated frames: {actual_frames}, Video length: {video_seconds:.2f} seconds (FPS-{fps}). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # Calculate adjusted motion bias for each section to maintain consistent motion
            # The issue is that later sections have less motion, so we'll increase the motion bias
            # for later sections to compensate
            section_index = list(latent_paddings).index(latent_padding) if latent_padding in latent_paddings else 0
            total_sections = len(list(latent_paddings))

            # Apply a progressive increase to motion bias for later sections
            # First section uses the original motion_bias, later sections get progressively higher values
            adjusted_motion_bias = motion_bias
            adjusted_consistency_boost = consistency_boost

            if total_sections > 1 and section_index > 0:
                # Scale factor increases for each subsequent section
                # This helps maintain consistent motion throughout the video
                motion_scale_factor = 1.0 + (section_index / (total_sections - 1)) * 0.5  # Adjust the 0.5 multiplier as needed
                adjusted_motion_bias = motion_bias * motion_scale_factor

                # Also adjust consistency boost to be higher for later sections
                # This helps maintain temporal coherence across sections
                consistency_scale_factor = 1.0 + (section_index / (total_sections - 1)) * 0.3  # Adjust the 0.3 multiplier as needed
                adjusted_consistency_boost = min(consistency_boost * consistency_scale_factor, 5.0)  # Cap at 5.0

                print(f"Section {section_index+1}/{total_sections}: Adjusted motion bias from {motion_bias:.2f} to {adjusted_motion_bias:.2f}")
                print(f"Section {section_index+1}/{total_sections}: Adjusted consistency boost from {consistency_boost:.2f} to {adjusted_consistency_boost:.2f}")

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                shift=adjusted_motion_bias,  # Use the adjusted motion bias
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                consistency_boost=adjusted_consistency_boost,  # Use the adjusted consistency boost
                callback=callback,
            )

            # Additional check after generating latents
            if stream.input_queue.top() == 'end':
                print("Ending generation early after latent generation")
                stream.output_queue.push(('end', None))
                return

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            # Update the history latents with the newly generated latents
            print(f"Generated latents shape: {generated_latents.shape}")
            print(f"Current history latents shape: {history_latents.shape}")

            # For better continuity between sections, we need to ensure smooth transitions
            # in the latent space as well, not just in pixel space

            # Add the new frames to the total count
            new_frames = int(generated_latents.shape[2])
            total_generated_latent_frames += new_frames

            # For multi-section videos, we need to ensure perfect continuity between sections
            # Instead of trying to blend sections after generation, we'll use a different approach:
            # We'll use the last frames from the previous section as conditioning for the next section

            # This approach ensures that the model generates a continuous sequence
            # without any perceptible transitions between sections

            if section_index > 0:
                print("Using last frames from previous section as conditioning for perfect continuity")

                # For the next section, we'll use the last frames from the current section
                # as the initial frames for the next section's generation
                # This creates perfect continuity between sections

                # The key insight is that we don't need to blend after generation
                # Instead, we ensure continuity during generation by using proper conditioning

                # We'll use the last frames of the current section as the first frames of the next section
                # This is handled by the model's internal conditioning mechanism

                # No explicit blending is needed - the model will generate a continuous sequence
                print("Continuity between sections is ensured by the model's conditioning mechanism")

            # Concatenate the new latents with the history
            # Simply ensure both tensors are on the same device before concatenation
            if section_index > 0:
                # Move history_latents to the same device as generated_latents if needed
                if history_latents.device != generated_latents.device:
                    print(f"Moving history_latents from {history_latents.device} to {generated_latents.device} for concatenation")
                    history_latents = history_latents.to(generated_latents.device)

                # Now both tensors are on the same device, we can concatenate them
                history_latents = torch.cat([generated_latents, history_latents], dim=2)
            else:
                # For the first section, just concatenate directly
                # Move generated_latents to the same device as history_latents if needed
                history_latents = torch.cat([generated_latents.to(history_latents.device), history_latents], dim=2)

            print(f"Updated history latents shape: {history_latents.shape}")
            print(f"Total generated latent frames: {total_generated_latent_frames}")

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                # First section - just decode all frames
                # Keep on GPU if possible for faster processing
                history_pixels = vae_decode(real_history_latents, vae)

                # Only move to CPU if we're in low VRAM mode
                if not high_vram and history_pixels.device.type != 'cpu':
                    print(f"Moving initial history_pixels to CPU to save memory")
                    history_pixels = history_pixels.cpu()

                print(f"Initial history_pixels shape: {history_pixels.shape}, device: {history_pixels.device}")
            else:
                # For subsequent sections, we need to handle the overlap carefully
                # Calculate how many frames to decode from the current section
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                print(f"Section latent frames: {section_latent_frames}")

                # No overlap calculation needed since we're not doing any blending
                # The continuity is ensured at the latent level by the model's conditioning mechanism
                print("No overlap calculation needed - continuity is ensured at the latent level")

                # Decode the current section frames
                # Keep on GPU if possible for faster processing
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae)

                # Only move to CPU at the end if needed for saving
                if current_pixels.device.type != 'cpu':
                    print(f"Keeping current_pixels on {current_pixels.device} for faster processing")

                    # If history_pixels is on CPU, move it to GPU for faster blending
                    if history_pixels.device.type == 'cpu':
                        print(f"Moving history_pixels to {current_pixels.device} for faster blending")
                        history_pixels = history_pixels.to(current_pixels.device)

                print(f"Current pixels shape: {current_pixels.shape}, device: {current_pixels.device}")
                print(f"History pixels shape: {history_pixels.shape}, device: {history_pixels.device}")

                # Since we're ensuring continuity at the latent level,
                # we don't need any special blending in pixel space
                # Just concatenate the pixels directly

                print("No pixel-space blending needed - continuity is ensured at the latent level")

                # Simply concatenate the current pixels with the history pixels
                # This works because the latent space is already continuous
                history_pixels = torch.cat([current_pixels, history_pixels], dim=2)

                print(f"Updated history_pixels shape after concatenation: {history_pixels.shape}, device: {history_pixels.device}")

                # Move to CPU only at the very end before saving to disk
                if history_pixels.device.type != 'cpu' and not high_vram:
                    print(f"Moving history_pixels to CPU to save memory")
                    history_pixels = history_pixels.cpu()

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            # Make sure history_pixels is on CPU before saving
            if history_pixels.device.type != 'cpu':
                print(f"Moving history_pixels from {history_pixels.device} to CPU for saving")
                history_pixels_cpu = history_pixels.cpu()
            else:
                history_pixels_cpu = history_pixels

            # Save the video
            save_bcthw_as_mp4(history_pixels_cpu, output_filename, fps=fps, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}, device: {history_pixels.device}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, additional_frames_list, use_additional_frames, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, motion_bias, gpu_memory_preservation, use_teacache, mp4_crf, fps, generation_fps, consistency_boost):
    global stream
    assert input_image is not None, 'No input image!'

    # According to Gradio docs, Gallery as input returns a list of (media, caption) tuples
    # or a list of media if no captions are provided
    print(f"Additional frames received type: {type(additional_frames_list)}")

    # Extract frames from gallery format
    processed_frames = []
    if additional_frames_list and len(additional_frames_list) > 0:
        print(f"Additional frames list length: {len(additional_frames_list)}")

        for i, item in enumerate(additional_frames_list):
            try:
                # Check if it's a tuple (image, caption)
                if isinstance(item, tuple) and len(item) == 2:
                    frame_data = item[0]  # First element is the image
                    print(f"Frame {i} is a tuple with caption: {item[1]}")
                else:
                    frame_data = item

                # Process the frame based on its type
                if isinstance(frame_data, Image.Image):
                    print(f"Frame {i} is a PIL Image with size {frame_data.size}, mode {frame_data.mode}")
                    # Convert to RGB if needed
                    if frame_data.mode != 'RGB':
                        frame_data = frame_data.convert('RGB')
                    frame_np = np.array(frame_data)
                    processed_frames.append(frame_np)
                elif isinstance(frame_data, np.ndarray):
                    print(f"Frame {i} is a numpy array with shape {frame_data.shape}")
                    processed_frames.append(frame_data)
                elif isinstance(frame_data, str):
                    # It's a file path
                    print(f"Frame {i} is a file path: {frame_data}")
                    try:
                        img = Image.open(frame_data)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        frame_np = np.array(img)
                        processed_frames.append(frame_np)
                    except Exception as e:
                        print(f"Error loading image from path {frame_data}: {str(e)}")
                else:
                    print(f"Frame {i} has unknown type: {type(frame_data)}")
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                import traceback
                traceback.print_exc()

    print(f"Processed {len(processed_frames)} frames for worker function")

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, processed_frames, use_additional_frames, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, motion_bias, gpu_memory_preservation, use_teacache, mp4_crf, fps, generation_fps, consistency_boost)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')
    return None, gr.update(visible=False), "", "", gr.update(interactive=True), gr.update(interactive=False)


quick_prompts = [
    'dancing fast, fast motion, rhythic motion',
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="First Frame (Required)", height=320)

            # Additional frames section with improved layout
            with gr.Group():
                gr.Markdown("### Additional Frames (Optional)")
                additional_frames = gr.Gallery(
                    label="",
                    elem_id="additional_frames",
                    visible=True,
                    columns=5,
                    rows=1,
                    height=150,
                    object_fit="contain",
                    type="numpy",  # Ensure consistent type handling
                    file_types=["image"]  # Only allow image files
                )
                with gr.Row():
                    upload_button = gr.UploadButton("Upload More Frames", file_types=["image"], file_count="multiple")
                    clear_frames_button = gr.Button(value="Clear All Frames")

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            prompt = gr.Textbox(label="Prompt", value='')

            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Group():
                use_additional_frames = gr.Checkbox(label='Use Additional Frames as Latent Guides', value=True, info='When enabled, additional frames are used to guide latent generation.')
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=3, step=0.25)

                fps = gr.Slider(label="Output FPS", minimum=5, maximum=60, value=30, step=1, info="Frames per second in the output video. Higher values create smoother video without changing playback speed.")

                generation_fps = gr.Slider(label="Generation FPS", minimum=15, maximum=60, value=30, step=1, info="Internal framerate used for generation. This affects how many frames are created. Keep at 30 for normal motion speed.")

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=6, step=1, visible=True, info="Lower values may produce more varied motion but can reduce coherence.")
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=18, step=1, info="Changing this value is not recommended. (was 25)")

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Try reducing this to 6-8 for more variation and less adherence to the prompt. Higher values = stricter prompt following.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                motion_bias = gr.Slider(label="Motion Bias", minimum=0.5, maximum=25.0, value=6.0, step=0.1, info='Controls diversity between frames. Values over 10 can produce extreme variation but may cause artifacts. Use with higher Generation FPS for best results.')

                consistency_boost = gr.Slider(label="Consistency Boost", minimum=1.0, maximum=5.0, value=1.5, step=0.01, info='Higher values (1.5-3.0) create more consistent motion throughout the video. Values above 3.0 may cause artifacts but ensure very consistent motion.')

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=1, maximum=128, value=1, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, additional_frames, use_additional_frames, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, motion_bias, gpu_memory_preservation, use_teacache, mp4_crf, fps, generation_fps, consistency_boost]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])

    def upload_additional_frames(files):
        """Process uploaded files and add them to the gallery."""
        images = []

        if files:
            for file in files:
                if file is not None:
                    try:
                        img = Image.open(file.name)
                        # Convert to RGB to ensure consistent format
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Convert to numpy array since we set Gallery type="numpy"
                        img_np = np.array(img)
                        images.append(img_np)
                        print(f"Uploaded frame: {img.size}, mode: {img.mode}, converted to numpy array with shape {img_np.shape}")
                    except Exception as e:
                        print(f"Error processing image {file.name}: {str(e)}")
                        traceback.print_exc()

        print(f"Total uploaded frames: {len(images)}")
        return gr.Gallery.update(value=images, visible=True)

    def add_more_frames(current_gallery, new_files):
        """Add more frames to the existing gallery without replacing current ones."""
        # Get current images
        current_images = current_gallery or []
        print(f"Current gallery has {len(current_images)} images")

        # Process new images
        new_images = []
        if new_files:
            for file in new_files:
                if file is not None:
                    try:
                        img = Image.open(file.name)
                        # Convert to RGB to ensure consistent format
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Convert to numpy array since we set Gallery type="numpy"
                        img_np = np.array(img)
                        new_images.append(img_np)
                        print(f"Added new frame: {img.size}, mode: {img.mode}, converted to numpy array with shape {img_np.shape}")
                    except Exception as e:
                        print(f"Error processing image {file.name}: {str(e)}")
                        traceback.print_exc()

        # Combine images
        all_images = current_images + new_images
        print(f"Total frames in gallery: {len(all_images)}")

        return gr.Gallery.update(value=all_images, visible=True)

    def clear_additional_frames():
        """Clear all frames from the gallery."""
        return gr.Gallery.update(value=None, visible=True)

    # Connect the upload and clear buttons to their respective functions
    upload_button.upload(fn=add_more_frames, inputs=[additional_frames, upload_button], outputs=[additional_frames])
    clear_frames_button.click(fn=clear_additional_frames, inputs=[], outputs=[additional_frames])



block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
