# Architecture of FramePack

FramePack is a next-frame prediction model that generates videos progressively. It compresses input contexts to a constant length so that the generation workload is invariant to video length.

## Model Structure

FramePack consists of several key components:

1. **Text Encoders**: LlamaModel and CLIPTextModel for processing text prompts.
2. **Image Encoder**: SiglipVisionModel for processing input images.
3. **VAE (Variational Autoencoder)**: AutoencoderKLHunyuanVideo for encoding/decoding images.
4. **Transformer**: HunyuanVideoTransformer3DModelPacked for generating video frames.

## Generation Pipeline

The video generation process follows these steps:

1. **Text Encoding**
   - Process user-provided prompts using LlamaModel
   - Process negative prompts (if any) for classifier-free guidance
   - Generate text embeddings and attention masks

2. **Image Processing**
   - Resize and center-crop input images to target dimensions
   - Convert to appropriate tensor format for model input
   - Process additional frames if provided

3. **Latent Encoding**
   - Encode input images to latent space using VAE
   - Store latent representations for generation guidance

4. **Vision Feature Extraction**
   - Extract visual features from input images using SiglipVisionModel
   - Prepare image embeddings for conditioning the generation

5. **Frame Generation**
   - Initialize latent noise with specified seed
   - Apply diffusion sampling (UniPC sampler) with guidance controls
   - Generate frames progressively in latent space
   - Apply consistency boosting between frame windows

6. **Decoding and Output**
   - Decode latent representations to pixel space using VAE
   - Compile frames into video with specified FPS
   - Save output as MP4 with configurable compression

## Memory Management

FramePack implements sophisticated memory management to accommodate different hardware configurations:

1. **Dynamic Mode Selection**
   - High VRAM mode (>60GB): All models remain in GPU memory
   - Low VRAM mode: Models are dynamically loaded/unloaded as needed

2. **Optimization Techniques**
   - VAE slicing and tiling in low VRAM environments
   - DynamicSwapInstaller for efficient model offloading (3x faster than Hugging Face's sequential offload)
   - GPU memory preservation parameter to prevent OOM errors
   - TeaCache optimization for faster inference (optional)

3. **Progressive Generation**
   - Videos are generated in sections for longer durations
   - Each section builds upon previous sections
   - Allows for generating videos of arbitrary length with constant memory usage

## User Interface

The application provides a Gradio-based web interface with:

1. **Input Controls**
   - Primary image upload
   - Additional frames gallery with upload/clear functionality
   - Text prompt input with quick prompt examples

2. **Parameter Settings**
   - Video length and FPS controls
   - Generation quality parameters (steps, guidance scales)
   - Motion control parameters (motion bias, consistency boost)
   - Resource management settings (memory preservation, TeaCache)

3. **Output Display**
   - Real-time progress visualization
   - Preview of in-progress generation
   - Final video playback with loop functionality

## Asynchronous Processing

FramePack implements an asynchronous processing system:

1. **Worker Thread**
   - Main generation process runs in a separate thread
   - Prevents UI blocking during long generation tasks

2. **Communication System**
   - AsyncStream for bidirectional communication
   - Progress updates sent to UI during generation
   - User-initiated cancellation support

3. **Error Handling**
   - Graceful error reporting with traceback
   - Resource cleanup on errors or completion
   - Automatic model unloading on process termination

## Technical Considerations

1. **Performance Optimization**
   - Latent window size control for balancing coherence and variation
   - Adjustable sampling steps for quality/speed tradeoff
   - Configurable guidance scales for creative control

2. **Resource Requirements**
   - Minimum VRAM: Works with consumer GPUs (8GB+) with dynamic offloading
   - Optimal performance: High-end GPUs with 24GB+ VRAM
   - CPU and disk used for model storage when not in active use

3. **Output Quality Control**
   - Adjustable MP4 compression (CRF parameter)
   - Configurable output FPS independent of generation FPS
   - Consistency boost parameter for temporal coherence

## Future Directions

1. **Model Improvements**
   - Support for additional model architectures
   - Fine-tuning capabilities for specific domains
   - Integration with other diffusion models

2. **Interface Enhancements**
   - Parameter presets for different generation styles
   - Batch processing capabilities
   - Advanced editing tools for input frames

3. **Performance Optimizations**
   - Multi-GPU support
   - Further memory usage improvements
   - Optimized sampling algorithms

4. **Output Options**
   - Additional video formats
   - Frame extraction capabilities
   - Integration with external editing tools
