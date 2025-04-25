# FramePack Design

## Overview

FramePack is an AI-powered video generation application that transforms a single image (or optionally multiple images) into a coherent video sequence. The application leverages Hunyuan Video models and provides a user-friendly interface built with Gradio.

## Architecture

### Core Components

1. **Model Components**
   - Text Encoders: LlamaModel and CLIPTextModel for processing text prompts
   - Image Encoder: SiglipVisionModel for processing input images
   - VAE (Variational Autoencoder): AutoencoderKLHunyuanVideo for encoding/decoding images
   - Transformer: HunyuanVideoTransformer3DModelPacked for generating video frames

2. **Memory Management System**
   - Dynamic memory allocation based on available GPU VRAM
   - Model offloading mechanism for low VRAM environments
   - TeaCache optimization for improved performance

3. **User Interface**
   - Gradio-based web interface
   - Interactive controls for generation parameters
   - Real-time progress visualization

4. **Processing Pipeline**
   - Asynchronous processing with callback system
   - Frame processing and latent generation
   - Video encoding and output

## Workflow

1. **Initialization**
   - Load models (text encoders, image encoder, VAE, transformer)
   - Configure memory management based on available VRAM
   - Set up Gradio interface

2. **Input Processing**
   - User uploads a primary image and optional additional frames
   - User configures generation parameters and provides text prompts
   - Input validation and preprocessing

3. **Generation Process**
   - Text encoding: Process text prompts using LlamaModel and CLIPTextModel
   - Image processing: Resize and prepare input images
   - VAE encoding: Convert images to latent representations
   - CLIP Vision encoding: Extract visual features from input images
   - Sampling: Generate video frames using the transformer model
   - VAE decoding: Convert latent representations back to pixel space
   - Video encoding: Compile frames into an MP4 video

4. **Output Delivery**
   - Display generated video in the interface
   - Save video file to outputs directory
   - Provide progress updates during generation

## Key Features

1. **Image-to-Video Generation**
   - Transform still images into fluid motion videos
   - Control video length, FPS, and motion characteristics

2. **Multi-Frame Guidance**
   - Optional use of additional frames to guide the generation process
   - Improved temporal consistency with multiple reference frames

3. **Customizable Parameters**
   - Adjustable video length, FPS, and quality settings
   - Fine-tuning controls for motion bias, guidance scale, and consistency

4. **Resource Optimization**
   - Adaptive memory management for different hardware configurations
   - Optional TeaCache for improved performance
   - Progressive generation for longer videos

## Technical Specifications

### Model Details

- **Text Encoders**:
  - LlamaModel: Processes primary text prompts
  - CLIPTextModel: Provides additional text understanding

- **Image Processing**:
  - SiglipVisionModel: Extracts visual features from input images
  - AutoencoderKLHunyuanVideo: Handles image encoding/decoding

- **Video Generation**:
  - HunyuanVideoTransformer3DModelPacked: Core model for frame generation
  - Sampling methods: UniPC sampler with guidance controls

### Performance Considerations

- **Memory Management**:
  - High VRAM mode (>60GB): Models remain in GPU memory
  - Low VRAM mode: Dynamic model offloading with DynamicSwapInstaller
  - GPU memory preservation parameter for OOM prevention

- **Processing Optimizations**:
  - Latent window size control for balancing coherence and variation
  - TeaCache for faster processing (with minor quality trade-offs)
  - Asynchronous processing with progress updates

### Interface Components

- **Input Section**:
  - Primary image upload
  - Additional frames gallery with upload/clear controls
  - Text prompt input with quick prompt examples

- **Parameter Controls**:
  - Video length and FPS settings
  - Generation quality parameters
  - Advanced controls for fine-tuning

- **Output Display**:
  - Preview of in-progress generation
  - Final video playback with loop functionality
  - Progress bar and status updates

## Implementation Notes

1. **Asynchronous Processing**
   - Worker function runs in a separate thread
   - Communication via AsyncStream for progress updates
   - Allows UI to remain responsive during generation

2. **Memory Optimization**
   - Models are loaded/unloaded as needed during generation
   - Memory preservation parameter prevents OOM errors
   - Slicing and tiling for VAE in low VRAM mode

3. **Progressive Generation**
   - Videos are generated in sections for longer durations
   - Each section builds upon previous sections
   - Allows for generating videos of arbitrary length

4. **Error Handling**
   - Graceful error reporting with traceback
   - User-initiated cancellation support
   - Resource cleanup on errors or completion

## Future Enhancements

1. **Model Improvements**
   - Support for additional model architectures
   - Fine-tuning options for specific use cases

2. **Interface Enhancements**
   - Saved parameter presets
   - Batch processing capabilities
   - Advanced editing tools for input frames

3. **Performance Optimizations**
   - Further memory usage improvements
   - Multi-GPU support
   - Optimized sampling algorithms

4. **Output Options**
   - Additional video formats and quality settings
   - Frame extraction capabilities
   - Integration with external editing tools
