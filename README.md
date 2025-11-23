# GR-MG Video Prediction (DL Final Project)

This repository contains the implementation for the Deep Learning Final Project. The goal is to adapt the **InstructPix2Pix** model for video frame prediction, specifically predicting the 21st frame given a sequence of 20 observed frames and a text instruction.

## Project Overview

We have refactored the original GR-MG codebase to focus on the video generation task. Key modifications include:

1.  **20-Frame Input Support**: 
    - The UNet input channels have been expanded from 8 (4 noisy + 4 image) to **84** (4 noisy + 80 conditioning from 20 frames).
    - The first convolutional layer weights are initialized by copying the original pre-trained weights for the first 8 channels and zero-initializing the rest.

2.  **Custom Pipeline**:
    - Implemented a custom `StableDiffusionInstructPix2PixPipeline` in `utils/pipeline.py`.
    - Added `prepare_image_latents` to handle encoding and concatenating 20 frames of latents.
    - Fixed tensor dimension mismatches for classifier-free guidance (3x concatenation).

3.  **Refactored Structure**:
    - Flattened the project structure for better readability.
    - Removed redundant policy learning code.

4.  **Device Compatibility**:
    - Added auto-detection for CUDA, MPS (Mac), and CPU.
    - Added support for FP16 weights to reduce memory usage.

## Installation

1.  **Create Conda Environment**:
    ```bash
    conda create -n DL_Project python=3.10
    conda activate DL_Project
    ```

2.  **Install Dependencies**:
    ```bash
    pip install torch torchvision torchaudio
    pip install diffusers transformers accelerate matplotlib moviepy einops timm ftfy tensorboard flamingo_pytorch
    pip install "numpy<2.0"  # Ensure compatibility
    ```

## Model Download

The project uses the **InstructPix2Pix** model. Due to its size (~30GB for all variants), it is not included in the git repository.

You can download it using the provided script (configured for Tsinghua Mirror for faster download in China):

```bash
python download_model.py
```

This will download the model to `resources/IP2P`.

## Usage

### Running a Test

We have provided a test script to verify the environment and model configuration using dummy data.

```bash
bash test_run.sh
```

This script will:
1.  Activate the `DL_Project` environment.
2.  Generate a dummy dataset (if not present).
3.  Run `evaluate.py` on a single sample.
4.  Save the visualization to `debug_vis/debug_0.png`.

### Visualization

The output image `debug_vis/debug_0.png` shows:
- **Input Frame t-19**: The first frame of the sequence.
- **Input Frame t**: The last observed frame (20th frame).
- **Ground Truth**: The expected 21st frame.
- **Predicted**: The model's generated 21st frame.

## Project Structure

- `configs/`: Configuration files.
- `data/`: Dataset loading logic (`calvindataset.py`).
- `models/`: Model definitions (`model.py`).
- `utils/`: Utility scripts and custom pipeline (`pipeline.py`).
- `evaluate.py`: Main evaluation script.
- `train.py`: Training script (refactored).
- `resources/`: Directory for model weights (ignored by git).

## Acknowledgements

Based on the InstructPix2Pix architecture and the GR-MG project base.
