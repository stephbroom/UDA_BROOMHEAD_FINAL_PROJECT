# UDA_HW4_Broomhead

## Overview


This repository contains the Jupyter Notebook `UDA_HW_4_Broomhead.ipynb`, which explores advanced imaging techniques for reversing swirl distortions in facial images. 

It also contains an example of some of the images from my test dataset. Note that upload of these images was limited by size, and they may not neccessarily be matching pairs, they are just for illustration. The code for creation of the full synthetic dataset and structure is detialed in the notebook. 

The work creates and leverages a synthetic dataset of swirled and unswirled image pairs from the EasyPortrait dataset and introduces a Distortion Field U-Net model. The model predicts pixel displacement fields to reverse swirl distortions without requiring prior knowledge of the distortion parameters or software used to create them.

## Key Features

- **U-Net-Based Model**: A U-Net architecture enhanced with self-attention layers for capturing long-range dependencies in the images.
- **Custom Composite Loss Function**:
  - **SSIM Loss**: Evaluates structural similarity between reconstructed and original images.
  - **Smoothness Loss**: Ensures smooth transitions in the predicted distortion field.
  - **Perceptual Loss**: Utilizes a pre-trained VGG16 network to evaluate high-level perceptual features.
- **Synthetic Dataset Generation**: 
  - Created using `skimage` to apply swirl distortions with controlled parameters (rotation strength and radius).
  - Each distorted image maintains a one-to-one correspondence with its original counterpart, allowing for rigorous evaluation.
- **Comprehensive Training Pipeline**:
  - Dynamic learning rate adjustment using a `ReduceLROnPlateau` scheduler.
  - Early stopping to prevent overfitting and save the best model.
- **Visualization**:
  - Training and validation loss over epochs.
  - Examples of reconstructed images from the validation and test datasets for qualitative analysis.

## Dataset

The synthetic dataset was generated using the EasyPortrait dataset, comprising high-resolution facial images. The dataset was processed to create swirled-unswirled pairs with varying distortion parameters:
- **Rotation Strength**: Randomly selected from a predefined range.
- **Radius**: Randomly selected from a predefined rang.

Preprocessing steps included:
- Resizing all images to a fixed resolution of 128x128 pixels.
- Normalizing pixel values using the calculated mean and standard deviation of the dataset.

The controlled nature of the synthetic dataset allows the model to focus on learning meaningful transformations rather than noise or unrelated features.

## Requirements

- **Python Version**: Python 3.7+
- **Development Environment**: Software that can work with .ipynb (e.g VS Code, Google Collab)
- **EasyPortrait Dataset**: Download from: https://www.kaggle.com/datasets/kapitanov/easyportrait
- **Dependencies**:
  - `torch` (PyTorch framework for model implementation and training)
  - `torchvision` (for pre-trained models like VGG16)
  - `skimage` (for generating swirl distortions)
  - `matplotlib` (for visualizing results)
  - `numpy` (for numerical computations)
  - `tqdm` (for progress bars in training loops)

## Usage

1. Clone this repository and navigate to the project directory.
2. Install the required dependencies
3. Open the `UDA_HW_4_Broomhead.ipynb` notebook in your chosen software.
4. Ensure that paths to correct drives are changed to connect to dataset correctly
5. For optimal performance, ensure you have adequate GPU, or runtimes may explode.
6. Execute the cells sequentially to train the model and visualize the results.

## Results and Findings

The project demonstrated the effectiveness of a U-Net-based architecture in reconstructing swirled images. The combination of SSIM, smoothness, and perceptual loss enabled the model to achieve both structural and perceptual fidelity in its outputs. While residual swirl artifacts persisted in some cases, the overall performance highlights the potential of this approach for real-world applications.

For more detailed insights, please refer to the notebook and accompanying documentation.

---


