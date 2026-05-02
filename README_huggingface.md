---
license: mit
library_name: pytorch
tags:
  - medical-imaging
  - synthetic-data
  - image-utility
  - interpretability
  - pytorch
  - computer-vision
  - research-code
pipeline_tag: image-classification
---

# Interpretable Similarity of Synthetic Image Utility (IUS)

[![GitHub stars](https://img.shields.io/github/stars/innoisys/ius.svg?style=flat&label=Star)](https://github.com/innoisys/ius/)
[![Readme](https://img.shields.io/badge/README-green.svg)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


> **Status:** This repository currently contains the official **PyTorch** implementation of the *IUS* measure, introduced in [**"Interpretable Similarity of Synthetic Image Utility"**](https://ieeexplore.ieee.org/document/11458792) , published in **IEEE Transactions on Medical Imaging (TMI)**.
> Pretrained checkpoints are not included yet and will be released separately.

## Overview
**IUS** (*Interpretable Utility Similarity*) is an interpretable measure for assessing the utility of synthetic medical image datasets for downstream clinical decision support (CDS) tasks, built upon the **EPU-CNN** framework introduced in [*E pluribus unum interpretable convolutional neural networks*](https://www.nature.com/articles/s41598-023-38459-1) and available in our [previous code repository](https://github.com/innoisys/epu-cnn-torch).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Prepare Configuration](#1-prepare-configuration)
  - [2. Supported Dataset Structures for EPU-CNN Training](#2-supported-dataset-structures-for-epu-cnn-training)
    - [2.1 Filename-based Structure](#21-filename-based-structure)
    - [2.2 Folder-based Structure](#22-folder-based-structure)
    - [2.3 MedMNIST Benchmark Structure](#23-medmnist-benchmark-structure)
  - [3. EPU-CNN Training](#3-epu-cnn-training)
  - [4. EPU-CNN Evaluation](#4-epu-cnn-evaluation)
  - [5. Feature contribution profile estimation](#5-feature-contribution-profile-estimation)
  - [6. Synthetic image utility evaluation with IUS](#6-synthetic-image-utility-evaluation-with-ius)
- [Paper Abstract](#-paper-abstract)
- [Citation](#citation)
- [Contact](#-contact)
- [License](#-license)
- [TODO](#todo)

## Features
- YAML-based configuration for reproducible training, inference, and evaluation
- Flexible support for multiple training dataset structures
- Integrated **TensorBoard** logging for experiment monitoring
- Complete implementation of the **IUS** pipeline:
  1. **EPU-CNN training** and classification performance evaluation
  2. **Baseline feature contribution profile** estimation
  3. **Synthetic dataset or singleton evaluation using IUS** 
- Support for both **color** and **grayscale** medical imaging modalities
- Reproducible experiment management through structured saving of checkpoints, logs, timestamped results, and evaluation outputs


## Installation
Clone the repository:
```bash
git clone https://github.com/innoisys/ius.git
cd ius
```
Install dependencies:
```bash
  pip install -r requirements.txt
```

## Project Structure
```
ius/
├── configs/                        # YAML configuration files
├── data/                           # Data loading and preprocessing
│   ├── data_utils.py                   # Common utilities for image/data handling
│   ├── dataloader.py                   # Dataloader definitions
│   ├── dataset.py                      # Dataset implementations
│   ├── loading.py                      # Dataset setup from YAML configuration
│   ├── parsers.py                      # Data parser implementations
│   └── perceptual_transforms.py        # PFM (Perceptual Feature Map) generation
├── datasets/                       # Real datasets for EPU-CNN training & baseline feature contribution profile estimation
├── datasets_synthetic/             # Synthetic data for IUS evaluation
├── ius/                          # IUS implementation
│   ├── ius.py                          # IUS measure class
│   └── ius_eval_parser.py              # Suggested synthetic data parser (not requiring label information)
├── model/                          # EPU-CNN model implementation
│   ├── epu.py                          # Main EPU-CNN model definition
│   ├── module_mapping.py               # Mappings from YAML config names to torch.nn layers/activations
│   ├── register_modules.py             # Registry for configurable model components
│   ├── subnetworks.py                  # Subnetwork implementation
│   └── subnetwork_utilities.py         # Subnetwork helper modules
├── results/                        # Training and inference outputs
│   ├── cb_vectors.py                   # Saved baseline feature contribution profiles (from infer_cb_vector.py)
│   ├── checkpoints.py                  # Saved EPU-CNN checkpoints and training configurations (from train_epu.py)
│   ├── classification_performance.py   # Classification performance results (from infer_epu.py)
│   ├── ius_eval.py                     # IUS evaluation results (from eval_ius.py)
│   └── logs.py                         # TensorBoard logs (from train_epu.py )
├── scripts/                        # Training and inference scripts
│   ├── eval_ius.py                     # Runs synthetic data evaluation with IUS
│   ├── infer_cb_vector.py              # Estimates baseline feature contribution profiles
│   ├── infer_epu.py                    # Runs EPU-CNN inference/evaluation 
│   └── train_epu.py                    # Trains EPU-CNN models
└── utils/                          # Utility functions
    ├── callbacks.py                    # Training callbacks
    ├── config_utils.py                 # YAML/configuration utilities
    ├── early_stopping.py               # Early stopping logic
    ├── eval_utils.py                   # Utilities for EPU-CNN evaluation scripts
    ├── metrics.py                      # Classification performance metrics 
    ├── omega_parser.py                 # OmegaConf-based configuration parser 
    ├── sanity_utils.py                 # Configuration validation and sanity checks
    ├── tensorboard.py                  # Tensorboard utilities
    ├── train_utils.py                  # Training setup and helping utilities
    └── trainer.py                      # Main training loop implementation
```

## Usage
### 1. Prepare Configuration

Create a YAML configuration file in `configs/` with the following structure:

```yaml
model:
    num_subnetworks: 4                      # set to 4, corresponds to number of perceptual feature maps, 
    num_classes:     1
    epu_activation: "sigmoid"               
    subnetwork_config:
        architecture:   "base_one"          # default ius backbone
        input_channels: 1                   # number of channels in perceptual feature decomposition, set to 1
        base_channels:  32
        fc_hidden_units: 64
        pred_activation: "tanh"
data_params:
    dataset_path: "../datasets/dataset_name"
    images_extension: "jpg"
    data_loading:
        batch_size:         64
        shuffle:            true
        num_workers:        0
        pin_memory:         false
        persistent_workers: false
    data_preprocessing:
        data_mode: "rgb"                        # "rgb" or "grayscale"
        data_parser: "filename"                 # "filename" or "folder" or "medmnist"
        resize_dims: [128, 128]
        medmnist_csv_file:  None
        label_mapping:
            abnormal: 1
            normal: 0
train_params:
    mode:       "binary"
    loss:       "binary_cross_entropy"
    epochs:     200
    optimizer:  "sgd"
    learning_rate:  0.001 
    momentum:       0.9
    weight_decay:   0.001
    early_stopping_patience: 30
    early_stopping_monitor:  "val_loss"         # "val_loss" or "val_metrics.auc"
    early_stopping_mode:     "min"              # "min" or "max"
log_dir:          "../results/logs"             # default parent path
checkpoint_dir:   "../results/checkpoints"      # default parent path
experiment_name:  "ius_dataset_name"            # desired experiment path
```

**Key Points:**
- For binary classification: Use `n_classes: 1`,  `epu_activation: "sigmoid"`
- Adjust `input_size` and `batch_size` based on your GPU memory
- Set `label_mapping` according to your dataset classes

### 2. Supported Dataset Structures for EPU-CNN Training

EPU-CNN training supports multiple dataset organization patterns. Complete examples are provided below:

#### 2.1 Filename-based Structure

```
datasets/
├── dataset_name/
    ├── train/
    │   ├── abnormal_001.jpg        
    │   ├── abnormal_002.jpg        
    │   ├── normal_001.jpg
    │   ├── normal_002.jpg
    ├── validation/
    │   ├── abnormal_003.jpg        
    │   ├── normal_003.jpg
    └── test/
        ├── abnormal_004.jpg        
        └── normal_004.jpg
```

**Configuration Example:**
```yaml
data_params:
    dataset_path: "../datasets/dataset_name"
    images_extension: "jpg"
    data_preprocessing:
        data_mode: "rgb"                        # "rgb" or "grayscale"
        data_parser: "filename"                 
        resize_dims: [128, 128]
        medmnist_csv_file:  None
        label_mapping:
            abnormal: 1
            normal: 0
```

**Key Points:**
- Requires a consistent organization across train/validation/test splits
- The validation folder name must be exactly "validation" (not "val" or other variants)
- Supported image formats include jpg, jpeg, png, etc, supported by the parser
- Class names defined in label_mapping must appear in filenames

#### 2.2 Folder-based Structure

```
datasets/
├── dataset_name/
    ├── train/
    │   ├── abnormal   
    │   │   ├── image_001.jpg
    │   │   └── image_002.jpg   
    │   └── normal
    │       ├── image_001.jpg
    │       └── image_002.jpg
    ├── validation/
    │   ├── abnormal   
    │   │   └── image_003.jpg
    │   └── normal
    │       └── image_003.jpg
    └── test/
        ├── abnormal   
        │   └── image_004.jpg
        └── normal
            └── image_004.jpg
```

**Configuration Example:**
```yaml
data_params:
    dataset_path: "../datasets/dataset_name"
    images_extension: "jpg"
    data_preprocessing:
        data_mode: "rgb"                        # "rgb" or "grayscale"
        data_parser: "folder"                 
        resize_dims: [128, 128]
        medmnist_csv_file:  None
        label_mapping:
            abnormal: 1
            normal: 0
```

**Key Points:**
- Requires a consistent organization across train/validation/test splits
- The validation folder name must be exactly "validation" (not "val" or other variants)
- Supported image formats include jpg, jpeg, png, etc. supported by the parser
- Class names defined in label_mapping must match the class folder names

#### 2.3 MedMNIST Benchmark Structure

```
datasets/
├── pneumoniamnist/
│   ├── pneumoniamnist.csv   
│   ├── test_0_0.png   
│   ├── test_1_1.png   
│   ├── train_0_1.png   
│   ├── train_1_0.png   
│   ├── train_2_0.png   
│   ├── train_3_1.png   
│   ├── val_0_1.png   
│   └── val_1_0.png  
```

**Configuration Example:**
```yaml
data_params:
    dataset_path: "../datasets/pneumoniamnist"
    images_extension: "png"
    data_preprocessing:
        data_mode: "rgb"                        # "rgb" or "grayscale"
        data_parser: "folder"                 
        resize_dims: [128, 128]
        medmnist_csv_file:  "../datasets/pneumoniamnist/pneumoniamnist.csv"
        label_mapping:
            pneumonia: 1
            normal: 0
```

**Key Points:**
- Supports any 2D dataset included in the MedMNIST Benchmark collection.
- Requires the data in the original format provided by the [MedMNIST benchmark](https://github.com/MedMNIST/MedMNIST/tree/main)
- The CSV file should be used as downloaded, without modification.
- Class names defined in label_mapping must match the official class names for the corresponding dataset provided in [info](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)

### 3. EPU-CNN Training
To train EPU-CNN, run one of the following commands after setting up a `config.yaml` file:
```bash

# Basic training
python scripts/train_epu.py --config_filepath configs/train_config.yaml

# Training with TensorBoard monitoring
python scripts/train_epu.py --config_filepath configs/train_config.yaml --tensorboard
```
The script saves trained model checkpoints and YAML training config in results/checkpoints under automatically generated name as:
{experiment_name}_{subnetwork_backbone}_{run_id}_{timestamp}

When using the `--tensorboard` flag, the script automatically:
- launches TensorBoard as a subprocess
- sets up monitoring for the `logs` directory
- makes TensorBoard available at `http://localhost:6006`
- enables real-time monitoring of training metrics and model graphs.

### 4. EPU-CNN Evaluation
To assess the classification performance of a trained EPU-CNN model run:
```bash
python scripts/infer_epu.py --experiment_folder_name ius_dataset_name_base_one_0000_timestamp
```
 - Argument --experiment_folder_name contains the saved checkpoint and saved YAML config used during the EPU-CNN training.
 - The classification performance report is saved in JSON format under results/classification_performance/{experiment_folder_name}

### 5. Feature contribution profile estimation
For each trained EPU-CNN model instance, the baseline feature contribution profile must be estimated only once using ether:
```bash
# Estimates all baseline feature contribution profiles  
python scripts/infer_cb_vector.py --experiment_folder_name ius_dataset_name_base_one_0000_timestamp

# Estimates the baseline feature contribution profile of a single class
python scripts/infer_cb_vector.py --experiment_folder_name ius_dataset_name_base_one_0000_timestamp --cb_data normal
```
- The --experiment_folder_name contains the saved checkpoint and saved YAML config used during the EPU-CNN training.
- If argument --cb_data is not provided, the script estimates the baseline feature contribution profile for each class defined in saved config YAML (automatically retrieved from results/checkpoints based on --experiment_folder_name)

### 6. Synthetic image utility evaluation with IUS
To evaluate the utility of synthetic images using IUS, run:
```bash
# For synthetic dataset IUS evaluation
python scripts/eval_ius.py --experiment_folder_name ius_dataset_name_base_one_0000_timestamp --cb_vector_tag normal --synthetic_images datasets_synthetic/dataset_name/normal --synthetic_img_extension png

# For a single image IUS evaluation
python scripts/eval_ius.py --experiment_folder_name ius_dataset_name_base_one_0000_timestamp --cb_vector_tag normal --synthetic_images datasets_synthetic/dataset_name/normal/seed_000.png --synthetic_img_extension png
```

Both commands produce two outputs saved under results/ius_eval/{experiment_folder_name}
  1. A JSON report containing information about IUS evaluation.  
  2. A CSV file containing the estimated IUS score for  each synthetic image.

**Key Points:**
- Synthetic data should be stored under datasets_synthetic
- No specific internal folder structure is required; only a valid path to a folder of images is needed.
- A suggested organization example is provided below 
- Supported image formats include jpg, jpeg, png, etc, supported by the parser
- Class labels are neither required nor used during  IUS evaluation

Suggested synthetic data structure
```
datasets_synthetic/
├── dataset_name/
    ├── normal/
        ├── seed_000.png
        └── seed_001.png
```

## 📄 Paper Abstract
<p align="justify">
Synthetic medical image data can unlock the potential of deep learning (DL)-based clinical decision support (CDS) systems through the creation of large scale, privacy-preserving, training sets. Despite the significant progress in this field, there is still a largely unanswered research question: “How can we quantitatively assess the similarity of a synthetically generated set of images with a set of real images in a given application domain?”. Today, answers to this question are mainly provided via user evaluation studies, inception-based measures, and the classification performance achieved on synthetic images. This paper proposes a novel measure to assess the similarity between synthetically generated and real sets of images, in terms of their utility for the development of DL based CDS systems. Inspired by generalized neural additive models, and unlike inception-based measures, the proposed measure is interpretable (Interpretable Utility Similarity, IUS), explaining why a synthetic dataset could be more useful than another one in the context of a CDS system based on clinically relevant image features. The experimental results on publicly available benchmark datasets from various color medical imaging modalities including endoscopic, dermoscopic and fundus imaging, indicate that selecting synthetic images of high utility similarity using IUS can result in relative improvements of up to 54.6% in terms of classification performance. The generality of IUS for synthetic data assessment is demonstrated also for grayscale X-ray and ultrasound imaging modalities. IUS implementation is available at https://github.com/innoisys/ius. 
</p>

## Citation
If you use this code or find our work useful in your research, please cite:

**APA**

P. Gatoula, G. Dimas and D. K. Iakovidis, "Interpretable Similarity of Synthetic Image Utility," in *IEEE Transactions on Medical Imaging*, doi: 10.1109/TMI.2026.3679527. 

**BibTeX**
```bibtex
@article{
  author    = {Panagiota Gatoula and George Dimas and Dimitris K. Iakovidis},
  title     = {Interpretable Similarity of Synthetic Image Utility},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/TMI.2026.3679527}
}
```

## 📧 Contact
Prof. Dimitris Iakovidis
Director of Biomedical Imaging Lab
University of Thessaly
diakovidis@uth.gr

## ⚖️  License
This project is licensed under the MIT License.

