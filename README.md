# Image_Processing_Final_Project_Noayelet
 Evaluating the robustness of vision models under distortions for BDD100K dataset.


# Evaluating the Robustness of Computer Vision Models

**Project by Ayelet & Noa**

> **Project status:** Work in progress. This README currently contains the completed experimental sections. The remaining deep-learning extension will be added in a future update.

## 1. Project Objective

The primary objective of this project is to evaluate the robustness of classical image-processing algorithms and deep-learning-based computer vision models. In real-world applications, computer vision systems must operate in dynamic environments and handle changing, non-ideal imaging conditions, including poor illumination, sensor noise, and degradation caused by data compression. These factors may distort the input, alter the model output, and significantly reduce the reliability of the overall system.

To examine this challenge in a realistic and demanding setting, we used the **BDD100K** dataset, a large-scale collection of road-scene images captured by vehicle-mounted cameras. The dataset provides a suitable test environment for measuring model performance under three deliberately introduced image distortions.

After establishing each model's performance on clean images as a baseline, we measured the degradation caused by each distortion and evaluated two recovery strategies:

1. **Classical image enhancement**, applied as a preprocessing stage before model inference.
2. **Model fine-tuning**, in which a model is retrained on distorted data so that it can learn more robust internal representations.

The project therefore investigates not only how rapidly different models degrade, but also whether image-level restoration or model-level adaptation provides the more effective recovery strategy.

## 2. Selected Tasks and Models

To obtain a broad and meaningful comparison, we selected three computer vision tasks that represent substantially different levels of image analysis and scene understanding.

### Task 1: Object Detection with YOLOv8

Object detection performs classification and localization at the object level using bounding boxes. We selected **YOLOv8** to examine how a modern real-time detector handles the localization of discrete, safety-critical road elements, such as vehicles and pedestrians, under uncertain and degraded imaging conditions.

### Task 2: Instance Segmentation with Mask R-CNN

Instance segmentation performs pixel-level classification while separating individual object instances and outlining their precise boundaries. We selected **Mask R-CNN** to determine whether dense, fine-grained separation of objects from the background degrades differently from the coarser bounding-box-based localization used in object detection.

### Task 3: Monocular Depth Estimation with MiDaS

Monocular depth estimation is a continuous regression task that extracts relative three-dimensional scene structure from a single two-dimensional image. We selected **MiDaS** because depth estimation depends strongly on structural cues such as object contours, shadows, surface continuity, and linear perspective. This task is therefore particularly sensitive to lighting changes, damaged edges, and the loss of smooth spatial information.

## 3. Model-Selection Rationale

The models were selected to represent a broad range of modern computer vision architectures and feature-extraction mechanisms. Our working assumption was that each architecture would exhibit a different vulnerability profile under the tested distortions.

### Why YOLOv8?

YOLOv8 was selected to represent the family of **single-stage detectors** designed for real-time operation, as required in applications such as autonomous driving. Its anchor-free detection mechanism and efficient feature-extraction pipeline prioritize speed and practical deployment. This choice allowed us to investigate whether a model optimized for fast inference is especially sensitive to noise, low illumination, or compression artifacts that interfere with spatial features and object boundaries.

### Why Mask R-CNN?

Mask R-CNN was selected as a well-established, high-accuracy model for tasks requiring precise localization. Unlike YOLOv8, it follows a **two-stage architecture**: it first generates region proposals and then analyzes the selected regions at a higher resolution to classify objects and produce instance masks. This architecture provides a useful contrast to YOLOv8 and enables us to test whether two-stage processing and dense pixel-level predictions provide greater robustness under degraded visual conditions.

### Why MiDaS?

The central challenge in monocular depth estimation is its dependence on structural cues such as contours, shadows, relative scale, texture gradients, and linear perspective. MiDaS was selected because it is designed to generalize depth relationships across diverse visual domains and datasets. We therefore used it to examine how a model that estimates continuous scene geometry responds when the visual cues supporting that geometry are deliberately corrupted.

## 4. Selected Distortions

We selected three distortions representing different physical and digital sources of image degradation.

### 4.1 Low-Light Distortion

Low-light distortion reduces image brightness while largely preserving the original geometric arrangement of the pixels. Its purpose is to evaluate model robustness under severe illumination changes, such as nighttime driving, twilight, underexposure, or insufficient camera exposure.

### 4.2 Gaussian Noise

Gaussian noise is a high-frequency distortion that adds random intensity variations across the image. It is used to approximate sensor noise and to test how the models respond when smooth surfaces, textures, and object boundaries are disrupted by random pixel-level interference.

### 4.3 JPEG Compression

JPEG compression is a lossy digital distortion that removes image information and may introduce blocking, color quantization, and artificial boundaries at aggressive compression levels. It enables us to investigate how the models respond when the original image geometry and fine visual details are damaged by compression artifacts.

## 5. Experimental Pipeline and Methodology

For each task and model, we followed a consistent experimental pipeline:

1. **Baseline evaluation:** The pretrained model was evaluated on clean images to establish a reference performance level.
2. **Distortion-intensity sweep:** Each distortion was applied at several increasing intensity levels to measure the model's rate of degradation and identify possible failure points.
3. **Classical enhancement:** Image-processing techniques, including methods such as CLAHE, bilateral filtering, Gaussian smoothing, and other distortion-specific filters, were applied before inference in an attempt to recover useful visual information.
4. **Fine-tuning:** Where applicable, the model was retrained on a dataset containing distorted samples so that it could learn to handle the degradation directly rather than relying only on external preprocessing.
5. **Evaluation and comparison:** The baseline, distorted, enhanced, and fine-tuned results were compared using task-appropriate quantitative metrics and qualitative visual analysis.

The following sections present each task in detail, including the baseline results, distortion sweeps, model degradation, classical enhancement methods, fine-tuning experiments where completed, and the main conclusions drawn from the results.
