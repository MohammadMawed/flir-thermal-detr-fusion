# 🔬 Deep Dive: Selective Multi-Scale Fusion with DETR for Thermal Object Detection

## 1. Motivation

Thermal perception challenges:
- Grayscale only -> no color cues and weak texture.
- High scale variation -> far pedestrians are tiny, nearby vehicles are large.
- Vanilla DETR uses a single stride-32 feature map, making it scale-blind.

Why not FPN or Deformable DETR?
- They solve scale issues but add complexity and deviate from DETR's simplicity.

Goal: keep DETR's end-to-end design while injecting scale-awareness for thermal detection.

## 2. Method
### a. Preprocessing
- Input: grayscale thermal images.
- Trick: replicate 1 channel to 3 and keep ImageNet normalization to leverage ResNet-50 pretrained on RGB COCO.
- Augmentation: random horizontal flips, multi-scale resize jitter (short side 480–800 px), slight intensity jitter.

### b. Backbone Feature Extraction
- Base: ResNet-50 pretrained on COCO.
- Harvest features at strides 8 (C3), 16 (C4), 32 (C5).
- Add C6 (stride 64) by downsampling C5 with a 3x3 stride-2 conv.
- Project all feature maps to 256 channels via 1x1 conv.
- Bilinearly resize all maps to a stride-16 grid for alignment.

### c. Selective Multi-Scale Fusion Module
- Each image gets its own fusion weights.
1. Global query q: average-pooled concatenation of all aligned maps -> 1x1 conv.
2. Keys per scale k_s: average-pooled F̃_s -> 1x1 conv.
3. Attention weights: alpha_s = exp(<q, k_s>/C) / sum_t exp(<q, k_t>/C)
4. Multiply by learnable per-scale bias w_s and normalize:
   alpha_hat_s = (alpha_s * softmax(w)_s) / sum_t (alpha_t * softmax(w)_t)
- Fused map: F_fuse = Conv3x3(sum_s alpha_hat_s F̃_s)
- Interpretation: model learns which scales matter per image.

### d. Scale-Aware Positional Encoding
- Fusion hides explicit scale identity.
- Add learned offsets beta_s and fuse: beta_bar = sum_s alpha_hat_s beta_s

### e. Transformer + Prediction Heads
- Flatten fused map to HxW tokens and add positional encodings.
- Standard DETR encoder-decoder with Q object queries.
- Heads: classification (K+1 classes including "no object") and bounding-box regression (normalized center and width/height).

### f. Training Objective
- Hungarian matching between predictions and ground truth.
- Loss: L = lambda_cls * L_CE + lambda_l1 * |b - b_hat|_1 + lambda_gIoU * (1 - gIoU(b, b_hat))
- Settings: AdamW (LR 1e-5 backbone, 1e-4 fusion/transformer/heads), gradient clipping, mixed precision, 50 epochs (LR drop x10 after epoch 40), effective batch size 16 (batch 4, accumulation 4), ~6.5 h on a single A100.

## 3. Results on FLIR ADAS
### a. Overall Results
| Method | mAP@0.50:0.95 | mAP@0.50 | mAPsmall |
| --- | --- | --- | --- |
| DETR baseline | 0.19 | 0.32 | 0.15 |
| Fusion DETR | 0.23 | 0.37 | 0.17 |

- +0.05 gain in mAP@0.50
- +0.02 gain for small objects

### b. Per-Class Performance
| Class | Baseline | Fusion DETR | Δ |
| --- | --- | --- | --- |
| Person | 0.28 | 0.34 | +0.06 |
| Bike | 0.22 | 0.28 | +0.06 |
| Vehicle | 0.40 | 0.42 | +0.02 |

- Biggest gains for small-scale classes (person, bike).

### c. Qualitative Observations
- Clearer detections for pedestrians and cyclists at distance.
- Some extremely small objects still missed.
- Fusion helps remove duplicates -> better precision.

## 4. Key Insights
- Minimal change, strong gains: tiny fusion module and scale-aware positional encoding boost performance while preserving DETR simplicity.
- Thermal-only, no RGB needed: robust at night and in adverse weather.
- Best for small objects: pedestrians and bikes benefit most.
- Simplicity preserved: encoder/decoder untouched, no NMS, reproducible with standard DETR recipe.

## 5. Limitations
- Struggles with very small (<10 px) objects.
- Fusion weights are global per image -> may miss local scale differences (e.g., near and far pedestrians in one frame).
- Dataset mapping collapsed 8 classes into 3, under-representing rare objects.

## 6. Future Work
- Finer fusion granularity: per-region or per-query scale weighting.
- Co-pretrain on KAIST Multispectral Pedestrian Dataset (thermal-only channel).
- Evaluate stronger backbones (e.g., Swin Transformer, ConvNeXt) with the same fusion.
- Incorporate temporal modeling for video tracking.

## ✅ Bottom Line
A lightweight selective fusion module plus scale-aware positional encoding makes DETR significantly more effective for thermal-only detection, especially for small safety-critical targets, while keeping inference fast, stable, and deployable.

