# Preliminary results (quick protocol)

Settings: 1000 training images (RESIDE-6K outdoor, first by name), one seed (two for the DehazeFormer-T R0/R3 comparison), iterations per run: DehazeFormer-T 3500, gUNet-T 8400, DehazeFormer-B 2100, DEA-Net 4600. Fine-tuned models start from the official RESIDE-OTS weights. Paired test sets are complete (45 / 55 / 55 images, long side 1024); RTTS uses a fixed subset of 500 images (quality metrics on 100). Total budget 8.0 h on one GPU.

Reading rule (fixed before the runs): a PSNR difference below 0.3 dB is treated as noise; a difference counts when its 95 % paired-bootstrap interval over the test images excludes zero and, where two seeds exist, it is also larger than twice the seed-to-seed standard deviation. Short schedules and one or two seeds make these preliminary indications, not final claims.

## 1. Does forming the haze through the camera help? (R3 minus R0, same network, images and haze)

| Model | Training | O-HAZE dPSNR [95 % CI] | NH-HAZE dPSNR [95 % CI] | Dense-Haze dPSNR [95 % CI] | RTTS dmAP50 | Seeds |
|---|---|---|---|---|---|---|
| DehazeFormer-T | from scratch | +0.39 [+0.03, +0.75] **(+)** | +0.01 [-0.11, +0.13] | -0.31 [-0.70, +0.02] | -0.1 | 2 (seed std 0.30 dB) |
| gUNet-T | from scratch | -0.18 [-0.62, +0.23] | -0.02 [-0.13, +0.09] | -0.29 [-0.61, +0.02] | +0.3 | 1 |
| DehazeFormer-B | fine-tuned official weights | +0.63 [+0.10, +1.15] **(+)** | +0.24 [+0.09, +0.42] | -0.13 [-0.39, +0.13] | +0.9 | 1 |
| DEA-Net | fine-tuned official weights | -0.62 [-1.17, -0.09] **(-)** | -0.09 [-0.34, +0.20] | -0.57 [-1.06, -0.08] **(-)** | +0.1 | 1 |

**(+)** / **(-)**: passes the reading rule in favour of / against our data. dPSNR and dmAP50 are R3 minus R0 (positive = our data is better). RTTS dmAP50 has no interval (one number per run).

**Summary:** our data (R3) gives a higher PSNR than standard sRGB synthesis (R0) in 4 of 12 model x test-set comparisons; 2 are clear gains, 2 are clear losses, 8 are within noise.

## 1b. Each step of the synthesis (DehazeFormer-T, from scratch, PSNR difference [95 % CI])

| Step | What changes | O-HAZE | NH-HAZE | Dense-Haze |
|---|---|---|---|---|
| R0 - B0 | our physically grounded haze (sRGB) vs RESIDE's own hazy images | +1.38 [+0.93, +1.83] **(+)** | +1.02 [+0.75, +1.31] **(+)** | +1.33 [+0.89, +1.78] **(+)** |
| R0_uncal - B0 | same, without matching the haze density to real haze (V 50 m-5 km, no rejection) | -0.16 [-0.50, +0.19] | +0.08 [-0.05, +0.22] | -0.28 [-0.56, -0.02] |
| R0_rtts - B0 | our haze calibrated ONLY on RTTS (disjoint unpaired real hazy set) vs RESIDE | -0.08 [-0.44, +0.28] | +0.19 [-0.02, +0.41] | +0.45 [+0.02, +0.89] **(+)** |
| R1_rtts - B0 | same with linear compositing (R1), calibrated only on RTTS, vs RESIDE | -0.05 [-0.42, +0.34] | +0.16 [-0.08, +0.44] | +0.61 [+0.15, +1.09] **(+)** |
| R0_rtts - R0_uncal | calibrated on RTTS vs not calibrated at all | +0.08 [-0.24, +0.41] | +0.11 [-0.06, +0.30] | +0.74 [+0.44, +1.03] **(+)** |
| R0 - N0 | sky-coloured airlight vs grey airlight (same depth and visibility) | +2.38 [+1.92, +2.87] **(+)** | +1.48 [+1.18, +1.81] **(+)** | +2.04 [+1.50, +2.63] **(+)** |
| N0 - B0 | our depth + visibility alone (grey airlight) vs RESIDE | -1.00 [-1.25, -0.75] **(-)** | -0.46 [-0.60, -0.32] **(-)** | -0.71 [-1.10, -0.33] **(-)** |
| R1 - R0 | + linear-radiance compositing | +0.91 [+0.65, +1.17] **(+)** | +0.34 [+0.22, +0.46] **(+)** | -0.10 [-0.30, +0.08] |
| R2 - R1 | + sensor noise after the haze | -0.81 [-1.01, -0.61] **(-)** | -0.34 [-0.48, -0.21] **(-)** | -0.49 [-0.84, -0.17] **(-)** |
| R3 - R2 | + randomised ISP and JPEG | +0.29 [+0.06, +0.52] | +0.01 [-0.08, +0.10] | +0.29 [+0.14, +0.44] |
| R3N - R1 | + randomised ISP and JPEG, without noise | -0.01 [-0.22, +0.20] | -0.08 [-0.16, -0.01] | +0.01 [-0.14, +0.17] |
| R3 - R3N | effect of the sensor noise inside the full camera | -0.52 [-0.68, -0.35] **(-)** | -0.25 [-0.34, -0.15] | -0.22 [-0.55, +0.05] |
| R3 - B0 | full pipeline vs RESIDE's own hazy images | +1.76 [+1.18, +2.35] **(+)** | +1.03 [+0.77, +1.31] **(+)** | +1.03 [+0.58, +1.45] **(+)** |
| R1 - B0 | best variant (R1) vs RESIDE's own hazy images | +2.29 [+1.72, +2.87] **(+)** | +1.36 [+1.10, +1.63] **(+)** | +1.23 [+0.82, +1.64] **(+)** |

Same network, same 1,000 clean images, same schedule and seed(s); only the way the hazy training images are formed differs. B0 uses RESIDE's own hazy versions of the same clean images.

## 1c. The same comparisons on the other networks (PSNR difference [95 % CI])

| Network | Comparison | O-HAZE | NH-HAZE | Dense-Haze |
|---|---|---|---|---|
| gUNet-T (from scratch) | R0 - B0 | +1.72 [+1.20, +2.28] **(+)** | +1.19 [+0.93, +1.46] **(+)** | +0.28 [-0.11, +0.68] |
| gUNet-T (from scratch) | R1 - R0 | +0.34 [+0.08, +0.62] **(+)** | -0.05 [-0.17, +0.05] | +0.50 [+0.37, +0.63] **(+)** |
| gUNet-T (from scratch) | R1 - B0 | +2.06 [+1.60, +2.56] **(+)** | +1.14 [+0.93, +1.35] **(+)** | +0.78 [+0.38, +1.19] **(+)** |
| gUNet-T (from scratch) | R3 - B0 | +1.54 [+0.97, +2.08] **(+)** | +1.17 [+0.89, +1.46] **(+)** | -0.01 [-0.48, +0.44] |
| gUNet-T (from scratch) | R1_rtts - B0 | +0.58 [+0.14, +1.05] **(+)** | +0.63 [+0.38, +0.89] **(+)** | +0.13 [-0.23, +0.48] |
| DehazeFormer-B (fine-tuned) | R0 - B0 | +1.73 [+1.41, +2.07] **(+)** | +0.99 [+0.80, +1.20] **(+)** | +1.10 [+0.81, +1.39] **(+)** |
| DehazeFormer-B (fine-tuned) | R1 - R0 | +0.17 [-0.16, +0.49] | -0.13 [-0.23, -0.03] | +0.17 [-0.01, +0.35] |
| DehazeFormer-B (fine-tuned) | R1 - B0 | +1.90 [+1.44, +2.36] **(+)** | +0.87 [+0.68, +1.08] **(+)** | +1.27 [+0.90, +1.64] **(+)** |
| DehazeFormer-B (fine-tuned) | R3 - B0 | +2.36 [+1.77, +2.96] **(+)** | +1.24 [+0.97, +1.52] **(+)** | +0.97 [+0.57, +1.39] **(+)** |
| DEA-Net (fine-tuned) | R0 - B0 | +1.85 [+1.42, +2.31] **(+)** | +0.67 [+0.52, +0.84] **(+)** | +0.31 [-0.01, +0.64] |
| DEA-Net (fine-tuned) | R1 - R0 | -1.31 [-1.80, -0.86] **(-)** | -0.54 [-0.75, -0.34] **(-)** | -1.88 [-2.26, -1.51] **(-)** |
| DEA-Net (fine-tuned) | R1 - B0 | +0.53 [-0.08, +1.15] | +0.13 [-0.09, +0.37] | -1.57 [-2.05, -1.09] **(-)** |
| DEA-Net (fine-tuned) | R3 - B0 | +1.22 [+0.56, +1.88] **(+)** | +0.58 [+0.30, +0.90] **(+)** | -0.26 [-0.73, +0.23] |

For fine-tuned networks, B0 means the official weights fine-tuned on RESIDE's own hazy images for the same number of iterations: it separates the effect of our data from the effect of extra training.

## 2. Image-formation ladder (DehazeFormer-T, from scratch)

| Arm | Synthesis | O-HAZE PSNR | SSIM | NH-HAZE PSNR | SSIM | Dense-Haze PSNR | SSIM | RTTS mAP50 | MUSIQ (higher = better) | BRISQUE (lower = better) | mean dPSNR vs R0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | RESIDE-OTS pairs (official synthesis) | 15.88 | 0.673 | 12.08 | 0.483 | 10.79 | 0.422 | 71.6 | 54.09 | 21.62 | -1.24 |
| N0 | sRGB ASM, grey airlight (our depth, V) | 14.87 | 0.701 | 11.62 | 0.518 | 10.08 | 0.426 | -- | -- | -- | -1.97 |
| R0 | sRGB ASM, sky-coloured airlight | 17.25 | 0.743 | 13.10 | 0.556 | 12.12 | 0.459 | 71.4 | 53.56 | 19.62 | -- |
| R1 | + linear-radiance compositing | 18.17 | 0.757 | 13.44 | 0.573 | 12.02 | 0.458 | 71.3 | 52.97 | 18.85 | +0.38 |
| R2 | + post-haze shot/read noise | 17.35 | 0.737 | 13.09 | 0.551 | 11.53 | 0.460 | 71.0 | 51.07 | 17.11 | -0.17 |
| R3 | + randomised ISP + JPEG (full camera) | 17.64 | 0.734 | 13.11 | 0.541 | 11.82 | 0.464 | 71.4 | 51.79 | 16.89 | +0.03 |
| R3N | R3 without sensor noise (linear + ISP + JPEG) | 18.16 | 0.750 | 13.36 | 0.561 | 12.03 | 0.461 | -- | -- | -- | +0.36 |
| H | sRGB ASM + heterogeneous beta | 15.47 | 0.708 | 12.52 | 0.536 | 9.72 | 0.423 | 71.7 | 54.21 | 19.87 | -1.59 |
| R3H | full camera + heterogeneous beta | 15.46 | 0.698 | 12.92 | 0.534 | 9.59 | 0.418 | 71.5 | 52.90 | 17.90 | -1.50 |

Each row adds one step of the camera model to the row above (R0 -> R1 -> R2 -> R3). B0 is RESIDE's own hazy images for the same clean images. Rows with two seeds show the mean.

## 3. Published models and ours under one protocol

| Method | Training data | O-HAZE PSNR | SSIM | NH-HAZE PSNR | SSIM | Dense-Haze PSNR | SSIM | RTTS mAP50 | MUSIQ (higher = better) | BRISQUE (lower = better) |
|---|---|---|---|---|---|---|---|---|---|---|
| Hazy input | -- | 13.78 | 0.619 | 11.48 | 0.451 | 9.61 | 0.404 | 71.7 | 54.59 | 34.57 |
| DCP | none | 16.07 | 0.677 | 12.31 | 0.473 | 12.54 | 0.425 | 71.9 | 52.25 | 30.30 |
| DehazeFormer-T | RESIDE-OTS | 16.08 | 0.681 | 12.17 | 0.488 | 11.24 | 0.436 | 72.0 | 54.84 | 31.10 |
| DehazeFormer-B | RESIDE-OTS | 16.20 | 0.683 | 12.18 | 0.486 | 11.23 | 0.435 | 72.1 | 54.76 | 32.40 |
| DEA-Net | RESIDE-OTS | 16.67 | 0.705 | 12.37 | 0.506 | 11.57 | 0.443 | 71.7 | 54.96 | 29.64 |
| HazeFlow (published outputs) | RIDCP500 + real unpaired (URHI/RTTS) | -- | -- | -- | -- | -- | -- | 71.3 | 64.58 | 4.26 |
| Baseline: DehazeFormer-T trained on RESIDE pairs (B0) | 1000 OTS pairs | 15.88 | 0.673 | 12.08 | 0.483 | 10.79 | 0.422 | 71.6 | 54.09 | 21.62 |
| Baseline: DehazeFormer-T trained on R0 | 1000 OTS clean images | 17.25 | 0.743 | 13.10 | 0.556 | 12.12 | 0.459 | 71.4 | 53.56 | 19.62 |
| **Ours**: DehazeFormer-T trained on R3 | 1000 OTS clean images | 17.64 | 0.734 | 13.11 | 0.541 | 11.82 | 0.464 | 71.4 | 51.79 | 16.89 |
| Baseline: gUNet-T trained on R0 | 1000 OTS clean images | 18.11 | 0.764 | 13.36 | 0.583 | 11.09 | 0.471 | 71.3 | 54.91 | 19.73 |
| **Ours**: gUNet-T trained on R3 | 1000 OTS clean images | 17.93 | 0.765 | 13.34 | 0.574 | 10.79 | 0.475 | 71.6 | 53.97 | 20.37 |
| Baseline: DehazeFormer-B, official weights fine-tuned on R0 | 1000 OTS clean images + official | 17.98 | 0.746 | 13.21 | 0.550 | 12.31 | 0.474 | 71.9 | 54.97 | 26.84 |
| **Ours**: DehazeFormer-B, official weights fine-tuned on R3 | 1000 OTS clean images + official | 18.61 | 0.765 | 13.45 | 0.562 | 12.18 | 0.495 | 72.8 | 54.32 | 29.79 |
| Baseline: DEA-Net, official weights fine-tuned on R0 | 1000 OTS clean images + official | 18.69 | 0.756 | 13.11 | 0.547 | 11.79 | 0.460 | 72.0 | 54.86 | 24.64 |
| **Ours**: DEA-Net, official weights fine-tuned on R3 | 1000 OTS clean images + official | 18.07 | 0.760 | 13.02 | 0.558 | 11.21 | 0.465 | 72.1 | 54.05 | 17.27 |

HazeFlow is scored from its published RTTS outputs (no paired-set outputs are published); it was trained with real unpaired hazy images, RTTS listed among them. Models trained here ran for minutes, the published ones for days: compare rows 'trained on R0' and 'trained on R3' for the effect of the data.

## 4. Camera versus heterogeneity (cost in PSNR of removing each from R3H)

| Test set | remove camera (R3H -> H) | remove heterogeneity (R3H -> R3) | reading |
|---|---|---|---|
| O-HAZE | -0.01 [-0.20, +0.17] | -2.18 [-2.78, -1.62] | camera no clear effect; heterogeneity hurts |
| NH-HAZE | +0.41 [+0.28, +0.54] | -0.18 [-0.43, +0.06] | camera helps; heterogeneity no clear effect |
| Dense-Haze | -0.13 [-0.29, +0.01] | -2.23 [-2.59, -1.87] | camera no clear effect; heterogeneity hurts |

Positive = removing the component lowers PSNR (the component helps). NH-HAZE was declared in advance as the set where heterogeneity may matter more.

