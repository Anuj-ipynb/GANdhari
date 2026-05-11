```text
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ ██████╗ ██╗██╗  ██╗██████╗ ██╗██╗  ██╗     ██████╗██╗████████╗██╗   ██╗███████╗ ██████╗ █████╗ ██████╗ ███████╗ ║
║ ██╔══██╗██║╚██╗██╔╝╚════██╗██║╚██╗██╔╝    ██╔════╝██║╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝ ║
║ ██████╔╝██║ ╚███╔╝  █████╔╝██║ ╚███╔╝     ██║     ██║   ██║    ╚████╔╝ ███████╗██║     ███████║██████╔╝█████╗   ║
║ ██╔═══╝ ██║ ██╔██╗ ██╔═══╝ ██║ ██╔██╗     ██║     ██║   ██║     ╚██╔╝  ╚════██║██║     ██╔══██║██╔═══╝ ██╔══╝   ║
║ ██║     ██║██╔╝ ██╗███████╗██║██╔╝ ██╗    ╚██████╗██║   ██║      ██║   ███████║╚██████╗██║  ██║██║     ███████╗ ║
║ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝     ╚═════╝╚═╝   ╚═╝      ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝ ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

AI-powered semantic-to-real urban scene generation with sustainability-aware infrastructure analysis.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-GAN-red?style=flat-square)
![Pix2Pix](https://img.shields.io/badge/Model-Pix2Pix-purple?style=flat-square)
![Cityscapes](https://img.shields.io/badge/Dataset-Cityscapes-green?style=flat-square)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![CUDA](https://img.shields.io/badge/GPU-GTX1050Ti-success?style=flat-square)
![GAN](https://img.shields.io/badge/Task-Image_Translation-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

# What Is This?

Pix2Pix CityScape is a semantic image translation and urban sustainability analysis system built using a lightweight Pix2Pix GAN architecture trained on the Cityscapes dataset. The project converts semantic urban layouts into realistic synthetic city imagery while simultaneously computing infrastructure-oriented sustainability metrics such as vegetation ratio, building density, road connectivity, and urban balance.

The system is designed as both a computer vision research project and a decision-support prototype for smart-city visualization workflows. Instead of focusing purely on photorealistic generation, the architecture prioritizes structural correctness and urban feature preservation. This allows the generated outputs to remain interpretable for downstream sustainability analysis.

The implementation is optimized for constrained hardware environments using a lightweight U-Net generator configuration (`ngf = 32`) and reduced VRAM requirements compatible with consumer GPUs such as the GTX 1050 Ti. The pipeline integrates semantic generation, post-processing, metric extraction, and interactive visualization into a single deployable workflow.

> The project treats GAN-generated imagery as structured urban simulation data rather than purely aesthetic image synthesis.

---

# Why Pix2Pix?

| Challenge | Classical Image Processing | Pix2Pix GAN Approach |
|---|---|---|
| Semantic-to-image conversion | Rule-based rendering | Learned conditional generation |
| Urban realism | Static texture mapping | Context-aware synthesis |
| Infrastructure understanding | Hardcoded geometry | Learned spatial relationships |
| Sustainability analysis | Manual GIS calculations | Automated computer vision metrics |
| Layout interpretation | Pixel heuristics | Semantic feature learning |
| Low-resource deployment | Heavy rendering engines | Lightweight GAN inference |

---

# System Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    PIX2PIX CITYSCAPE                      │
└────────────────────────────────────────────────────────────┘

        Semantic Segmentation Map
                      │
                      ▼
┌──────────────────────────────────────────┐
│ data/loader.py                           │
│ - Cityscapes preprocessing               │
│ - Resize and normalization               │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ models/generator.py                      │
│ Lightweight U-Net Generator              │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ models/discriminator.py                  │
│ PatchGAN discriminator                   │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ inference/postprocess.py                 │
│ - Enhancement                            │
│ - Artifact reduction                     │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌────────────────────┐
│ utils/metrics.py│  │ utils/sustain.py   │
│ SSIM / L1 / Edge│  │ Urban sustainability│
└────────┬────────┘  └──────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
┌──────────────────────────────────────────┐
│ app.py                                   │
│ Gradio interactive dashboard             │
└──────────────────────────────────────────┘
```

---

# Generator Architecture

The generator follows a lightweight U-Net encoder-decoder configuration optimized for constrained GPU environments.

## Generator Specifications

| Component | Configuration |
|---|---|
| Architecture | U-Net |
| Base Filters (`ngf`) | 32 |
| Input Resolution | 256×256 |
| Skip Connections | Enabled |
| Activation | ReLU / LeakyReLU |
| Output Activation | Tanh |

---

# Discriminator Architecture

The discriminator uses a PatchGAN configuration that evaluates local image realism instead of global image classification.

## Why PatchGAN?

PatchGAN improves:

- edge consistency
- texture realism
- local structural alignment
- training stability

while reducing computational overhead.

---

# Training Pipeline

```text
Semantic Map
      ↓
Generator Forward Pass
      ↓
Synthetic Urban Image
      ↓
PatchGAN Evaluation
      ↓
GAN Loss + L1 Loss
      ↓
Backpropagation
      ↓
Weight Update
```

---

# Loss Function

The training objective combines adversarial learning with structural preservation.

## Objective Function

```latex
L_total = L_GAN + λL_{L1}
```

Where:

| Loss Component | Purpose |
|---|---|
| GAN Loss | Realism generation |
| L1 Loss | Structural consistency |
| λ | Balancing coefficient |

---

# Sustainability Analysis Pipeline

The generated urban scene is processed using computer vision metrics to estimate urban sustainability indicators.

```text
Generated Image
        ↓
Semantic Feature Extraction
        ↓
Coverage Analysis
        ↓
Connectivity Analysis
        ↓
Weighted Sustainability Score
```

---

# Sustainability Metrics

| Metric | Description |
|---|---|
| Green Coverage | Percentage of vegetation |
| Building Density | Built-up infrastructure ratio |
| Road Coverage | Road surface occupancy |
| Road Connectivity | Road continuity estimation |
| Sustainability Score | Weighted urban quality indicator |

---

# Sustainability Scoring Logic

The sustainability score is influenced by urban structural balance.

## Positive Contributors

- higher vegetation ratio
- connected road layouts
- balanced infrastructure spacing

## Negative Contributors

- excessive building density
- low vegetation presence
- disconnected road topology

---

# Urban Metric Extraction

## Green Coverage

```python
green_ratio = green_pixels / total_pixels
```

## Building Density

```python
building_density = building_pixels / total_pixels
```

## Road Coverage

```python
road_ratio = road_pixels / total_pixels
```

---

# Performance Metrics

The project evaluates both image quality and urban structural correctness.

| Metric | Purpose |
|---|---|
| SSIM | Structural similarity |
| L1 Loss | Pixel reconstruction accuracy |
| Edge Consistency | Boundary preservation |
| Sustainability Score | Urban infrastructure quality |

---

# Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Cityscapes |
| Resolution | 256×256 |
| Epochs | 30 |
| Fine-Tuning Epochs | 10 |
| Generator Filters | 32 |
| Batch Size | Hardware dependent |
| Optimizer | Adam |
| Training Objective | GAN + L1 |

---

# Lightweight VRAM Optimization

The project is optimized for constrained GPUs.

## Optimization Decisions

| Optimization | Purpose |
|---|---|
| `ngf = 32` | Lower VRAM usage |
| Reduced resolution | Faster training |
| Lightweight U-Net | Memory efficiency |
| PatchGAN | Reduced discriminator complexity |

---

# Post-Processing Pipeline

The inference pipeline includes image enhancement stages to improve visual coherence.

```text
Generated Image
       ↓
Contrast Adjustment
       ↓
Artifact Reduction
       ↓
Edge Smoothing
       ↓
Enhanced Output
```

---

# Project Structure

```text
📦 Pix2Pix-CityScape
├── 📂 src
│   ├── 📂 data
│   │   └── ...                         ← Dataset loading and preprocessing
│   │
│   ├── 📂 models
│   │   ├── generator.py                ← U-Net generator
│   │   └── discriminator.py            ← PatchGAN discriminator
│   │
│   ├── 📂 training
│   │   └── train.py                    ← GAN training pipeline
│   │
│   ├── 📂 inference
│   │   └── postprocess.py              ← Enhancement and cleanup
│   │
│   ├── 📂 utils
│   │   ├── metrics.py                  ← Image quality metrics
│   │   └── sustainability.py           ← Urban metric calculations
│   │
│   └── 📂 losses
│       └── perceptual_loss.py          ← Optional perceptual objective
│
├── 📄 app.py                           ← Gradio dashboard
├── 📄 train.py                         ← Training entrypoint
├── 📄 requirements.txt                 ← Dependency manifest
└── 📄 README.md                        ← Project documentation
```

---

# Quickstart

## Linux / macOS

```bash
git clone <your-repo-url>

cd SustainableUrbanPix2Pix

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python app.py
```

---

## Windows PowerShell

```powershell
git clone <your-repo-url>

cd SustainableUrbanPix2Pix

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python app.py
```

---

## Windows CMD

```cmd
git clone <your-repo-url>

cd SustainableUrbanPix2Pix

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

---

# Training

## Train From Scratch

```bash
python train.py --epochs 30
```

## Fine-Tune Existing Model

```bash
python train.py --epochs 10 --lr 0.00005
```

---

# Inference Workflow

```text
Upload Semantic Map
         ↓
GAN Image Generation
         ↓
Post-Processing
         ↓
Metric Extraction
         ↓
Sustainability Analysis
         ↓
Dashboard Visualization
```

---

# Example Urban Analysis

## Input

```text
Semantic layout with:
- dense buildings
- limited vegetation
- disconnected roads
```

## Output Interpretation

```text
Generated Result:
- moderate realism
- low sustainability score
- high urban density detected
- weak road connectivity
```

---

# Results

| Observation | Status |
|---|---|
| Structural consistency | Strong |
| Semantic preservation | Strong |
| Texture realism | Moderate |
| Sustainability estimation | Reliable |
| Low-VRAM compatibility | Verified |

---

# Known Limitations

| Limitation | Cause |
|---|---|
| Blurry textures | Lightweight architecture |
| Checkerboard artifacts | ConvTranspose operations |
| Limited realism | Reduced model complexity |
| Resolution constraints | Hardware optimization |

---

# Future Improvements

| Planned Improvement | Purpose |
|---|---|
| Upsampling layers | Reduce checkerboard artifacts |
| Super-resolution stage | Improve realism |
| Diffusion refinement | Better textures |
| Multi-layout comparison | Comparative urban analysis |
| PDF report export | Planning documentation |
| Interactive editing tools | Urban layout manipulation |

---

# Real-World Applications

| Domain | Application |
|---|---|
| Smart Cities | Urban planning simulation |
| Infrastructure Research | Sustainability analysis |
| Education | GAN visualization learning |
| Transportation Planning | Road layout evaluation |
| Environmental Studies | Green coverage analysis |

---

# Gradio Dashboard

The system includes an interactive Gradio interface for semantic map upload, generation, visualization, and sustainability analysis.

## Dashboard Responsibilities

| Component | Function |
|---|---|
| Upload Panel | Semantic map input |
| Visualization Panel | Generated urban scene |
| Metrics Panel | Sustainability statistics |
| Analysis Section | Structural interpretation |

---

# Research Direction

This project explores how conditional GANs can extend beyond image synthesis into interpretable infrastructure simulation systems. The architecture prioritizes urban structural preservation and downstream sustainability analysis over purely photorealistic output generation.

---

# Conclusion

Pix2Pix CityScape demonstrates a practical integration of computer vision, GAN-based semantic synthesis, and sustainability-oriented infrastructure analytics within a lightweight deployment framework. The system combines urban visualization with quantitative environmental analysis, enabling GAN-generated outputs to function as structured planning artifacts instead of purely aesthetic renderings.

---

# License

MIT License

---

<div align="center">

# PIX2PIX CITYSCAPE

Semantic Urban Intelligence

*Generate • Analyze • Sustain*

</div>
