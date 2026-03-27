
# SustainableUrbanPix2Pix

**Sketch-to-Photorealistic Urban Layouts for Sustainable City Planning**  
*Using Conditional Pix2Pix GAN optimized for low-end hardware*

---

## 📋 Project Overview

This project implements a **conditional Pix2Pix GAN** that converts raw hand-drawn urban plans (JPG/PNG sketches) into photorealistic sustainable city layouts.  

It allows users to control **green intensity** (more parks & trees) and **building density** to generate climate-responsive urban designs — ideal for sustainable city planning visualizations.

### Key Features
- Supports raw hand-drawn architectural drafts and sketches
- Built-in Canny edge preprocessing for messy drawings
- Controllable sustainability parameters (Green Intensity & Building Density)
- Heavily optimized for **GTX 1050 Ti 4GB VRAM + 16GB RAM**
- Professional modular code structure
- Ready for academic technical report on GAN variants

---

## 🛠️ Hardware Compatibility
- **GPU**: GTX 1050 Ti (4GB VRAM)
- **RAM**: 16GB System RAM
- **Optimizations Used**: Automatic Mixed Precision (AMP), Gradient Checkpointing, Gradient Accumulation, Tiny Mode (ngf=32)

---

## 📥 Installation & Setup

### 1. Clone / Download Project
```powershell
# If you created it via script, just navigate to the folder
cd SustainableUrbanPix2Pix
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download Dataset (Important)
Download the **Cityscapes Pix2Pix Dataset** from:  
[https://www.kaggle.com/datasets/balraj98/cityscapes-pix2pix-dataset](https://www.kaggle.com/datasets/balraj98/cityscapes-pix2pix-dataset)

Extract the files into:
- `data/cityscapes/train/`
- `data/cityscapes/val/`

---

## 🚀 Usage Commands

### Training
```powershell
# Recommended: Tiny mode (safest for GTX 1050 Ti)
python train.py --tiny

# Normal mode
python train.py
```

### Launch Gradio Web Demo
```powershell
python app.py
```
Open browser at: `http://127.0.0.1:7860`

### Inference (Command Line)
```powershell
python inference.py --sketch custom_sketches/my_plan.jpg --green 0.7 --density 0.6
```

---

## 📁 Project Structure

```
SustainableUrbanPix2Pix/
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── app.py
├── inference.py
├── src/
│   ├── models/           # UNetGenerator + PatchGAN
│   ├── data/             # CityscapesDataset
│   ├── utils/            # Preprocessing, metrics
│   ├── training/         # Trainer class
│   └── inference/        # Inference logic
├── data/cityscapes/      # Dataset goes here
├── custom_sketches/      # Put your hand-drawn plans here
└── outputs/
    ├── checkpoints/      # Saved model weights
    ├── samples/          # Training generated images
    └── results/          # Final generated layouts
```

---

## 📊 Report Ready Sections

Use this project for your technical report:

- **Selected GAN Variant**: Pix2Pix (Conditional GAN for Image-to-Image Translation)
- **Architecture**: U-Net Generator + PatchGAN Discriminator
- **Loss Functions**: Adversarial Loss + L1 Reconstruction Loss (λ=100)
- **Optimizations**: AMP, Gradient Checkpointing, Gradient Accumulation
- **Sustainability Features**: Conditional green intensity & density control
- **Evaluation**: Visual results from `outputs/samples/` and `outputs/results/`

**Suggested Report Sections**:
1. Abstract
2. Introduction to Pix2Pix GAN
3. Methodology & Architecture
4. Low-VRAM Optimizations
5. Sustainability Conditioning
6. Results & Discussion (include before/after images)
7. Ethics & Limitations
8. Conclusion

---

## 📝 Tips for Best Results

- Draw plans with clear lines (roads, buildings, green areas)
- Use **Canny Edge Cleaning** option in the Gradio app for messy sketches
- Increase **Green Intensity** for more eco-friendly layouts
- Lower **Building Density** to reduce urban heat island effect

---

## 🔗 References

- Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks.
- Cityscapes Pix2Pix Dataset: https://www.kaggle.com/datasets/balraj98/cityscapes-pix2pix-dataset


```

