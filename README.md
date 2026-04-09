# 🌆 SustainableUrbanPix2Pix

An AI-powered urban planning assistant that converts **semantic city layouts into realistic images** and evaluates their **sustainability** using computer vision and GANs.

---

## 🚀 Overview

**SustainableUrbanPix2Pix** is a Pix2Pix-based Generative Adversarial Network (GAN) trained on the Cityscapes dataset. It generates realistic urban scenes from semantic segmentation maps and computes meaningful urban metrics such as:

* 🌳 Green Coverage
* 🏢 Building Density
* 🛣️ Road Coverage
* ♻️ Sustainability Score

The system helps visualize and analyze urban layouts for better planning decisions.

---

## 🧠 Key Features

* 🎨 **Semantic → Real Image Generation** using Pix2Pix GAN
* 📊 **Urban Metrics Extraction** (green, roads, buildings)
* ♻️ **Sustainability Scoring System**
* 🖥️ **Interactive Gradio Dashboard**
* ⚡ **Optimized for Low VRAM (GTX 1050 Ti)**
* 🔧 **Robust Post-Processing Pipeline for Image Enhancement**

---

## 🏗️ Model Architecture

### Generator

* U-Net (lightweight)
* `ngf = 32` (optimized for low GPU memory)

### Discriminator

* PatchGAN

### Training Setup

* Dataset: Cityscapes
* Image Size: 256×256
* Loss: GAN Loss + L1 Loss
* Training: 30 epochs + 10 epoch fine-tuning

---

## 📊 Metrics Computed

| Metric                   | Description                                    |
| ------------------------ | ---------------------------------------------- |
| **SSIM**                 | Structural similarity between input and output |
| **L1 Loss**              | Pixel-wise difference                          |
| **Edge Consistency**     | Structural alignment                           |
| **Green Coverage**       | Percentage of vegetation                       |
| **Road Coverage**        | Percentage of roads                            |
| **Building Density**     | Built-up area ratio                            |
| **Road Connectivity**    | Connectivity of road network                   |
| **Sustainability Score** | Combined urban quality metric                  |

---

## ♻️ Sustainability Score

The score is computed based on:

* 🌳 Higher green coverage → better
* 🏢 Lower building density → better
* 🛣️ Balanced road coverage → better
* 🔗 Good connectivity → better

---

## 📂 Project Structure

```
src/
├── data/              # Dataset loader
├── models/            # Generator & Discriminator
├── training/          # Training pipeline
├── inference/         # Inference + post-processing
├── utils/             # Metrics and utilities
├── losses/            # (optional) perceptual loss
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd SustainableUrbanPix2Pix

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 🧪 Training

### Train from scratch

```bash
python train.py --epochs 30
```

### Fine-tune existing model

```bash
python train.py --epochs 10 --lr 0.00005
```

---

## 🔍 Inference

Run the app:

```bash
python app.py
```

Then:

1. Upload a semantic map
2. Click generate
3. View output + metrics

---

## 🎯 Results

* ✅ Structurally correct urban layouts
* ⚠️ Limited texture realism (due to lightweight model)
* ✅ Reliable sustainability metrics

---

## ⚠️ Limitations

* Blurry textures due to lightweight architecture
* Checkerboard artifacts (partially mitigated)
* Not photorealistic (focus is structural accuracy)

---

## 🚀 Future Improvements

* Replace ConvTranspose with Upsampling (reduce artifacts)
* Add super-resolution module
* Improve UI with layout editing tools
* Add multi-layout comparison
* Integrate diffusion-based refinement
* Export PDF reports for urban analysis

---

## 🧠 Real-World Applications

* Urban planning & smart city design
* Sustainability analysis
* Infrastructure simulation
* Educational visualization tools

---

## 🏆 Conclusion

This project demonstrates how GANs can be used not just for image generation, but as **decision-support tools** for real-world problems like urban sustainability.

---

## 👨‍💻 Author

Developed as an academic project using deep learning and computer vision techniques.

---

## ⭐ Acknowledgements

* Cityscapes Dataset
* Pix2Pix (Isola et al.)
* OpenCV & PyTorch

---
