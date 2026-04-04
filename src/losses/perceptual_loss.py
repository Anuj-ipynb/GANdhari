# src/losses/perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layers=[3, 8, 17, 26, 35]):
        super().__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        self.vgg = nn.Sequential(*list(vgg.children())[:36]).eval()  # up to relu5_4
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.to(device)
        
        self.criterion = nn.L1Loss()
        self.layers = layers
        self.weights = [1.0, 0.75, 0.5, 0.25, 0.1]

    def forward(self, input_img, target_img):
        # Normalize to VGG input range [-2, 2] approx (common practice)
        input_features = self.get_features(input_img)
        target_features = self.get_features(target_img)
        
        loss = 0.0
        for i, (inp, tgt) in enumerate(zip(input_features, target_features)):
            loss += self.weights[i] * self.criterion(inp, tgt)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features