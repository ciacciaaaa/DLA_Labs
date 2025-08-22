# explainability.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


# ------------------------------------------------------------
def preprocess_image(img_path, img_size=160):
    """Load image and convert to tensor"""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE), img


# ------------------------------------------------------------
def generate_cam(model, feature_layer, img_tensor, target_class=None):
    """
    Generate Class Activation Map for a single image.
    
    Args:
        model: CNN model
        feature_layer: last convolutional layer from which to extract features
        img_tensor: preprocessed input image tensor (1,C,H,W)
        target_class: class index for which CAM is computed; if None, use predicted class
    """
    model.eval()
    
    # Hook to get feature maps
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    handle = feature_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    output = model(img_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Get weights from the classifier (linear layer)
    if hasattr(model, "classifier"):
        # For CNN class: last linear layer
        weight_softmax = model.classifier[-1].weight.detach().cpu().numpy()
    else:
        # For ResNet: final linear layer
        weight_softmax = model.linear.weight.detach().cpu().numpy()
    
    # Get feature map from hook
    feature_map = features[0].squeeze(0).cpu().numpy()  # shape: (C,H,W)
    
    # Compute CAM
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    for i, w in enumerate(weight_softmax[target_class]):
        cam += w * feature_map[i, :, :]
    
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (img_tensor.size(3), img_tensor.size(2)))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    handle.remove()
    return cam, target_class

# ------------------------------------------------------------
# Function to plot CAM over image

def show_cam_on_image(img_pil, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = np.array(img_pil)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()


