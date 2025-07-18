# utils/xai.py

import torch
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class SemanticSegmentationTarget:
    def __init__(self, category, mask, device=None):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if device is not None:
            self.mask = self.mask.to(device)

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

def generate_cam(
    model, rgb_img, mask, target_layers, algo="gradcam", device="cpu", category=1
):
    """
    model: PyTorch model
    rgb_img: H×W×3 float in [0,1]
    mask: H×W float binary mask
    target_layers: list of layers
    algo: string, e.g. "gradcam"
    device: "cpu" or "cuda"
    category: int, semantic class index
    """
    cam_class = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "eigengradcam": EigenGradCAM,
        "ablationcam": AblationCAM,
        "layercam": LayerCAM,
    }[algo.lower()]
    cam = cam_class(model=model, target_layers=target_layers, use_cuda=(device != "cpu"))
    targets = [SemanticSegmentationTarget(category, mask, device=device)]
    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    input_tensor = input_tensor.to(device)
    model.to(device)
    with torch.no_grad():
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_img, grayscale, use_rgb=True)
