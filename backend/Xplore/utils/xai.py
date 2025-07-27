# utils/xai.py
import torch
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# --- TARGET CLASS FOR SEGMENTATION ---
class SemanticSegmentationTarget:
    """Helper class to tell the CAM algorithm what to explain in a segmentation mask."""
    def __init__(self, category, mask, device=None):
        self.category = category
        self.mask = torch.from_numpy(mask).float()
        self.device = device
        if device:
            self.mask = self.mask.to(device)
    
    def __call__(self, model_output):
        # model_output: [B, C, H, W] or [C, H, W]
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        if model_output.ndim == 4:
            # [B, C, H, W] -> [C, H, W]
            model_output = model_output[0]
        # Ensure mask and model_output are on the same device and shape
        mask = self.mask
        if mask.shape != model_output[self.category].shape:
            # Resize mask if needed (should not happen if mask is correct)
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=model_output[self.category].shape, mode='nearest'
            )[0,0]
        mask = mask.to(model_output.device)
        return (model_output[self.category] * mask).sum()

# --- CAM GENERATOR FOR SEGMENTATION ---
def generate_cam(model, rgb_img, model_output_mask, target_layers, target_category_name, all_classes, algo="gradcam", device="cpu"):
    """Generates a Class Activation Map (CAM) for a segmentation model."""
    sem_class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    if target_category_name not in sem_class_to_idx:
        raise ValueError(f"Target category '{target_category_name}' not found in the list of classes.")
    
    category_index = sem_class_to_idx[target_category_name]
    mask_float = np.float32(model_output_mask == category_index)
    targets = [SemanticSegmentationTarget(category_index, mask_float, device=device)]
    
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(device)

    # --- Ensure input_tensor dtype matches model's first parameter dtype ---
    model_param_dtype = next(model.parameters()).dtype
    if input_tensor.dtype != model_param_dtype:
        input_tensor = input_tensor.to(dtype=model_param_dtype)
    
    cam_algorithm = {
        "gradcam": GradCAM, "scorecam": ScoreCAM, "eigengradcam": EigenCAM,
        "ablationcam": AblationCAM, "layercam": LayerCAM,
    }.get(algo.lower())

    if cam_algorithm is None:
        raise ValueError(f"Unsupported CAM algorithm: {algo}")

    cam_kwargs = {'model': model, 'target_layers': target_layers}
    # Only add use_cuda for algorithms that accept it (not GradCAM/EigenCAM)
    if cam_algorithm in [ScoreCAM, AblationCAM, LayerCAM]:
        cam_kwargs['use_cuda'] = (device != "cpu")

    try:
        with cam_algorithm(**cam_kwargs) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cam_image
    except Exception as e:
        # Add debug info for easier troubleshooting
        raise RuntimeError(f"CAM generation failed: {e}")

# --- TARGET CLASS FOR DETECTION ---
class DetectionBoxTarget:
    """Target the confidence score of a specific detection (e.g., the first one)."""
    def __init__(self, box_index=0):
        self.box_index = box_index

    def __call__(self, model_output):
        # model_output: (batch, num_boxes, 6) or similar
        # For YOLO, output is usually (batch, num_boxes, 6) [x1, y1, x2, y2, conf, class]
        # We target the confidence score of the first detection
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        if model_output.ndim == 3:
            # batch, num_boxes, 6
            return model_output[0, self.box_index, 4]
        elif model_output.ndim == 2:
            # num_boxes, 6
            return model_output[self.box_index, 4]
        else:
            # fallback: sum all confidences
            return model_output.sum()

# --- CAM GENERATOR FOR OBJECT DETECTION ---
def generate_detection_cam(model, input_tensor, target_layers, rgb_img):
    """Generates a LayerCAM visualization for an object detection model."""
    # --- Add a target for the first detection ---
    targets = [DetectionBoxTarget(box_index=0)]
    with LayerCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return cam_image
