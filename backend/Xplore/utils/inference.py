import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')


def image_segmentation(image, model, device):
    # Image transformation
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    with torch.no_grad():
        # Convert the image to RGB and apply the transform
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Model inference
        output = model(image_tensor)
        predicted_mask = torch.argmax(output, dim=1).cpu().numpy()

        return predicted_mask
    
def human_detection(image, model):
    results = model(image)

    return results[0]