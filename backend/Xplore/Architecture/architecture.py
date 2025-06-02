import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO


def load_image_segmentation():
    model = smp.Unet(encoder_name='resnet34',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model, device

def load_human_detection(weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(weight)
    model.to(device)

    return model