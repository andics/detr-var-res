import torch
from PIL import Image, ImageDraw
from collections import OrderedDict
import torchvision.transforms as T
import argparse
import numpy as np

# Define the colors for visualization (used for semi-transparent overlays)
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Define the classes for COCO dataset
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path):
    # Use the panoptic version so we get segmentation masks
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu')

    # Adjust if your checkpoint uses "detr." prefix or not
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        # If the keys contain "detr.", remove that prefix
        if "detr." in k:
            k_n = k.replace("detr.", "")
            state_dict[k_n] = v
        else:
            # Just in case some keys do not have "detr."
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model

def plot_segmentations(img, masks):
    """
    Overlays segmentation masks on the image with partial opacity.
    If you want unique colors for each mask, you can iterate over COLORS.
    """
    rgba_img = img.convert('RGBA')
    w, h = rgba_img.size

    for i, mask in enumerate(masks):
        # Threshold the mask, making it binary
        bin_mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Convert binary mask to the same size as the image
        mask_img = Image.fromarray(bin_mask, mode='L').resize((w, h))
        # Pick a semi-transparent color
        color = COLORS[i % len(COLORS)]
        overlay = Image.new('RGBA', (w, h), (
            int(color[0]*255),
            int(color[1]*255),
            int(color[2]*255),
            100
        ))
        rgba_img = Image.composite(overlay, rgba_img, mask_img)

    return rgba_img

def main(args):
    model = load_model(args.model_path)
    img = Image.open(args.image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    outputs = model(img_transformed)

    # Panoptic DETR has 'pred_logits' and 'pred_masks'
    # pred_logits: [batch_size, num_queries, num_classes]
    # pred_masks:  [batch_size, num_queries, H, W]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.1

    # Keep only masks whose associated query has confidence above threshold
    masks = outputs['pred_masks'][0, keep]

    # Plot segmentations
    result_img = plot_segmentations(img, masks)
    result_img.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR panoptic inference script")
    parser.add_argument("--model_path", required=True, help="Path to the panoptic DETR checkpoint (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", required=True, help="Path to save the output image with segmentations")
    args = parser.parse_args()
    main(args)