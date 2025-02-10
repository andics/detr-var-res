import torch
from PIL import Image, ImageDraw
from collections import OrderedDict
import torchvision.transforms as T
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', num_classes=91, pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu')

    # Adjust if your checkpoint uses "detr." prefix or not
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            k_n = k.replace("detr.", "")
            state_dict[k_n] = v
        else:
            state_dict[k] = v

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model

def plot_segmentations_with_text(img, masks, probas):
    """
    Overlays the selected segmentation masks on the image with partial opacity
    and adds a text label for each mask.
    """
    rgba_img = img.convert('RGBA')
    draw = ImageDraw.Draw(rgba_img)
    w, h = rgba_img.size

    for i, mask in enumerate(masks):
        # Threshold the mask, making it binary
        bin_mask = (mask > 0.03).cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(bin_mask, mode='L').resize((w, h))

        # Pick a semi-transparent color
        color = COLORS[i % len(COLORS)]
        overlay = Image.new('RGBA', (w, h), (
            int(color[0]*255),
            int(color[1]*255),
            int(color[2]*255),
            80
        ))
        rgba_img = Image.composite(overlay, rgba_img, mask_img)

        # Find bounding box for the mask (for text placement)
        np_mask = np.array(mask_img)
        ys, xs = np.where(np_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        # Determine class label
        class_idx = probas[i].argmax().item()
        score = probas[i][class_idx].item()
        label_text = f"{CLASSES[class_idx]}: {score:.2f}"

        # Draw label near bounding box
        draw.text((xmin, ymin), label_text, fill=(255, 255, 255, 255))

    return rgba_img

def main(args):
    model = load_model(args.model_path)
    img = Image.open(args.image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    outputs = model(img_transformed)
    # We want the two MOST confident proposals
    # pred_logits: [batch_size, num_queries, num_classes]
    # pred_masks:  [batch_size, num_queries, H, W]
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]         # ignore no-object
    scores = probas.max(dim=1).values                              # shape: [num_queries]
    top_scores, top_inds = scores.topk(3)                          # pick top-3 instead of top-2
    # Extract top-3 masks and their probabilities
    masks = outputs['pred_masks'].sigmoid()[0, top_inds]           # Apply sigmoid to convert logits to probabilities
    top_probas = probas[top_inds]

    # Plot segmentations with text
    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    for i, (mask, prob) in enumerate(zip(masks, top_probas)):
        # Create binary mask
        bin_mask = (mask > 0.03).cpu().numpy()
        
        # Add colored overlay
        color = np.array(COLORS[i % len(COLORS)])
        plt.imshow(np.ones_like(np.array(img)) * color, alpha=0.1 * bin_mask)  # Reduce alpha for better visibility
        
        # Add text label with improved visibility
        y, x = np.where(bin_mask)
        if len(y) > 0 and len(x) > 0:
            centroid_y = int(np.mean(y))
            centroid_x = int(np.mean(x))
            
            class_idx = prob.argmax().item()
            score = prob[class_idx].item()
            label_text = f"{CLASSES[class_idx]}\n{score:.2f}"
            
            plt.text(centroid_x, centroid_y, label_text,
                    color='lime',
                    fontsize=24,
                    fontweight='bold',
                    bbox=dict(facecolor='black',
                            alpha=0.7,
                            edgecolor='lime',
                            pad=1.0),
                    ha='center',
                    va='center')
    
    plt.axis('off')
    plt.savefig(args.output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR panoptic inference script (top-2 proposals)")
    parser.add_argument("--model_path", required=True, help="Path to the panoptic DETR checkpoint (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", required=True, help="Path to save the output image with segmentations")
    args = parser.parseArgs()
    main(args)