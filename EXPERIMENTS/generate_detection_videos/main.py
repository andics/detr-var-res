import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import argparse

# Define the colors for visualization
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
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    draw = ImageDraw.Draw(pil_img)
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=tuple(int(255 * x) for x in c), width=3)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        draw.text((xmin, ymin), text, fill=(255, 255, 255, 255))
    return pil_img

def main(args):
    model = load_model(args.model_path)
    img = Image.open(args.image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    outputs = model(img_transformed)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    result_img = plot_results(img, probas[keep], bboxes_scaled)
    result_img.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR inference script")
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", required=True, help="Path to save the output image with detections")
    args = parser.parse_args()
    main(args)