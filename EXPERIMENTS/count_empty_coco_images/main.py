from PIL import Image
import matplotlib.pyplot as plt
import time
import json

# Open an image file
img = Image.open("/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Datasets/coco_variable/val2017/000000000139.jpg")  # replace with your image file path

# Display the image
plt.imshow(img)
plt.show()

# Provide the path to your json file
json_path = '/home/projects/bagon/shared/coco/annotations/instances_val2017.json'

# Open the json file and load its contents into a Python object
with open(json_path, 'r') as f:
    data = json.load(f)

# Now data contains your json

with open(json_path) as f:
    data = json.load(f)

# images with no annotations
empty_images = 0

# Iterate over each image in the dataset
for image in data['images']:
    has_annotations = False
    for annotation in data['annotations']:
        if image['id'] == annotation['image_id']:
            has_annotations = True
            break
    if not has_annotations:
        empty_images += 1

print(f'Number of empty images: {empty_images}')
print(data)
