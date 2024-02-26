import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import requests
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
torch.set_grad_enabled(False)


TOP_K = 4
# How inward the static attention fixations will be in terms of image width and height.
STATIC_ATTN_OFFSET = 0.17
# Image resolution
DPI = 200
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
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


class DETRModel:

    def __init__(self, path_to_checkpoint):
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=False)
        checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()


class InferenceHandler:

    def __init__(self, model_paths, dataset_paths, output_path, fixation_dynamics_static: bool):
        self.models = [DETRModel(path).model for path in model_paths]
        self.dataset_paths = dataset_paths
        self.output_path = output_path
        self.fixation_dynamics_static = fixation_dynamics_static

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self.transform = Compose([
            Resize(800),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.static_attention_points = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = InferenceHandler.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    @staticmethod
    def plot_results(pil_img, prob, boxes):
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        fig.savefig('mygraph.png', bbox_inches='tight', pad_inches=0)
        plt.show()

    @staticmethod
    def find_file_with_suffix(folder, suffix):
        # List all files in the folder
        files = os.listdir(folder)

        # Filter the files that end with the given suffix
        matching_files = [f for f in files if f.endswith(suffix)]

        # If there are any matching files, return the first one
        assert len(matching_files) == 1

        return matching_files[0]

    @staticmethod
    def calculate_static_attn_coordinates(image_width, image_height):
        width_offset = int(image_width // (1/STATIC_ATTN_OFFSET))  # 15% of width
        height_offset = int(image_height // (1/STATIC_ATTN_OFFSET))  # 15% of height

        # Creating the coordinates
        # Start from top left and move in clockwise direction
        top_left = (height_offset, width_offset)  # Top left
        top_right = (height_offset, image_width - width_offset)  # Top right
        bottom_right = (image_height - height_offset, image_width - width_offset)  # Bottom right
        bottom_left = (image_height - height_offset, width_offset)  # Bottom left

        return [top_left, top_right, bottom_right, bottom_left]


    def collect_images(self):
        '''
        Collects the full paths to images in the provided args.dataset_paths variable such that every
        image is matched by name to every other image in the remaining folders

        The smallest number of images present in each folder sets the number of images that will be matched
         for all other folders
        :return: None
        '''
        # Initialize a dictionary to store image names for each dataset
        dataset_images = {path: [] for path in self.dataset_paths}

        # Initialize a list to store common image names
        common_images = None

        # Iterate over dataset paths
        for path in self.dataset_paths:
            # List all images in the dataset directory
            images = [file for file in os.listdir(path) if file.endswith(".jpg")]
            # Extract the universal part of image names (last 9 chars + 4 for the extension)
            stripped_names = [name[-13:] for name in images]
            # Update the dictionary with stripped image names
            dataset_images[path] = stripped_names

            # If common_images is not initialized, set it to the current set of stripped image names
            if common_images is None:
                common_images = set(stripped_names)
            else:
                # Intersect the current set of names with the common_images set
                common_images = common_images.intersection(stripped_names)

        # Sort the set common_images
        common_images = sorted(common_images)

        # Now common_images contains only image names that exist in all dataset paths

        # Prepare a list for each dataset path with full image paths that are common in all datasets
        dataset_common_images = []
        for path in self.dataset_paths:
            # Filter the images that contain one of the common names
            common_images_dataset = [os.path.join(path, InferenceHandler.find_file_with_suffix(path, image)) for image in common_images]
            dataset_common_images.append(common_images_dataset)

        self.tuple_of_img_paths_lists = tuple(dataset_common_images)


    def infer(self):
        for image_paths in zip(*self.tuple_of_img_paths_lists):
            fig, ax = plt.subplots(len(self.models), 3, figsize=(len(self.models) * 5, 3*5), dpi = DPI)
            # Get base name of the image (file name without extension)
            _base_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
            # Extract the last 8 characters
            _last_chars = _base_name[-8:]

            image_name = _last_chars + ".png"
            full_img_collage_save_path = os.path.join(self.output_path, image_name)
            if os.path.exists(full_img_collage_save_path):
                print(f"Image composition {image_name} exists. Moving on...")
                continue
            else:
                print(f"Working on image {image_name}")

            for i, (model, img_path) in enumerate(zip(self.models, image_paths)):
                img_pil = Image.open(img_path).convert("RGB")
                img = self.transform(img_pil).unsqueeze(0)

                # arrays in which we save the featuremaps
                conv_features = []
                enc_self_attn_weights_l0, enc_self_attn_weights_l2, enc_self_attn_weights_l5 = [], [], []
                dec_mult_head_attn_weights_l0, dec_mult_head_attn_weights_l2, dec_mult_head_attn_weights_l5 = [], [], []
                dec_self_attn_weights_l0, dec_self_attn_weights_l2, dec_self_attn_weights_l5 = [], [], []

                hooks = [
                    model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)
                    ),
                    model.transformer.encoder.layers[0].self_attn.register_forward_hook(
                        lambda self, input, output: enc_self_attn_weights_l0.append(output[1])
                    ),
                    model.transformer.encoder.layers[2].self_attn.register_forward_hook(
                        lambda self, input, output: enc_self_attn_weights_l2.append(output[1])
                    ),
                    model.transformer.encoder.layers[5].self_attn.register_forward_hook(
                        lambda self, input, output: enc_self_attn_weights_l5.append(output[1])
                    ),
                    model.transformer.decoder.layers[0].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_mult_head_attn_weights_l0.append(output[1])
                    ),
                    model.transformer.decoder.layers[2].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_mult_head_attn_weights_l2.append(output[1])
                    ),
                    model.transformer.decoder.layers[5].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_mult_head_attn_weights_l5.append(output[1])
                    ),
                    model.transformer.decoder.layers[0].self_attn.register_forward_hook(
                        lambda self, input, output: dec_self_attn_weights_l0.append(output)
                    ),
                    model.transformer.decoder.layers[2].self_attn.register_forward_hook(
                        lambda self, input, output: dec_self_attn_weights_l2.append(output[1])
                    ),
                    model.transformer.decoder.layers[5].self_attn.register_forward_hook(
                        lambda self, input, output: dec_self_attn_weights_l5.append(output[1])
                    )
                ]

                # propagate through the model
                outputs = model(img)

                for hook in hooks:
                    hook.remove()


                # keep only top k predictions
                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                _, idxs = probas.max(-1).values.sort(descending=True)

                # keep only the top 5
                keep = idxs[:TOP_K]

                #---PLOT-PREDICTIONS---
                # convert boxes from [0; 1] to image scales
                bboxes_scaled = InferenceHandler.rescale_bboxes(outputs['pred_boxes'][0, keep], img_pil.size)

                ax[i, 0].imshow(img_pil)
                colors = COLORS * 100
                for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         fill=False, color=c, linewidth=3)
                    ax[i, 0].add_patch(rect)
                    cl = p.argmax()
                    text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                    ax[i, 0].text(xmin, ymin, text, fontsize=15,
                                  bbox=dict(facecolor='yellow', alpha=0.5))
                ax[i, 0].set_title(f'Predictions')
                #----------------------
                # ---PLOT-DECODER-ATTENTION---
                # don't need the list anymore
                conv_features = conv_features[0]
                enc_self_attn_weights_l0 = enc_self_attn_weights_l0[0]
                dec_mult_head_attn_weights_l0 = dec_mult_head_attn_weights_l0[0]
                dec_self_attn_weights_l0 = dec_self_attn_weights_l0[0]
                enc_self_attn_weights_l2 = enc_self_attn_weights_l2[0]
                dec_mult_head_attn_weights_l2 = dec_mult_head_attn_weights_l2[0]
                dec_self_attn_weights_l2 = dec_self_attn_weights_l2[0]
                enc_self_attn_weights_l5 = enc_self_attn_weights_l5[0]
                dec_mult_head_attn_weights_l5 = dec_mult_head_attn_weights_l5[0]
                dec_self_attn_weights_l5 = dec_self_attn_weights_l5[0]

                '''
                enc_self_attn_weights_l0 = enc_self_attn_weights_l2
                dec_mult_head_attn_weights_l0 = dec_mult_head_attn_weights_l2
                '''

                # get the feature map shape
                h, w = conv_features['0'].tensors.shape[-2:]

                fig_2, ax_2 = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
                for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep, ax_2.T, bboxes_scaled):
                    axs = ax_i[0]
                    axs.imshow(dec_mult_head_attn_weights_l0[0, idx].view(h, w))
                    axs.axis('off')
                    axs.set_title(f'query id: {idx.item()}')

                    axs = ax_i[1]
                    axs.imshow(img_pil)
                    axs.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                               fill=False, color='blue', linewidth=3))
                    axs.axis('off')
                    axs.set_title(CLASSES[probas[idx].argmax()])

                fig_2.tight_layout()
                # Save the subplot fig_2 as an image
                fig_2.savefig(os.path.join(self.output_path, 'temp_fig_dec_at.png'), dpi = DPI)
                # Clear the current figure to free memory
                plt.close(fig_2)

                # Load the saved image
                loaded_image = plt.imread(os.path.join(self.output_path, 'temp_fig_dec_at.png'))

                # Display the saved image in the original subplot
                ax[i, 2].imshow(loaded_image)
                ax[i, 2].axis('off')
                ax[i, 2].set_title(f'Decoder attention')
                #----------------------------
                #---PLOT-ENCODER-ATTENTION---
                # output of the CNN
                f_map = conv_features['0']
                print("Encoder attention:      ", enc_self_attn_weights_l0[0].shape)
                print("Feature map:            ", f_map.tensors.shape)
                # get the HxW shape of the feature maps of the CNN
                shape = f_map.tensors.shape[-2:]
                # and reshape the self-attention to a more interpretable shape
                sattn = enc_self_attn_weights_l0[0].reshape(shape + shape)
                print("Reshaped self-attention:", sattn.shape)

                # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
                fact = 32

                #---ATTENTION-POINTS-SELECTION---
                # let's select 4 reference points for visualization
                idxs = []
                if self.fixation_dynamics_static:
                    idxs = InferenceHandler.calculate_static_attn_coordinates(image_width = img.shape[-1],
                                                                              image_height = img.shape[-2])
                else:
                    # The transformation scale factor applied before inference
                    _transform_rescale_factor = img.shape[-2] / img_pil.height
                    for (xmin, ymin, xmax, ymax) in bboxes_scaled.tolist():
                        # Get the box center
                        idxs.append( (int((ymin + (ymax - ymin)/2) * _transform_rescale_factor), int((xmin + (xmax - xmin)/2)) * _transform_rescale_factor) )

                # here we create the canvas
                fig_3 = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
                # and we add one plot per reference point
                gs = fig_3.add_gridspec(2, 4)
                ax_3 = [
                    fig_3.add_subplot(gs[0, 0]),
                    fig_3.add_subplot(gs[1, 0]),
                    fig_3.add_subplot(gs[0, -1]),
                    fig_3.add_subplot(gs[1, -1]),
                ]

                # for each one of the reference points, let's plot the self-attention
                # for that point
                for idx_o, axs in zip(idxs, ax_3):
                    idx = (int(idx_o[0] // fact), int(idx_o[1] // fact))
                    axs.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                    axs.axis('off')
                    axs.set_title(f'self-attention{idx}')

                # and now let's add the central image, with the reference points as red circles
                fcenter_ax = fig_3.add_subplot(gs[:, 1:-1])
                fcenter_ax.imshow(img_pil)
                for (y, x) in idxs:
                    scale = img_pil.height / img.shape[-2]
                    x = ((x // fact) + 0.5) * fact
                    y = ((y // fact) + 0.5) * fact
                    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
                    fcenter_ax.axis('off')

                fig_3.tight_layout()
                # Save the subplot fig_2 as an image
                fig_3.savefig(os.path.join(self.output_path, 'temp_fig_enc_at.png'), dpi = DPI)

                # Clear the current figure to free memory
                plt.close(fig_3)

                # Load the saved image
                loaded_image = plt.imread(os.path.join(self.output_path, 'temp_fig_enc_at.png'))

                # Display the saved image in the original subplot
                ax[i, 1].imshow(loaded_image)
                ax[i, 1].axis('off')
                ax[i, 1].set_title('Encoder (self) Attention')

                #----------------------------

            fig.tight_layout()
            fig.savefig(os.path.join(self.output_path, image_name), dpi = DPI)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_paths", required=True, nargs='+', help="Paths to the model checkpoints.")
    parser.add_argument("--dataset_paths", required=True, nargs='+', help="Paths to the dataset folders.")
    parser.add_argument("--output_path", required=True, help="Path to save the output figures.")
    parser.add_argument("--attention_fixation", required=True, help="Choose 'dynamic' or 'static'.")
    args = parser.parse_args()

    if args.attention_fixation == 'dynamic':
        is_static = False
    elif args.attention_fixation == 'static':
        is_static = True
    else:
        raise ValueError("The value of --attention_fixation must be 'dynamic' or 'static'.")

    inference_handler = InferenceHandler(args.model_paths, args.dataset_paths, args.output_path, is_static)
    inference_handler.collect_images()
    inference_handler.infer()