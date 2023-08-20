import torch

from PIL import Image

from EXPERIMENTS.explore_detr_architecture.model_wrapper import modelWrapper

#-Architecture parameters-
num_classes = 91
hidden_dim = 2
nheads = 1
num_encoder_layers = 1
num_decoder_layers = 1
#------------------------

detr_model = modelWrapper(num_classes, hidden_dim=hidden_dim, nheads=nheads,
                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
detr_model.eval()

path = '/home/projects/bagon/shared/coco/val2017/000000039769.jpg'
im = Image.open(path)

detr_model.detect(im)
detr_model.plot_and_save_results()
