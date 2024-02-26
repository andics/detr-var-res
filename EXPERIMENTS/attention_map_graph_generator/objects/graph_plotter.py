import torch
import os
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import time

from tqdm import tqdm
from typing import List, Tuple
from matplotlib import pyplot as plt
from math import floor

class graphPlotter():
    baseline_color = "ed7d31"
    variable_color = "ffc000"
    equiconst_color = "a6a6a6"


    @staticmethod
    def compute_avg_scaled_distance(tensor, attention_region,
                                    attention_region_boundary):
        """Compute the average scaled distance for all (x, y) points."""
        """This is akin to simply computing the attention distance, following Dosovitskiy (2021) et. al."""

        if len(attention_region_boundary) == 1:
            attention_region_inner_boundary = attention_region_boundary[0]
        elif len(attention_region_boundary) == 2 and attention_region == "periphery_strip":
            attention_region_inner_boundary = attention_region_boundary[1]
            strip_outer_boundary = attention_region_boundary[0]

        mid_begin = attention_region_inner_boundary
        mid_size = (1 - attention_region_inner_boundary) - mid_begin
        scale_factor = 32

        # Ensure we are working with torch tensor
        tensor = torch.tensor(tensor)

        # image (height, width, height, width)
        _, _, d1, d2 = tensor.shape

        if attention_region == "center":
            y_coord_cycle_beginning, y_coord_cycle_end = (floor(d1 * mid_begin), floor(d1 * mid_begin + d1 * mid_size))
            x_coord_cycle_beginning, x_coord_cycle_end = (floor(d2 * mid_begin), floor(d2 * mid_begin + d2 * mid_size))
        elif attention_region == "global":
            y_coord_cycle_beginning, y_coord_cycle_end = (0, d1)
            x_coord_cycle_beginning, x_coord_cycle_end = (0, d2)
        elif attention_region == "periphery":
            y_coord_cycle_beginning, y_coord_cycle_end = (0, d1)
            x_coord_cycle_beginning, x_coord_cycle_end = (0, d2)
            inner_hr_square_begin_y, inner_hr_square_end_y = (floor(d1 * mid_begin), floor(d1 * mid_begin + d1 * mid_size))
            inner_hr_square_begin_x, inner_hr_square_end_x = (floor(d2 * mid_begin), floor(d2 * mid_begin + d2 * mid_size))
        elif attention_region == "periphery_strip":
            #The strip will last from outer boundary to mid begin
            y_coord_cycle_beginning, y_coord_cycle_end = (0, d1)
            x_coord_cycle_beginning, x_coord_cycle_end = (0, d2)
            outer_square_begin_y_hr, outer_square_end_y_hr = (floor(d1 * strip_outer_boundary), floor(d1 * (1-strip_outer_boundary)))
            outer_square_begin_x_hr, outer_square_end_x_hr = (floor(d2 * strip_outer_boundary), floor(d2 * (1-strip_outer_boundary)))
            inner_hr_square_begin_y, inner_hr_square_end_y = (
            floor(d1 * mid_begin), floor(d1 * mid_begin + d1 * mid_size))
            inner_hr_square_begin_x, inner_hr_square_end_x = (
            floor(d2 * mid_begin), floor(d2 * mid_begin + d2 * mid_size))

        # Initialize a tensor to store the sum of all scaled distances
        total_scaled_distances = torch.zeros((1))
        query_points_counted = 0

        # Iterate over all (x, y) points
        for y in range(y_coord_cycle_beginning, y_coord_cycle_end):
            for x in range(x_coord_cycle_beginning, x_coord_cycle_end):
                # Exclude any points that are inside the high-resolution area
                if attention_region == "periphery":
                    if (inner_hr_square_begin_y <= y and y <= inner_hr_square_end_y) and (inner_hr_square_begin_x <= x and x <= inner_hr_square_end_x):
                        #print(f"d1: {d1}, d2: {d2}, skipping y: {y}, x: {x}")
                        continue
                if attention_region == "periphery_strip":
                    # Skip if we're not in outer rectangle
                    if not ((outer_square_begin_y_hr <= y and y <= outer_square_end_y_hr)
                            and (outer_square_begin_x_hr <= x and x <= outer_square_end_x_hr)):
                        continue
                    # Skip if we're in inner rectangle
                    if (inner_hr_square_begin_y <= y and y <= inner_hr_square_end_y) and (inner_hr_square_begin_x <= x and x <= inner_hr_square_end_x):
                        #print(f"d1: {d1}, d2: {d2}, skipping y: {y}, x: {x}")
                        continue

                # Extract the 2D map associated with (x,y)
                map_at_xy = tensor[..., y, x]
                #map_at_xy = map_at_xy / map_at_xy.flatten().sum()

                h, w = map_at_xy.size()

                # Compute coordinate grid
                grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                grid_y = grid_y * scale_factor**2
                grid_x = grid_x * scale_factor**2

                # Compute distances
                distances = torch.sqrt((grid_x - x) ** 2 +
                                       (grid_y - y) ** 2)

                # Scale distances by the pixel values in the map
                scaled_distances = distances * map_at_xy

                # Sum the scaled distances and add to total
                total_scaled_distances += scaled_distances.mean()
                query_points_counted += 1

        # Compute the average scaled distance
        avg_scaled_distance = total_scaled_distances / query_points_counted

        return avg_scaled_distance

    @staticmethod
    def plot_data(data_dict, save_path):
        # prepare data list for dataframe
        plot_data = []
        for key in sorted(data_dict.keys()):  # sort the keys
            for sublist in data_dict[key]:
                for j, point in enumerate(sublist):
                    plot_data.append((j + 1, point, key))

        # create dataframe
        df = pd.DataFrame(plot_data, columns=['x', 'y', 'line'])

        # create plot
        palette = {
            "baseline": "#" + graphPlotter.baseline_color,
            "equiconst": "#" + graphPlotter.equiconst_color,
            "variable": "#" + graphPlotter.variable_color
        }
        markers = {
            "baseline": "o",  # circle
            "equiconst": "s",  # square
            "variable": "^"  # triangle
        }
        line_style = [(2, 0), (2, 2), (2, 1)]  # line styles corresponding to 'variable', 'equiconst', 'baseline'

        ax = sns.lineplot(x='x',
                          y='y',
                          hue='line',
                          style='line',
                          markers=markers,
                          dashes=line_style,
                          data=df,
                          errorbar=None,
                          palette=palette)
        # Change axis titles
        plt.xlabel("Depth")
        plt.ylabel("Attention distance (px)")

        # Change legend labels
        legend_text_dict = {"variable": "Variable", "equiconst": "Uniform", "baseline": "Baseline"}
        legend_labels, legend_texts = ax.get_legend_handles_labels()
        for i, t in enumerate(legend_texts):
            legend_texts[i] = legend_text_dict.get(t, t)
        ax.legend(legend_labels, legend_texts)

        # Save the fig
        plt.savefig(save_path)
        plt.show()

    def __init__(self, list_of_path_tuples: List[Tuple[str, str, str]],
                 attention_region,
                 attention_region_boundary,
                 visualizations_parent_dir,
                 num_img_to_use):
        self.list_of_path_tuples = list_of_path_tuples
        self.attention_region = attention_region
        self.attention_region_boundary = attention_region_boundary
        self.visualizations_parent_dir = visualizations_parent_dir
        self.num_img_to_use = num_img_to_use


    def define_storage(self):
        self.storage_list = []
        all_model_names = set("_".join(os.path.basename(path).split("_")[1:]).split(".")[0]
                              for path in self.list_of_path_tuples[0])

        self.model_separate_storage_dict = dict([(name, []) for name in all_model_names])

        _region_boundary_folder_ident = f"_{self.attention_region_boundary[0]}_" if\
            len(self.attention_region_boundary) == 1 else f"_{self.attention_region_boundary[0]}_{self.attention_region_boundary[1]}_"

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.graph_save_path = os.path.join(self.visualizations_parent_dir,
                                            f"{self.attention_region}" +
                                            f"{_region_boundary_folder_ident}" +
                                            f"{self.num_img_to_use}_{current_time}.png")


    def run(self):

        for paths in tqdm(self.list_of_path_tuples):
            for attention_map_path in paths:
                model_name = "_".join(os.path.basename(attention_map_path).split("_")[1:]).split(".")[0]
                print(f"Working on map {os.path.basename(attention_map_path)}")
                current_tensor = torch.load(attention_map_path)

                current_model_existing_stored_distances = self.model_separate_storage_dict[model_name]

                current_model_current_image_distances_for_every_layer = []
                for layer_index in range(current_tensor.size()[0]):
                    current_layer_mean_attention_distance =\
                        graphPlotter.compute_avg_scaled_distance(current_tensor[layer_index, :, :, :, :],
                                                                 attention_region = self.attention_region,
                                                                 attention_region_boundary = self.attention_region_boundary)
                    current_model_current_image_distances_for_every_layer.append(float(current_layer_mean_attention_distance))

                current_model_existing_stored_distances.append(current_model_current_image_distances_for_every_layer)

