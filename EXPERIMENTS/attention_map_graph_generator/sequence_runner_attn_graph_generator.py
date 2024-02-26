import argparse
import glob
import os
import random

from objects.graph_plotter import graphPlotter

class PathReader:
    def __init__(self, folder_path, suffixes_to_consider):
        self.folder_path = folder_path
        self.suffixes_to_consider = suffixes_to_consider
        self.files_grouped_by_key = {}

    def read_files(self):
        # Get all .pt files in the folder
        files = glob.glob(os.path.join(self.folder_path, '*.pt'))
        # Group files by the part of the name before the first underscore
        for file in files:
            base_name = os.path.basename(file)
            group_key = base_name.split('_')[0]
            if group_key not in self.files_grouped_by_key:
                self.files_grouped_by_key[group_key] = {}
            suffix = "_".join(base_name.split('_')[1:]).split('.')[0]
            if suffix in self.suffixes_to_consider:
                self.files_grouped_by_key[group_key][suffix.split('.')[0]] = file

    def get_tuples(self):
        tuples_list = []
        for files in self.files_grouped_by_key.values():
            _tmp_tupple = tuple([files.get(_suffix, None) for _suffix in self.suffixes_to_consider])
            if None in _tmp_tupple:
                try:
                    print(f"We encountered an image that was not present for one of the suffixes "
                          f"{os.path.basename(list(files.values())[0])[0:8]}")
                    continue
                except:
                    print("Encountered an error")

            tuples_list.append(_tmp_tupple)
        return tuples_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group .pt files in a directory.')
    parser.add_argument('--folder-path', required=True, help='Path to the directory containing .pt files')
    parser.add_argument("--suffixes-to-consider", required=True, nargs='+', help="Paths to the dataset folders.")
    parser.add_argument("--attention-region", required=True, type=str,
                        help="The region in which to calculate attention distance."
                             "Possible values: 'global', 'center', 'periphery'")
    parser.add_argument("--attention-region-boundary", required=True, nargs='+', type=float,
                        help="The beginning of the mid-rectangle, counting from top-left corner."
                             "Used only for 'periphery' or 'center' attention regions."
                             "For periphery_strip, provide outer bound first then inner bound")
    parser.add_argument("--visualizations-parent-dir", required=True, type=str,
                        help="The main directory in which to save the graph")
    parser.add_argument("--num-img-to-use", required=True, type=int,
                        help="The number of images ot use for generating the graph")

    args = parser.parse_args()

    reader = PathReader(args.folder_path, args.suffixes_to_consider)
    reader.read_files()
    attention_maps_list = reader.get_tuples()
    random.Random(4).shuffle(attention_maps_list)

    graph_plotter_obj = graphPlotter(attention_maps_list[0:args.num_img_to_use],
                                     attention_region = args.attention_region,
                                     attention_region_boundary = args.attention_region_boundary,
                                     visualizations_parent_dir = args.visualizations_parent_dir,
                                     num_img_to_use = args.num_img_to_use)
    graph_plotter_obj.define_storage()
    graph_plotter_obj.run()
    graph_plotter_obj.plot_data(graph_plotter_obj.model_separate_storage_dict,
                                graph_plotter_obj.graph_save_path)

    print("Finished collecting attention maps")