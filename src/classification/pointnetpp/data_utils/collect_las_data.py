import os
import sys
from las_data_utils import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# so go one more directory up to get to the root directory
ROOT_DIR = os.path.dirname(BASE_DIR)


sys.path.append(BASE_DIR)  # add the path to the root directory to the system path

output_folder = os.path.join(ROOT_DIR, 'data', 'data_label')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

out_filename = 'classified_merged_points.npy'
collect_point_label(os.path.join(output_folder, out_filename), 'numpy')
