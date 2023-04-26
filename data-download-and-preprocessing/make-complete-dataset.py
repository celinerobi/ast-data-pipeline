"""
Import Packages
"""
import os
import sys
import shutil
import xml.etree.ElementTree as et
import argparse
import tqdm
import fiona
import geopandas as gpd
import shapely
from PIL import Image
import numpy as np
from glob import glob
import src.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Create complete dataset of aerial imagery')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the annotation directory.')
    parser.add_argument('--xml_folder_name', type=str, default="chips_positive_corrected_xml",
                        help='name of folder in complete dataset directory that contains annotations ')
    parser.add_argument('--include_tiles', type=bool, default=False, 
                        help='include tiles (False), or do not include tiles (True) annotations')
    parser.add_argument('--complete_dir', type=str, default=False,
                        help='Path to directory to store complete dataset')
    parser.add_argument('--tile_level_annotation_dataset_path', type=str, 
                        default="/hpc/group/borsuklab/ast/tile-level-annotations/tile_level_annotations.geojson", 
                        help='path to dataset holding of tile level annotation')
    args = parser.parse_args()
    return args


def main(args):
    ### Get the subdirectories within the subdirectories (the folders from each of the allocations)
    sub_directories = list()
    for sub_directory in ap.list_of_sub_directories(args.parent_dir):
        for root, dirs, files in os.walk(sub_directory):
            if "chips_positive" in dirs:
                sub_directories.append(root)
    #identify images with annotations using tile level annotation dataset
    tile_level_annotations = gpd.read_file(args.tile_level_annotation_dataset_path)
    images_with_annotations = [item for images in tile_level_annotations.image_name for item in images]
    images_with_annotations = np.unique(images_with_annotations)
    images_with_annotations = set(images_with_annotations)  # this reduces the lookup time from O(n) to O(1)

    ### Move the annotations + images 
    counter_images = 0
    for i in tqdm.tqdm(range(len(sub_directories))):
        sub_directory = sub_directories[i].rsplit("/", 1)[1] # get the sub folders for each annotator
        print("The current subdirectory:", sub_directory)
        # Functions to move the annotations + images into folders
        dist = ap.annotator(sub_directory)
        dist.state_parent_dir(args.parent_dir)
        dist.make_subdirectories()    
        images = dist.move_images_annotations_to_complete_dataset(args.complete_dir, images_with_annotations,
                                                                               args.xml_folder_name)
        counter_images += images # count the number of images
        print(counter_images) # print the counters


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)