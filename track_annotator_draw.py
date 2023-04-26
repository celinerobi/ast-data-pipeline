"""
Import Packages
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import os
from glob import glob
import src.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script records the annotator who has labeled each image')
    parser.add_argument('--tracker_file_path', type=str, default='outputs/tile_img_annotation_annotator.npy',
                        help='The file path of the numpy array that contains thes the image tracking')
    parser.add_argument('--parent_dir', type=str, default = "C:\chip_allocation",
                        help='path to parent directory, holding the annotation directories.')
    args = parser.parse_args()
    return args
    
def main(args): 
    img_anno = ap.img_path_anno_path(ap.list_of_sub_directories(args.parent_dir))
    tracking_array = ap.reference_image_annotation_file_with_annotator(img_anno, args.tracker_file_path) #load existing and update 
    
    #Save numpy array
    np.save('outputs/tile_img_annotation_annotator.npy', tracking_array)
    
    #Create a dataframe
    column_names = ["tile_name", "chip_name", "chip pathway", "xml annotation", 
                    "annotator - draw","annotator - verify coverage",
                    "annotator - verify quality", "annotator - verify classes"]
    tile_img_annotation_annotator_df = pd.DataFrame(data = tracking_array, 
                                                   index = tracking_array[:,1], 
                                                   columns = column_names)
    tile_img_annotation_annotator_df.to_csv('outputs/tile_img_annotation_annotator_df.csv')

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)



