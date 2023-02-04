"""
Verification and Tracking
"""

"""
Import Packages
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
from glob import glob
import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--tracker_file_path', type=str, default='outputs/tile_img_annotation_annotator.npy',
                        help='The file path of the numpy array that contains thes the image tracking')
    parser.add_argument('--home_directory', type=str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk",
                        help='root path to the data commons storage space')
    parser.add_argument("--verifiers",  nargs="+", default=None,
                       help = "The names of researchers who will verify, in order of the verifications phase, coverage_quality_class")
    parser.add_argument("--annotator_allocation", nargs="+", default=str)
    parser.add_argument('--set_number', type=str, default=None,
                        help='The iteration of verification')
    args = parser.parse_args()
    return args

def main(args):  
    #update the location of the images in the tracker .csv + .npy if it already exists
    if os.path.exists(args.tracker_file_path):
        unverified_images = os.path.join(args.home_directory, "unverified_images", "student_reviewed_unverified_images_set" + args.set_number)
        tile_img_annotation_annotator = ap.update_path(unverified_images, args.tracker_file_path)
      
    folder_annotator_list, verification_dir = ap.verification_folders(args.home_directory, args.verifiers, args.annotator_allocation, args.set_number)
    tile_img_annotation_annotator = ap.seperate_images_for_verification_update_tracking(folder_annotator_list, verification_dir, 
                                                                                        args.set_number, tile_img_annotation_annotator)
    
    np.save('outputs/tile_img_annotation_annotator.npy', tile_img_annotation_annotator)
    
    column_names = ["tile_name", "chip_name", "chip pathway", "xml annotation", 
                    "annotator - draw","annotator - verify coverage",
                    "annotator - verify quality", "annotator - verify classes"]
    tile_img_annotation_annotator_df = pd.DataFrame(data = tile_img_annotation_annotator, 
                                                   index = tile_img_annotation_annotator[:,1], 
                                                   columns = column_names)
    tile_img_annotation_annotator_df.to_csv('outputs/tile_img_annotation_annotator_df.csv')

    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)



    
        

