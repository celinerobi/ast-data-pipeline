"""
Import Packages
"""
import shutil
import xml.etree.ElementTree as et
import argparse
import tqdm
import os
import sys
from PIL import Image
import numpy as np
from glob import glob
import data_eng.form_calcs as fc


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Correct inconsistent labels and reclassify tanks based on tanks size')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the img/annotation sub directories.')
    parser.add_argument('--orig_xml_folder_name', type=str, default="chips_positive_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    parser.add_argument('--corrected_xml_folder_name', type=str, default="chips_positive_corrected_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    args = parser.parse_args()
    return args


def main(args):
    tile_names = os.listdir(args.parent_dir)
    # Correct inconsistent labfolders_of_images_xmls_by_tileels
    for tile_name in tqdm.tqdm(tile_names):  # iterate over tile folders
        # get original and corrected xml directories
        orig_xml_dir = os.path.join(args.parent_dir, tile_name, args.orig_xml_folder_name)
        corrected_xml_dir = os.path.join(args.parent_dir, tile_name, args.corrected_xml_folder_name)

        for xml in os.listdir(orig_xml_dir):
            # get original and corrected xml paths
            orig_xml_path = os.path.join(orig_xml_dir, xml)
            corrected_xml_path = os.path.join(corrected_xml_dir, xml)
            # reformat xmls to add resolution/date/filename/path
            fc.reformat_xml_for_compiled_dataset(orig_xml_path)
            # correct inconsistent labels
            fc.correct_inconsistent_labels_xml(orig_xml_path, corrected_xml_path)
            # reclassify narrow/closed roof tanks based on size
            fc.reclassify_narrow_closed_roof_and_closed_roof_tanks(corrected_xml_path)


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)

