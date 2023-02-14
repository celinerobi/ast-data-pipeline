"""
Data Summary of Labeled Images (To - Date)
"""

"""
Import Packages
"""
from PIL import Image
import os
import pandas as pd
import numpy as np
import shutil
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
import argparse

import os
import sys
from glob import glob
import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--complete_dir', type=str, default=None,
                        help='path to complete dataset directory.')
    parser.add_argument('--annotation_dir', type=str, default=None,
                        help='Name of the folder containing the annotations for positive images of interests.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store output files.')
    args = parser.parse_args()
    return args


def main(args): 
    img_path = os.path.join(args.complete_dir, 'chips_positive')
    anno_path = os.path.join(args.complete_dir, args.annotation_dir)
    print(np.hstack((img_path, anno_path)))
    ap.dataset_summary_assessment(np.hstack((img_path, anno_path)), args.output_dir, multiple=False)

if __name__ == '__main__':
    args = get_args_parse()
    main(args)    