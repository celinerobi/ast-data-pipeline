"""
Import Packages
"""
import os
import shutil
import sys
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
from lxml.etree import Element,SubElement,tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
import argparse
import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Creates a data summary of labeled images including the number of closed_roof_tanks, water_treatment_tank, spherical_tank, external_floating_roof_tank, water_tower for all of the images.')
    parser.add_argument('--complete_dir', type=str, default=None,
                        help='path to complete dataset directory.')
    parser.add_argument('--annotation_dir', type=str, default=None,
                        help='Name of the folder containing the annotations for positive images of interests.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store output files.')
    args = parser.parse_args()
    return args


def main(args): 
    img_dir = os.path.join(args.complete_dir, 'chips_positive')
    anno_dir = os.path.join(args.complete_dir, args.annotation_dir)
    ap.dataset_summary(img_dir, anno_dir, args.output_dir)

if __name__ == '__main__':
    args = get_args_parse()
    main(args)    