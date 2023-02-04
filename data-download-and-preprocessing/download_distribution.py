#Make into module, function specify annotator, directory as args
#URG: updae tile_undone_npy

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
import os.path
import urllib.request
import progressbar # pip install progressbar2, not progressbar

import os
import shutil

import argparse

import tempfile
import urllib
import shutil
import os
import os.path
import sys

import PIL
from PIL import Image

import math
import numpy as np
import pandas as pd
import rtree
import pickle

import progressbar # pip install progressbar2, not progressbar

from geopy.geocoders import Nominatim

from contextlib import suppress

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import warnings
from zipfile import ZipFile

import argparse
import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_dir', type=str, default="//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//",
                        help='path to parent directory; the directory of the storge space.')
    parser.add_argument('--annotation_dir', type=str, default="verified/complete_dataset",
                        help='path to the verified complete dataset.')
    parser.add_argument('--number_of_tiles', type=str, default='outputs/tile_img_annotation_annotator.npy',
                        help='The file path of the numpy array that contains the image tracking.')
    parser.add_argument('--tiles_remaining', type=str,
                        default="image_download_azure/tile_name_tile_url_complete_array.npy",
                        help='The file path of the numpy array that contains the tile names and tile urls of the complete arrays.')
    parser.add_argument('--tiles_labeled', type=str,
                        default="image_download_azure/tile_name_tile_url_complete_array.npy",
                        help='The file path of the numpy array that contains the tile names and tile urls of the complete arrays.')

    args = parser.parse_args()
    return args

def main(args):    
    dist = ap.annotator(args.annotation_dir) #create the processing class
    dist.state_dcc_directory(args.parent_dir)
    
    dist.number_of_tiles(args.number_of_tiles)
    
    dist.get_tile_urls(args.tiles_remaining)

    dist.make_subdirectories()
    dist.download_images()
    dist.tile_rename()
    dist.chip_tiles()
    
    dist.track_tile_annotations(args.tiles_labeled)
    np.save('tile_name_tile_url_remaining_expanded', dist.tile_name_tile_url_remaining)
    np.save('tile_name_tile_url_labeled', dist.tile_name_tile_url_labeled)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
