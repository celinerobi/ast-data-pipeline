#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import shutil
import urllib
import urllib.request
import tempfile
import sys
import math
import rtree
import pickle
import warnings
from zipfile import ZipFile
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from contextlib import suppress
import progressbar # pip install progressbar2, not progressbar
import argparse
import PIL
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import data_eng.az_proc as ap

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='allocate images for annotation')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='Path to parent dir; see dist_folder_map.txt for directory map.')
    parser.add_argument('--img_anno_dir', type=str, default=None,
                        help='Path to dir holding imgs and annotations; see dist_folder_map.txt for directory map.')
    parser.add_argument('--num_tiles', type=int, 
                        help='Number of tiles to be chipped and allocated.')
    parser.add_argument('--item_dim', type=str, default= None,
                        help='dimension of chipped image. Assumed square.')
    parser.add_argument('--tiles_remaining', type=str, default=None,
                        help='File path to numpy array that contains the remaining tile names and tile urls')
    parser.add_argument('--tiles_labeled', type=str, default= None,
                        help='File path to numpy array that contains the labeled the tile names and tile urls')

    args = parser.parse_args()
    return args

def main(args):    
    dist = ap.annotator(args.img_anno_dir) #create the processing class
    dist.state_dcc_directory(args.parent_dir)
    dist.number_of_tiles(args.num_tiles)
    dist.get_tile_urls(args.tiles_remaining)
    dist.make_subdirectories()
    dist.download_images()
    dist.tile_rename()
    dist.chip_tiles(args.item_dim)
    dist.track_tile_annotations(args.tiles_labeled)
    np.save(args.tiles_remaining, dist.tile_name_tile_url_remaining)
    np.save(args.tiles_labeled, dist.tile_name_tile_url_labeled)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
