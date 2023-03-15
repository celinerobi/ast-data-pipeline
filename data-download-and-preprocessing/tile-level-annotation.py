"""
Create pandas csv of image characteristics  
"""

"""
Import Packages
"""
help("modules")
import os
import argparse
import cv2
import math
from glob import glob
# Standard packages
from datetime import datetime
import tempfile
import warnings
import urllib
import shutil
import pickle
# import requests
from PIL import Image
from io import BytesIO
import tqdm
from tqdm.notebook import tqdm_notebook
from skimage.metrics import structural_similarity as compare_ssim
import random
import numpy as np
import fiona  # must be import before geopandas
import geopandas as gpd
import rasterio
import rioxarray
import re
import rtree
import pyproj
import shapely
from shapely.geometry import Polygon, Point
from shapely.ops import transform
# from cartopy import crs
import collections
# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Parsing/Modifying XML
from lxml.etree import Element, SubElement, tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom

import data_eng.az_proc as ap
import data_eng.form_calcs as fc


def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_dir', type=str, default=None,
                        help='path to parent directory, holding the img/annotation sub directories.')
    parser.add_argument('--tile_dir', type=str, default=None,
                        help='path to directory holding tiles.')
    parser.add_argument('--tile_level_annotation_dir', type=str, default=None,
                        help='path to directory which holds tile level annotation and related files.')
    parser.add_argument('--tile_level_annotation_dataset_filename', type=str, default="tile_level_annotation",
                        help='File name of tile level annotation')
    parser.add_argument('--item_dim', type=int, default=512,
                        help='Dimensions of image (assumed sq)')
    parser.add_argument('--distance_limit', type=int, default=int(5),
                        help='The maximum pixel distance between bbox adjacent images, to merge')
    parser.add_argument('--county_gpd_path', type=str, default=None,
                        help='Path to dataset of county boundaries')
    parser.add_argument('--xml_folder_name', type=str, default="chips_positive_corrected_xml",
                        help="name of folder in complete dataset directory that contains annotations")
    parser.add_argument('--output_dir', type=str, 
                        default="/hpc/home/csr33/ast-dataset/data-download-and-preprocessing/outputs",
                        help="output directory")
    args = parser.parse_args()
    return args


def main(args):
    # Consider update to make empty dataframe, then iterate over tiles to add in characteristics to daframe
    # and generate tile xmls over iterations
    # Generate table of characteristics for tiles/images
    # change where they are written
    start_time = datetime.now()
    
    #image/tile characteristics 
    tile_characteristics, image_characteristics = fc.image_tile_characteristics(args.parent_dir, args.tile_dir,
                                                                                args.xml_folder_name, args.output_dir)
    img_tile_char_time = datetime.now()
    print('Duration For Image Tile Characteristics: {}'.format(img_tile_char_time - start_time))

    # Generate tile level XMLs
    tiles_xml_dir = os.path.join(args.tile_level_annotation_dir, "tiles_xml")
    os.makedirs(tiles_xml_dir, exist_ok=True)
    fc.generate_tile_xmls(args.parent_dir, args.tile_dir, args.xml_folder_name, tiles_xml_dir, args.item_dim)
    tile_xmls_time = datetime.now()
    print('Duration For Tile XMLs: {}'.format(tile_xmls_time - img_tile_char_time))
    
    # Merge neighboring bounding boxes within each tile
    tile_database = fc.merge_tile_annotations(tile_characteristics, tiles_xml_dir, distance_limit=args.distance_limit)
    merge_tile_database_time = datetime.now()
    print('Duration For Tile Database: {}'.format(merge_tile_database_time - tile_xmls_time))
    
    # Add in state
    county_data = fc.read_shp_from_zip(args.county_gpd_path)
    tile_database = fc.identify_political_boundaries_for_each_tank(county_data, tile_database)
    political_boundaries_time = datetime.now()
    print('Duration For Adding the state: {}'.format(political_boundaries_time - merge_tile_database_time))
    
    # add quadid and capture data 
    #tile_level_annotations.tile_name[2:12]
    #df.applymap(lambda x: x**2)
    tile_database["quadid"] = tile_database.apply(lambda row: row.tile_name[2:12], axis=1)
    tile_database["capturedate"] = tile_database.apply(lambda row: row.tile_name.rsplit('_',1)[1], axis=1)
    
    # Save tile dabasebase
    fc.write_gdf(tile_database, args.tile_level_annotation_dir, args.tile_level_annotation_dataset_filename)
    end_time = datetime.now()
    print('Duration For creating the tile level annotations: {}'.format(end_time - start_time))

if __name__ == '__main__':
    args = get_args_parse()
    main(args)
