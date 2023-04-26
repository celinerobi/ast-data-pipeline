"""
Module containing azure functions, and label work distribution and image processing functions.
"""

"""
Load Packages
"""
#conda install -c conda-forge gdal fiona rasterio pyproj geopy geopandas scipy shapely matplotlib tqdm rioxarray scikit-image lxml numpy pandas jupyterlab jupyter
#pip install opencv-python progressbar2 regex Rtree Pillow

# Standard modules
import tempfile
import warnings
import urllib
import urllib.request
import shutil
import os
import os.path
from pathlib import Path
import sys
from zipfile import ZipFile
import pickle
import math
from contextlib import suppress
from glob import glob
import re

# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import progressbar  
from tqdm import tqdm
import cv2

# Parsing/Modifying XML
from lxml.etree import Element, SubElement, tostring
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom
import xml

# Image processing files
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.image as mpimg
import rtree
#import shapely
from geopy.geocoders import Nominatim
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import src.form_calcs as fc


"""
Find file paths
"""
def list_of_sub_directories(path_to_images): 
    """
    Define a function to create a list of the directories in the storage space containing images
    Find the subdirectories containing images
    """

    sub_directories = []  # initialize list

    for folder in os.listdir(path_to_images):  # identifies the subfolders
        d = path_to_images + "/" + folder  # creates the complete path for each subfolder
        if os.path.isdir(d):
            sub_directories.append(d)  # adds the subfolder to the list

    return sub_directories


def img_path_anno_path(sub_directories): 
    """
    ### Define a function to create a list of the annotation and positive_chip paths for each of the subdirectories 
        "Create an array of the paths to the folders containing the images and annotation given a subdirectory"
    Only create paths for subdirectories that have these paths and for subdirectories that are correctly formated (Qianyu's thesis, etc.)
    """

    img_path = []
    anno_path = []

    for i in range(len(sub_directories)):
        if "chips" in os.listdir(sub_directories[i]):
            img_path.append(sub_directories[i] + "/" + "chips_positive")
            anno_path.append(sub_directories[i] + "/" + "chips_positive_xml")
        elif "chips_positive" in os.listdir(sub_directories[i]):
            img_path.append(sub_directories[i] + "/" + "chips_positive")
            anno_path.append(sub_directories[i] + "/" + "chips_positive_xml")
        else:
            for ii in range(len(os.listdir(sub_directories[i]))):
                img_path.append(sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive")
                anno_path.append(
                    sub_directories[i] + "/" + os.listdir(sub_directories[i])[ii] + "/" + "chips_positive_xml")

    img_annotation_path = np.empty((1, 2))  # form a numpy array
    for i in range(len(img_path)):
        if os.path.isdir(img_path[i]) == True:
            img_annotation_path = np.vstack((img_annotation_path,
                                             [img_path[i], anno_path[i]]))
    img_annotation_path = np.delete(img_annotation_path, 0, axis=0)  # 0 removes empty row
    return img_annotation_path

"""
Azure Functions
"""
class DownloadProgressBar():
    """
    A progressbar to show the completed percentage and download speed for each image downloaded using urlretrieve.

    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """

    tile_rtree = None
    tile_index = None
    base_path = None

    def __init__(self, base_path=None):
        blob_root = 'https://naipeuwest.blob.core.windows.net/naip'
        index_files = ["tile_index.dat", "tile_index.idx", "tiles.p"]
        index_blob_root = re.sub('/naip$', '/naip-index/rtree/', blob_root)

        if base_path is None:

            base_path = os.path.join(tempfile.gettempdir(), 'naip')
            os.makedirs(base_path, exist_ok=True)

            for file_path in index_files:
                download_url(index_blob_root + file_path, destination_folder=base_path,
                             destination_filename=base_path + '/' + file_path,
                             progress_updater=DownloadProgressBar())

        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path + "/tiles.p", "rb"))

    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns a list of COG file paths.
        """

        point = shapely.geometry.Point(float(lon), float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point):
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            print('''Error: there are overlaps with tile index, 
                      but no tile completely contains selection''')
            return None
        elif len(intersected_files) <= 0:
            print("No tile intersections")
            return None
        else:
            return intersected_files


def download_url(url, destination_folder, destination_filename=None, progress_updater=None, force_download=False): 
    """
    Download a URL to a a file
    Args:
    url(str): url to download
    destination_folder(str): directory to download folder
    destination_filename(str): the name for each of files to download
    return:
    destination_filename
    """

    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is not None:
        destination_filename = os.path.join(destination_folder, destination_filename)
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = os.path.join(destination_folder, url_as_filename)
    if os.path.isfile(destination_filename):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    #  print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))

    return destination_filename


"""
# Functions to retrieve filepathways from lat lon data
"""
def lons_lat_to_filepaths(lons, lats, index): 
    """
    Calculate file paths given lat and lat
    """
    all_paths = np.empty(shape=(1, 8))
    for lon, lat in tqdm(zip(lons, lats)):
        naip_file_pathways = index.lookup_tile(lat, lon)
        if naip_file_pathways != None:
            select_path = []
            for file_pathway in naip_file_pathways:
                tmp = file_pathway.split('/')
                tmp = np.hstack((tmp, file_pathway.split('/')[3].split("_")[1]))
                quad_id = iter(tmp[5].split("_", 4))
                quad_id = list((map("_".join, zip(*[quad_id] * 4))))
                tmp = np.hstack((tmp, quad_id))
                select_path.append(tmp)
            select_path = np.array(select_path)
             # filter out years to get the most recent data that will include the highest resolution data
            select_path = select_path[select_path[:, 2] >= "2018"] 
            # select only pathways with 60cm
            select_path = select_path[(select_path[:, 6] == "60cm") | (select_path[:, 6] == "060cm")]  
            all_paths = np.vstack((all_paths, select_path))  # add to the rest of the paths

    file_pathways = np.delete(all_paths, 0, axis=0)
    file_pathways = np.unique(file_pathways, axis=0)  # select unique values
    return file_pathways


def filepaths_to_tile_name_tile_url(file_pathways, blob_root): 
    """
    Determine the tile name and url for a given file pathway
    # Tiles are stored at: [blob root]/v002/[state]/[year]/[state]_[resolution]_[year]/[quadrangle]/filename
    """
    tile_name = []
    tile_url = []
    for index, row in file_pathways.iterrows():
        tile_name.append(row["tile_name"])
        tile_url.append(os.path.join(blob_root, row["version"], row["state"], row["year"],
                                     row["state_res_year"], row["lat_lon"], row["tile_name"]))
    return (tile_name, tile_url)

def get_urls_from_lat_lons(lat_lon_dir, index, blob_root): 
    all_dataset_pathways = pd.DataFrame()
    lat_lon_paths = glob(lat_lon_dir + "/*.csv")
    
    #get pathways from dataset
    col = ["version","state","year","state_res_year","lat_lon","tile_name","resolution","quad"]
    for lat_lon_path in lat_lon_paths:
        lat_lon_dataset = pd.read_csv(lat_lon_path) #read in sheet of quadrangles
        if set(['X','Y']).issubset(lat_lon_dataset.columns):
            lons = lat_lon_dataset["X"].tolist()
            lats = lat_lon_dataset["Y"].tolist()
            pathways = lons_lat_to_filepaths(lons, lats, index)
            pathways = pd.DataFrame(pathways, columns = col)
            all_dataset_pathways = pd.concat([all_dataset_pathways, pathways])

        elif set(['LONGITUDE','LATITUDE']).issubset(lat_lon_dataset.columns):
            lons = lat_lon_dataset["LONGITUDE"].tolist()
            lats = lat_lon_dataset["LATITUDE"].tolist()
            pathways = lons_lat_to_filepaths(lons, lats, index)
            pathways = pd.DataFrame(pathways, columns = col)
            all_dataset_pathways = pd.concat([all_dataset_pathways, pathways])
        else:
            print("no lon, lat match:", lat_lon_path)
            
    #get tile names and urls   
    all_dataset_pathways = all_dataset_pathways.drop_duplicates()
    tile_names, tile_urls = filepaths_to_tile_name_tile_url(all_dataset_pathways, blob_root)
    tile_names_tile_urls = pd.DataFrame({"tile_name": tile_names, "tile_url": tile_urls})
    return(tile_names_tile_urls)


"""
Function to retrieve file pathways from quads
"""


def collected_quads_to_tile_name_tile_url(quads, blob_root): 
    """
    #Tiles are stored at: [blob root]/v002/[state]/[year]/[state]_[resolution]_[year]/[quadrangle]/filename
    #Read in a excel sheet which includes the quadrangle 
    """

    tile_names = []
    tile_urls = []
    file_name_index = {'m': 0, 'qqname': 1, 'direction': 2, 'YY': 3, 'resolution': 4, 'capture_date': 5,
                       'version_date': 5}
    two_digit_state_resolution = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
                                  "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
                                  "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
                                  "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
                                  "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]

    for index, row in quads.iterrows():
        file_name = row[0].split('_')  # filename
        state = row[1].lower()  # state
        year = row[2]  # YYYY

        if state in two_digit_state_resolution:
            resolution = file_name[file_name_index["resolution"]][1:3] + "cm"
        else:
            resolution = file_name[file_name_index["resolution"]] + "cm"
        quadrangle = file_name[file_name_index["qqname"]][0:5]  # qqname

        tile_name = file_name + '.tif'
        tile_names.append(tile_name)
        tile_urls.append(os.path.join(blob_root, "v002", state, str(year), 
                                      state + '_' + str(resolution) + '_' + str(year),
                                      str(quadrangle), tile_name))
    return (tile_names, tile_urls)

def get_urls_from_quads(quad_dir): 
    tile_names_tile_urls = pd.DataFrame() #dataframe to hold all tile_names and urls
    quad_paths = glob(quad_dir + "/*.csv")
    for quad_path in quad_paths:
        quad_dataset = pd.read_csv(quad_path) #read in sheet of quadrangles
        tile_names, tile_urls = collected_quads_to_tile_name_tile_url(quad_dataset, blob_root) # identify filespaths/urls for quads of interest
        quad_names_urls = pd.DataFrame({"tile_name": tile_names, "tile_url": tile_urls})
        tile_names_tile_urls = pd.concat([tile_names_tile_urls, quad_names_urls])
    return tile_names_tile_urls


def tile_name_tile_url_characteristics(tile_names_tile_urls, output_path = None): 
    """tabulates the tile characteristics (the states, year resolution ranges), 
      returns the tile charcateristics 
       (quadrange names, the filenames,the states, year resolution ranges)

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """

    state_array = np.empty((len(tile_names_tile_urls), 1), dtype=object)
    year_array = np.empty((len(tile_names_tile_urls), 1))
    quad_array = np.empty((len(tile_names_tile_urls), 1))
    resolution_array = np.empty((len(tile_names_tile_urls), 1), dtype=object)
    filename_array = np.empty((len(tile_names_tile_urls), 1), dtype=object)

    for i in range(len(tile_names_tile_urls)):
        state_array[i] = tile_names_tile_urls[i, 1].split('/')[5]
        year_array[i] = tile_names_tile_urls[i, 1].split('/')[6]
        quad_array[i] = tile_names_tile_urls[i, 1].split('/')[8]
        filename_array[i] = tile_names_tile_urls[i, 1].split('/')[9]
        resolution_array[i] = tile_names_tile_urls[i, 1].split('/')[-3].split('_')[1]

    state_abbreviations = np.unique(state_array)
    num_states = len(state_abbreviations)
    years = np.unique(year_array)
    resolutions = np.unique(resolution_array)

    print("the number of tiles includes", len(tile_names_tile_urls))
    print("The number of states included", num_states)
    print("Postal abriviations of the states included", state_abbreviations)
    print("The years in which the images were collected", years)
    print("The resolutions of the images", resolutions)
    if output_path is not None:
        np.save(os.path.join(output_path, 'states_in_tile_urls.npy'), state_abbreviations)

    return num_states, state_abbreviations, years, resolutions, quad_array, filename_array


"""
Tile distribution functions
"""


class annotator: 
    def __init__(self, img_anno_dir): 
        self.img_anno_dir = img_anno_dir

    def state_parent_dir(self, parent_dir): 
        #formerly state_dcc_directory
        self.parent_dir = parent_dir

    def number_of_tiles(self, num_tiles): 
        self.num_tiles = num_tiles

    def get_tile_urls(self, tile_name_tile_url_unlabeled): 
        """
        self.tile_name_tile_url_unlabeled: npy array of the initial tiles that have not been labeled
        self.tile_name_tile_url_tiles_for_annotators: npy array of the tiles to be allocated to the annotator 
        """

        self.tile_name_tile_url_unlabeled = np.load(
            tile_name_tile_url_unlabeled)  # the tiles that have not yet been labeled to date
        print("Unlabeled Tiles", self.tile_name_tile_url_unlabeled.shape)
        self.tile_name_tile_url_tiles_for_annotators = self.tile_name_tile_url_unlabeled[range(self.num_tiles),
                                                       :]  # create an array of the tiles that will be allocated to this annotator

        self.tile_url = self.tile_name_tile_url_tiles_for_annotators[:,
                        1]  # get the urls of the tiles that will allocated to the annotator

    def track_tile_annotations(self, tile_name_tile_url_labeled): 
        """
        self.tile_name_tile_url_remaining: npy array of the remaining tiles to be annotated; 
            this will then be passed in the next iteration
        self.tile_name_tile_url_labeled: npy array of the tiles labeled
        """
        # the tiles that have not yet been labeled to date
        self.tile_name_tile_url_labeled = np.load(tile_name_tile_url_labeled)  
        self.tile_name_tile_url_labeled = np.concatenate((self.tile_name_tile_url_labeled, 
                                                          self.tile_name_tile_url_tiles_for_annotators), 
                                                          axis=0)
        print("Labeled Tiles", self.tile_name_tile_url_labeled.shape)
        
        # the numpy array of the remaining tiles
        self.tile_name_tile_url_remaining = np.delete(self.tile_name_tile_url_unlabeled, 
                                                      range(self.num_tiles),0)  
        # (remove the tiles that the annotator is labeling)
        print(self.tile_name_tile_url_remaining.shape)

        if len(self.tile_name_tile_url_tiles_for_annotators) + len(self.tile_name_tile_url_remaining) != len(
                self.tile_name_tile_url_unlabeled):
            raise Exception("The number of remaining tiles and the tiles allocated to annotaters is less \
                             than the number of tiles passed through this function")

    def make_subdirectories(self): 
        self.img_anno_dir = os.path.join(self.parent_dir, self.img_anno_dir)
        os.makedirs(self.img_anno_dir, exist_ok=True)

        self.tiles_dir = os.path.join(self.img_anno_dir, 'tiles')  # directory for the naip data
        os.makedirs(self.tiles_dir, exist_ok=True)

        self.chips_dir = os.path.join(self.img_anno_dir, 'chips')  # directory to hold chips that are clipped from naip tiles
        os.makedirs(self.chips_dir, exist_ok=True)

        self.chips_positive_dir = os.path.join(self.img_anno_dir, 'chips_positive')  # directory to hold chips with tanks
        os.makedirs(self.chips_positive_dir, exist_ok=True)

        self.chips_negative_dir = os.path.join(self.img_anno_dir, 'chips_negative')  # directory to hold chips with tanks
        os.makedirs(self.chips_negative_dir, exist_ok=True)

        self.chips_xml_dir = os.path.join(self.img_anno_dir, 'chips_positive_xml')  # directory to hold xml files
        os.makedirs(self.chips_xml_dir, exist_ok=True)

        # Make directory to store all xml after correction
        self.chips_positive_corrected_xml_dir = os.path.join(self.img_anno_dir, "chips_positive_corrected_xml")
        os.makedirs(self.chips_positive_corrected_xml_dir, exist_ok=True)

    def download_images(self): 
        destination_of_filenames = []  # use so that we can index over the file names for processing later
        for i in range(self.num_tiles):
            print(i)
            destination_of_filenames.append(download_url(self.tile_url[i], self.tiles_dir,
                                                         progress_updater=DownloadProgressBar()))
        return destination_of_filenames

    def tile_rename(self): 
        """Rename all the tiles into the standard format outlined in repo readme 
        """
        self.tile_names = os.listdir(self.tiles_dir)  # get a list of all of the tiles in tiles directory
        print(self.tile_names)

        for tile_name in self.tile_names:
            tile_name_split = tile_name.split('_')
            old_tile_path = os.path.join(self.tiles_dir, tile_name)
            new_tile_path = os.path.join(self.tiles_dir,
                                         tile_name_split[6] + '_' + tile_name_split[7] + '_' + tile_name_split[
                                             8] + '_' + tile_name_split[9] + '_' + \
                                         tile_name_split[10] + '_' + tile_name_split[11] + '_' + tile_name_split[
                                             12] + '_' + tile_name_split[13] + '_' + \
                                         tile_name_split[14] + '_' + tile_name_split[15].split(".")[0] + ".tif")

            if os.path.isfile(new_tile_path):
                print('Bypassing download of already-downloaded file {}'.format(os.path.basename(new_tile_path)))

            else:
                os.rename(old_tile_path, new_tile_path)


    def chip_tiles(self, item_dim): 
        """Segment tiles into item_dim x item_dim pixel chips, preserving resolution
        """
        item_dim = int(512)
        print("chip tiles")
        self.tile_names = os.listdir(self.tiles_dir)  # get a list of all of the tiles in tiles directory
        for tile_name in self.tile_names:  # index over the tiles in the tiles_dir
            print(tile_name)
            tile_name_wo_ext, ext = os.path.splitext(tile_name)  # File name
            tile = cv2.imread(os.path.join(self.tiles_dir, tile_name))
            tile_height, tile_width, tile_channels = tile.shape  # the size of the tile
            # divide the tile into item_dim by item_dim chips (rounding up)
            row_index = math.ceil(tile_height / item_dim)
            col_index = math.ceil(tile_width / item_dim)

            count = 0
            for y in range(0, row_index):
                for x in range(0, col_index):
                    # 
                    # specify the path to save the image
                    chip_img = fc.tile_to_chip_array(tile, x, y, item_dim) #chip tile
                    chip_name = tile_name_wo_ext + '_' + f"{y:02}" + '_' + f"{x:02}" + '.jpg'  #
                    chips_save_path = os.path.join(self.chips_dir, chip_name)  # row_col.jpg                    
                    cv2.imwrite(os.path.join(chips_save_path), chip_img) # save image
                    count += 1
            print(count)

    def copy_positive_images(self): 
        """seperate out positive chips into specific directory.
        """

        # Input .xml files' names
        print("it ran")
        for annotation in os.listdir(self.chips_xml_dir):  # iterate through the annotations
            annotation_filename = os.path.splitext(annotation)[0]

            for image in os.listdir(self.chips_dir):  # iterate through the images
                image_filename = os.path.splitext(image)[0]
                if image_filename == annotation_filename:
                    shutil.copy(os.path.join(self.chips_dir, image),
                                self.chips_positive_dir)  # copy images with matching .xml files in the "chips_tank" folder
        print("it finished")

    def copy_negative_images(self): 
        """seperate out negative chips into specific directory.
        """

        print("it ran")
        for image in os.listdir(self.chips_dir):
            shutil.copy(os.path.join(self.chips_dir, image),
                        self.chips_negative_dir)  # copy all chips into negative folder

        for annotation in os.listdir(self.chips_xml_dir):
            annotation_filename = os.path.splitext(annotation)[0]

            for image in os.listdir(self.chips_dir):
                image_filename = os.path.splitext(image)[0]
                if image_filename == annotation_filename:
                    os.remove(os.path.join(self.chips_negative_dir,
                                           image))  # delete positive images according to the .xml files
        print("it finished")


    def move_images_annotations_to_complete_dataset(self, complete_dir, images_with_annotations,
                                                    xml_folder_name="chips_positive_corrected_xml"): 
        """make a complete datasetseperate out all of the positive chips, annotations, and conditionally tiles from one directory into a new folder.
        Args:
            file_loc (str): The file location of the spreadsheet
            original include_tiles (bool; default = True): Specifies whether the original annotation in chips positive or the corrected
                                                          annotation in chips_positive_xml should be used
            xml_folder_name(str): 
                                   #remove at later date, kept to ensure it is note dependent elsewhere
        Returns:
            len(annotations): number of annotations
            len(images): number of images
        """   
        #specify + make directory to hold chips
        self.complete_dataset_chips_dir = os.path.join(complete_dir, "chips_positive")
        os.makedirs(self.complete_dataset_chips_dir, exist_ok=True)  # directory to hold xml files

        # specify + make directory to hold  annotations
        if xml_folder_name == "chips_positive_xml":
            annotations_path = self.chips_xml_dir
            self.complete_dataset_xml_dir = os.path.join(complete_dir, "chips_positive_xml")
        if xml_folder_name == "chips_positive_corrected_xml":
            annotations_path = self.chips_positive_corrected_xml_dir
            self.complete_dataset_xml_dir = os.path.join(complete_dir, "chips_positive_corrected_xml")
        else:
            print("error")
        os.makedirs(self.complete_dataset_xml_dir, exist_ok=True) 

        #identify image names
        fc.remove_thumbs(self.chips_positive_dir) # remove thumbs
        images = os.listdir(self.chips_positive_dir)
        image_names = [os.path.splitext(item)[0] for item in images]
        labeled_image_names = [item for item in image_names if item in images_with_annotations]
        
        # Move annotations + images
        for image_name in labeled_image_names:
            shutil.copy(os.path.join(annotations_path, image_name + ".xml"), 
                        self.complete_dataset_xml_dir) # copy annotations
            shutil.copy(os.path.join(self.chips_positive_dir, image_name + ".jpg"), 
                        self.complete_dataset_chips_dir) # copy images
        return len(labeled_image_names)


"""
Tracking and Verification 
"""


def reference_image_annotation_file_with_annotator(img_annotation_path,
                                                   tracker_file_path='outputs/tile_img_annotation_annotator.npy'): 
    """
    Track image annotations
    """
    if os.path.isfile(tracker_file_path):  # check if the tracking file exists
        print("Initializing annotation tracking array; add new annotations to tracking array")
        tile_img_annotation_annotator = np.load(tracker_file_path)  # load existing
    else:
        print("Create new tracking array")
        tile_img_annotation_annotator = np.empty((1, 8))  # form a numpy array

    for i in range(len(img_annotation_path)):  # index over each folder
        print(img_annotation_path[i, 0])
        # img files + image_file_pathways
        img_files = []  # pull the files in the img folder
        img_file_pathways = []  # pull the files in the img folder
        for image in os.listdir(img_annotation_path[i, 0]):  # iterate through the images
            if image.endswith(".jpg"):
                img_files.append(image)
                img_file_pathways.append(os.path.join(img_annotation_path[i, 0]))

        # sort so that the paths/file names match
        img_file_pathways = sorted(img_file_pathways)
        img_files = sorted(img_files)
        num_img_files = len(img_files)

        # tiles
        tiles = []  # create a list of the tile names
        for image in img_files:  # iterate through the images
            tiles.append(image.rsplit("_", 1)[0])

        # annotation files
        anno_files = sorted(os.listdir(img_annotation_path[i, 1]))  # pull the files in the annotation folder

        # annotator
        path = Path(
            img_annotation_path[i, 0]).parent.absolute()  # get root path of chips postive/chips postive xml folder
        annotator = str(path).rsplit('\\')[-2]  # get the annotator name from the root path
        annotator_list = [annotator] * len(anno_files)

        # annotator - verify coverage
        annotator_verify_coverage = [""] * num_img_files

        # annotator - verify coverage
        annotator_verify_quality = [""] * num_img_files

        # annotator - verify coverage
        annotator_verify_classes = [""] * num_img_files

        tile_img_annotation_annotator = np.vstack((tile_img_annotation_annotator,
                                                   np.column_stack(
                                                       [tiles, img_files, img_file_pathways, anno_files, annotator_list,
                                                        annotator_verify_coverage, annotator_verify_quality,
                                                        annotator_verify_classes])))

    if not os.path.isfile(tracker_file_path):  # if the file does not exist; remove the initalizing dummy array
        tile_img_annotation_annotator = np.delete(tile_img_annotation_annotator, 0, axis=0)  # 0 removes empty row

    return tile_img_annotation_annotator


def update_path(path, tracker_file_path): 
    """
    If the verfification has not yet been completed, update the image/xml path
    """
    img_annotation_path = img_path_anno_path(list_of_sub_directories(path))

    # get the correct img files + image_file_pathways
    img_files = []  # pull the files in the img folder
    img_file_pathways = []  # pull the files in the img folder
    for i in range(len(img_annotation_path)):  # index over each folder
        for image in os.listdir(img_annotation_path[i, 0]):  # iterate through the images
            if image.endswith(".jpg"):
                img_file_pathways.append(os.path.join(img_annotation_path[i, 0].rsplit("/", 1)[0]))
                img_files.append(image)
    imgs_and_pathways = np.array(list(zip(img_file_pathways, img_files)))

    # replace incorrect pathways
    tile_img_annotation_annotator = np.load(tracker_file_path)
    for i in range(len(tile_img_annotation_annotator)):  # i - index for tracker .npy
        for ii in np.where(imgs_and_pathways[:, 1] == tile_img_annotation_annotator[i, 1])[
            0]:  # find the same images, (ii -index for img and pathway array)
            if imgs_and_pathways[ii, 0] != tile_img_annotation_annotator[i, 2]:
                tile_img_annotation_annotator[i, 2] = imgs_and_pathways[ii, 0]

    np.save('outputs/tile_img_annotation_annotator.npy', tile_img_annotation_annotator)
    column_names = ["tile_name", "chip_name", "chip pathway", "xml annotation",
                    "annotator - draw", "annotator - verify coverage",
                    "annotator - verify quality", "annotator - verify classes"]
    tile_img_annotation_annotator_df = pd.DataFrame(data=tile_img_annotation_annotator,
                                                    index=tile_img_annotation_annotator[:, 1],
                                                    columns=column_names)
    tile_img_annotation_annotator_df.to_csv('outputs/tile_img_annotation_annotator_df.csv')
    return tile_img_annotation_annotator


def verification_folders(home_directory, folder_name, annotator_allocation, set_number): 
    """
    Create folder for workers to verify images
    Args:
    """
    # create verification folder
    verification_dir = os.path.join(home_directory, 'verification_set' + set_number)
    os.makedirs(verification_dir, exist_ok=True)

    # pair folder name with annotors
    print(folder_name[0])
    ##create verification subfolder for each group
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0] + "_" + set_number),
                exist_ok=True)  # verification folder for each group
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0] + "_" + set_number, "chips_positive"),
                exist_ok=True)  # image sub folder
    os.makedirs(os.path.join(verification_dir, "verify_" + folder_name[0] + "_" + set_number, "chips_positive_xml"),
                exist_ok=True)  # xml sub folder
    folder_annotator_list = [folder_name[0], annotator_allocation]
    return (folder_annotator_list, verification_dir)


def seperate_images_for_verification_update_tracking(folder_annotator_list, verification_dir, set_number,
                                                     tile_img_annotation_annotator): 
    """
    Move images to verifcation folder
    """
    print("folder", folder_annotator_list[0])  # the current folder
    count = 0
    for i in range(len(folder_annotator_list[1])):  # iterate over annotator
        print("annotator", folder_annotator_list[1][i])  # the current annotator
        for ii in np.where(tile_img_annotation_annotator[:, 4] == folder_annotator_list[1][i])[0]:
            if len(tile_img_annotation_annotator[ii, 5]) == 0:
                tile_img_annotation_annotator[ii, 5] = folder_annotator_list[0].split("_")[0].capitalize()  # coverage
                tile_img_annotation_annotator[ii, 6] = folder_annotator_list[0].split("_")[1].capitalize()  # quality
                tile_img_annotation_annotator[ii, 7] = folder_annotator_list[0].split("_")[2].capitalize()  # class

                shutil.copy(os.path.join(tile_img_annotation_annotator[ii, 2], "chips_positive",
                                         tile_img_annotation_annotator[ii, 1]),
                            os.path.join(verification_dir, "verify_" + folder_annotator_list[0] + "_" + set_number,
                                         "chips_positive"))  # copy images

                shutil.copy(os.path.join(tile_img_annotation_annotator[ii, 2], "chips_positive_xml",
                                         tile_img_annotation_annotator[ii, 3]),
                            os.path.join(verification_dir, "verify_" + folder_annotator_list[0] + "_" + set_number,
                                         "chips_positive_xml"))  # copy annotations

                count += 1  # count the files allocated to each
        print(count)
    return tile_img_annotation_annotator


"""
Review Characteristics 
"""
def dataset_summary(img_dir, anno_dir, output_dir): 
    ### Define function to count the number of objects in each category
    """Get summary of the whole dataset

    Args: 
        img_dir (str): The path of the folder containing original images
        anno_dir (str): The path of the folder containing original annotation files

    Returns: 
        summary_table (pandas df): A dataframe summary table of the number of objects in each class
        unknown_object_name (array): An array of the labels ascribes to objects that are not counted in the other existing categories 
        number_of_images (int): the number of images in the summary table
    """
    # Initial variables to count the number of objects in each category (set to zero)
    all_objects_count = 0  # all objects
    closed_roof_tank_count = 0  # closed_roof_tank
    narrow_closed_roof_tank_count = 0  # narrow_closed_roof_tank
    external_floating_roof_tank_count = 0  # external_floating_roof_tank
    spherical_tank_count = 0  # spherical_tank
    sedimentation_tank_count = 0  # water_treatment_tank
    water_tower_count = 0  # water_tower
    undefined_object_count = 0  # undefined_object

    # Create an list to save unknown object names
    unknown_object_name = []
    # "enumerate each image" This chunk is actually just getting the paths for the images and annotations
    anno_list = glob(anno_dir + '/*.xml')
    for anno_path in anno_list:
        # read .xml file
        dom_tree = xml.dom.minidom.parse(anno_path)
        annotation = dom_tree.documentElement
        file_name_list = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
        file_name = file_name_list[0].childNodes[0].data
        object_list = annotation.getElementsByTagName('object')

        for objects in object_list:
            # print objects
            all_objects_count += 1
            namelist = objects.getElementsByTagName('name')
            object_name = namelist[0].childNodes[0].data
            if object_name == "closed_roof_tank":
                closed_roof_tank_count += 1
            elif object_name == "narrow_closed_roof_tank":
                narrow_closed_roof_tank_count += 1
            elif object_name == "external_floating_roof_tank":
                external_floating_roof_tank_count += 1
            elif object_name == "spherical_tank":
                spherical_tank_count += 1
            elif object_name == "sedimentation_tank":
                sedimentation_tank_count += 1
            elif object_name == "water_tower":
                water_tower_count += 1
            elif object_name == "undefined_object":
                undefined_object_count += 1
            else:
                unknown_object_name.append(object_name)

    summary_table = pd.DataFrame(
        {"categories": ["all_objects_count", "closed_roof_tank_count", "narrow_closed_roof_tank_count",
                        "external_floating_roof_tank_count", "spherical_tank_count", "sedimentation_tank_count",
                        "water_tower_count", "undefined_object"],
         "values": [all_objects_count, closed_roof_tank_count, narrow_closed_roof_tank_count,
                    external_floating_roof_tank_count,
                    spherical_tank_count, sedimentation_tank_count, water_tower_count, undefined_object_count]})
    summary_table.set_index('categories', inplace=True)
    summary_table.to_csv(os.path.join(output_dir, 'summary_table.csv'))
    
    unknown_object_name = np.unique(unknown_object_name)
    print("Array unknown objects", unknown_object_name)
    
    # calculate the number of images
    img_list = glob(img_dir + '/*.jpg')  # os.listdir(img_path)
    number_of_images = len(img_list)
    print("The number of clipped images included in the assessment", number_of_images)
