"""
Functions to process, format, and conduct calculations on the annotated or verified dataset
"""
# Standard packages
#from __future__ import print_function
import tempfile
import shutil
import os
import math
from glob import glob
import json
#Parsing/Modifying XML
import xml.dom.minidom
from xml.dom.minidom import parseString
import xml.etree.ElementTree as et
from xml.dom import minidom

#install standard
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import matplotlib 
import matplotlib.pyplot as plt

#process geospatial images
import fiona #must be import before geopandas
import geopandas as gpd
import rasterio
import rioxarray
import rtree
import pyproj
import shapely
from shapely.ops import transform
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, MultiLineString

from skimage.metrics import structural_similarity as compare_ssim

#import data_eng.az_proc as ap
def read_shp_from_zip(data_path): #used
    """ Unzips and reads zipped shapefle data 
    Args:
        data_path(str): zipped shapefile data
    Returns:
        data(geopandas dataframe): unzipped shapefile
    """
    temp_dir = tempfile.mkdtemp()
    shutil.unpack_archive(data_path, temp_dir)
    data = gpd.read_file(glob(temp_dir + "/*.shp")[0])
    shutil.rmtree(temp_dir)
    return(data)

## Write files
def write_list(list_, file_path): #used
    print("Started writing list data into a json file")
    with open(file_path, "w") as fp:
        json.dump(list_, fp)
        print("Done writing JSON data into .json file")

# Read list to memory
def read_list(file_path): #used
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        list_ = json.load(fp)
        return list_
    

############################################################################################################
###########################################   Remove Thumbs    #############################################
############################################################################################################
def remove_thumbs(path_to_folder_containing_images): #used
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_images(str): path to folder containing images
    Returns:
    None
    """
    if len(glob(path_to_folder_containing_images + "/*.db", recursive = True)) > 0:
        os.remove(glob(path_to_folder_containing_images + "/*.db", recursive = True)[0])
    

def add_formatted_and_standard_tile_names_to_tile_names_time_urls(tile_names_tile_urls): #used
    """Extract information from tile_names_tile urls numpy arrays
    """
    #get a list of the formated tile names
    tile_names = []
    for tile_url in tile_names_tile_urls:
        tile_url = tile_url[1].rsplit("/",3)
        #get the quad standard tile name 
        tile_name = tile_url[3]
        tile_name = os.path.splitext(tile_name)[0] 
        #format tile_names to only include inital capture date 
        if tile_name.count("_") > 5:
            tile_name = tile_name.rsplit("_",1)[0]
        #get the tile name formated (1/14/2022)
        tile_name_formatted = tile_url[1] + "_" + tile_url[2] + "_" + tile_url[3]
        tile_name_formatted = os.path.splitext(tile_name_formatted)[0] 
        tile_names.append([tile_name, tile_name_formatted])
    #create array that contains the formated tile_names and tile_names
    tile_names_tile_urls_formatted_tile_names = np.hstack((tile_names_tile_urls, np.array(tile_names)))
    
    return(tile_names_tile_urls_formatted_tile_names)


def unique_formatted_standard_tile_names(tile_names_tile_urls_complete_array): #keep for convinence
    """ identify  unique formated and standard tile names
    """
    unique_tile_name_formatted, indicies = np.unique(tile_names_tile_urls_complete_array[:,3], return_index = True)
    tile_names_tile_urls_complete_array_unique_formatted_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique formatted tile names", tile_names_tile_urls_complete_array_unique_formatted_tile_names.shape) 

    unique_tile_name_standard, indicies = np.unique(tile_names_tile_urls_complete_array[:,2], return_index = True)
    tile_names_tile_urls_complete_array_unique_standard_tile_names = tile_names_tile_urls_complete_array[indicies,:]
    print("unique standard tile names", tile_names_tile_urls_complete_array_unique_standard_tile_names.shape) 
    
    return(tile_names_tile_urls_complete_array_unique_standard_tile_names, tile_names_tile_urls_complete_array_unique_formatted_tile_names)


def move_tiles_of_verified_images_to_complete_dataset(tile_img_annotation, tiles_complete_dataset_path, path_to_verified_sets): #used
    """Move already downloaded tiles to completed dataset
    """
    #obtain the paths of tifs in the verified sets
    path_to_tifs_in_verified_sets = glob(path_to_verified_sets + "/**/*.tif", recursive = True)
    print("Number of tifs to be moved", len(path_to_tifs_in_verified_sets))

    #move verified tifs 
    for path in path_to_tifs_in_verified_sets:
        base = os.path.basename(path)
        tif = os.path.splitext(base)[0] #name of tif with the extension removed
        if tif in tile_img_annotation[:,0]:
            shutil.move(path, os.path.join(tiles_complete_dataset_path,base)) # copy images with matching .xml files in the "chips_tank" folder
            
def tiles_in_complete_dataset(tiles_complete_dataset_path): #used
    #Make a list of the tiles in the completed dataset
    os.makedirs(tiles_complete_dataset_path, exist_ok=True)
    
    tiles_downloaded = os.listdir(tiles_complete_dataset_path)
    tiles_downloaded_with_ext_list = []
    tiles_downloaded_without_ext_list = []
    
    for tile in tiles_downloaded:
        tiles_downloaded_with_ext_list.append(tile)
        tiles_downloaded_without_ext_list.append(os.path.splitext(tile)[0]) #name of tif with the extension removed
    return(np.array(tiles_downloaded_with_ext_list), np.array(tiles_downloaded_without_ext_list))

def jpg_paths_to_tiles_without_ext(jpg_paths): #used
    """
    Determine which tiles corresponding to jpg that have been annotated #jpg_tiles
    Get a numpy array of the unique standard tile names corresponding to a list of jpg paths
    Args:
    jpgs_paths(list): list of jpg paths
    Returns:
    tiles(numpy): 
    """
    tiles = []
    for path in jpg_paths:
        base = os.path.basename(path)
        img = os.path.splitext(base)[0] #name of tif with the extension removed
        tile = img.rsplit("_",1)[0]
        tile = tile.split("_",4)[4] #get the tile names to remove duplicates from being downloaded
        tiles.append(tile)
    return(np.unique(tiles))
##############################################################################################################################
###################################                  Chip Tiles              #################################################
##############################################################################################################################
def tile_to_chip_array(tile, x, y, item_dim): #used
    """
    https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    ##
    x: col index
    y: row index
    """
    dimensions = tile.shape[2]
    chip_img = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]
    #add in back space if it is the edge of an image
    if (chip_img.shape[0] != item_dim) & (chip_img.shape[1] != item_dim): #width
        #print("Incorrect Width")
        chip = np.zeros((item_dim,item_dim,dimensions), np.uint8)
        chip[0:chip_img.shape[0], 0:chip_img.shape[1]] = chip_img
        chip_img = chip
    if chip_img.shape[0] != item_dim:  #Height
        black_height = item_dim  - chip_img.shape[0] #Height
        black_width = item_dim #- chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width,  dimensions), np.uint8)
        chip_img = np.concatenate([chip_img, black_img])
    if chip_img.shape[1] != item_dim: #width
        black_height = item_dim #- chip_img.shape[0] #Height
        black_width = item_dim - chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width, dimensions), np.uint8)
        chip_img = np.concatenate([chip_img, black_img],1)
    return(chip_img)


############## Download Tiles ##########################################################################################
def download_tiles_of_verified_images(positive_images_complete_dataset_path, tiles_complete_dataset_path, 
                                      tiles_downloaded, tile_names_tile_urls_complete_array): #used
    """
    # Download remaining tiles that correspond to ONLY to verified images
    #Gather the locations of tiles that have already been downlaoded and verified 
    """
    # Make a list of the tiles moved to completed dataset
    tiles_downloaded_with_ext, tiles_downloaded = tiles_in_complete_dataset(tiles_complete_dataset_path)
    
    positive_jpg_paths = glob(positive_images_complete_dataset_path + "/*.jpg", recursive = True)
    print("number of positive and verified images:", len(positive_jpg_paths))
    
    #  Determine which tiles corresponding to jpg that have been annotated #jpg_tiles
    positive_jpg_tiles = jpg_paths_to_tiles_without_ext(positive_jpg_paths)
    print("the number of tiles corresponding to verified images:", len(positive_jpg_tiles))

    # Identify tiles that have not already been downloaded
    tiles_to_download = []
    for tile in positive_jpg_tiles: #index over the downloaded tiled
        if tile not in tiles_downloaded: #index over the tiles that should be downloded
            tiles_to_download.append(tile)
    print("the number of tiles that need to be downloaded:", len(tiles_to_download))
    
    # Download Tiles  
    tile_names = []
    tile_urls = []
    file_names = []
    tile_names_without_year = []
    for tile in tiles_to_download:   
        ### download the tiles if they are not in the tiles folder 
        #check if the tile name is contained in the string of complete arrays
        tile_name = [string for string in tile_names_tile_urls_complete_array[:,0] if tile in string]          
        if len(tile_name) == 1: #A single tile name # get tile url from the first (only) entry
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1] 
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif len(np.unique(tile_name)) > 1: # Multiple (different tiles) possibly the same tiles in different states, possible different years
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[0]][0][1]# get tile url
            tile_names.append(tile_name[0])
            tile_urls.append(tile_url)
        elif (len(tile_name) > 1): #Multiple different tile names that are the same, probably different naip storage locations
            # get tile url from the second entry 
            tile_url = tile_names_tile_urls_complete_array[tile_names_tile_urls_complete_array[:,0]==tile_name[1]][1][1] 
            tile_names.append(tile_name[1])
            tile_urls.append(tile_url)
            
        #get file name
        file_name = tile_name[0]
        if tile_name[0].count("_") > 5:
            tile_name = tile_name[0].rsplit("_",1)[0]
            file_name = tile_name + ".tif"
        print(file_name)
        ### Download tile
        file_names.append(ap.download_url(tile_url, tiles_complete_dataset_path,
                                                     destination_filename = file_name,       
                                                             progress_updater=ap.DownloadProgressBar()))
    #get the tile_names without the year
    for file_name in file_names:
        tile_names_without_year.append(file_name.rsplit("_",1)[0])
        
    return(np.array((tile_names, tile_urls, file_names, tile_names_without_year)).T)


###################################################################################################################
#################################   Obtain Location Data (UTM to WGS84)  ##########################################
###################################################################################################################   
def tile_dimensions_and_utm_coords(tile_path): #used
    """ Obtain tile band, height and width and utm coordinates
    Args: tile_path(str): the path of the tile 
    Returns: 
    utmx(np array): the x utm coordinates corresponding with the tile coordinate convention (origin in upper left hand corner)
    utmy(np array): the y utm coordinates corresponding with the tile coordinate convention (origin in upper left hand corner)
    tile_band(int): the number of bands
    tile_height(int): the height of the tile (in pixels)
    tile_width(int): the width of the tile (in pixels)
    """
    ## Get tile locations
    da = rioxarray.open_rasterio(tile_path) ## Read the data
    # Compute the lon/lat coordinates with rasterio.warp.transform
    # lons, lats = np.meshgrid(da['x'], da['y'])
    tile_band, tile_height, tile_width = da.shape[0], da.shape[1], da.shape[2]
    utmx = np.array(da['x'])
    utmy = np.array(da['y'])
    return(utmx, utmy, tile_band, tile_height, tile_width)
def get_utm_proj(tile_path): #used
    """ Obtain utm projection as a string 
    Args: tile_path(str): the path of the tile 
    Returns: 
    utm_proj(str): the UTM string as the in term of EPSG code
    """
    da = rasterio.open(tile_path)
    utm_proj = da.crs.to_string()
    return(utm_proj)
def transform_point_utm_to_wgs84(utm_proj, utm_xcoord, utm_ycoord): #used
    """ Convert a utm pair into a lat lon pair 
    Args: 
    utm_proj(str): the UTM string as the in term of EPSG code
    utmx(int): the x utm coordinate of a point
    utmy(int): the y utm coordinates of a point
    Returns: 
    (wgs84_pt.x, wgs84_pt.y): the 'EPSG:4326' x and y coordinates 
    """
    #https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    #get utm projection
    utm = pyproj.CRS(utm_proj)
    #get wgs84 proj
    wgs84 = pyproj.CRS('EPSG:4326')
    #specify utm point
    utm_pt = Point(utm_xcoord, utm_ycoord)
    #transform utm into wgs84 point
    project = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    wgs84_pt = transform(project, utm_pt)
    return wgs84_pt.x, wgs84_pt.y


#################################################################################################################
######################## correct_incon_labels.py Correct object names in xmls     ###############################
#################################################################################################################


def correct_inconsistent_labels_xml(xml_path, corrected_xml_path): #used
    """ Correct incorrect or inconsistent labels
    """
    # Create a list of the possible names that each category may take
    correctly_formatted_object = ["closed_roof_tank", "narrow_closed_roof_tank",
                                  "external_floating_roof_tank", "sedimentation_tank",
                                  "water_tower", "undefined_object", "spherical_tank"]
    object_dict = {"closed_roof_tank": "closed_roof_tank", "closed_roof_tank ": "closed_roof_tank",
                   "closed roof tank": "closed_roof_tank",
                   "narrow_closed_roof_tank": "narrow_closed_roof_tank",
                   "external_floating_roof_tank": "external_floating_roof_tank",
                   "external floating roof tank": "external_floating_roof_tank",
                   'external_floating_roof_tank ': "external_floating_roof_tank",
                   'external_closed_roof_tank': "external_floating_roof_tank",
                   "water_treatment_tank": "sedimentation_tank", 'water_treatment_tank ': "sedimentation_tank",
                   "water_treatment_plant": "sedimentation_tank", "water_treatment_facility": "sedimentation_tank",
                   "water_tower": "water_tower", "water_tower ": "water_tower", 'water_towe': "water_tower",
                   "spherical_tank": "spherical_tank", 'sphere': "spherical_tank", 'spherical tank': "spherical_tank",
                   "undefined_object": "undefined_object", "unknown_object": "undefined_object",
                   "silo": "undefined_object"}

    # "enumerate each image" This chunk is actually just getting the paths for the images and annotations
    # use the parse() function to load and parse an XML file
    tree = et.parse(xml_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        for name in obj.findall('name'):
            if name.text not in correctly_formatted_object:
                name.text = object_dict[name.text]
        if int(obj.find('difficult').text) == 1:
            obj.find('truncated').text = '1'
            obj.find('difficult').text = '1'
        if int(obj.find('truncated').text) == 1:
            obj.find('truncated').text = '1'
            obj.find('difficult').text = '1'
    tree.write(corrected_xml_path)


def reformat_xml_for_compiled_dataset(xml_path): #used
    """ reformat xml files for rechipped images to include resolution, year, updated filename, and updated path.
    Args:
    xml_path(str): path to xml file

    https://docs.python.org/3/library/xml.etree.elementtree.html
    https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
    https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
    """
    # load xml
    chip_name = os.path.splitext(os.path.basename(xml_path))[0]
    tree = et.parse(os.path.join(xml_path))
    root = tree.getroot()
    # add resolution to xml
    resolution = et.Element("resolution")
    resolution.text = chip_name.split("_")[4]  # resolution
    et.indent(tree, space="\t", level=0)
    root.insert(3, resolution)
    # add year to xml
    capture_date = et.Element("capture_date")
    capture_date.text = chip_name.split("_")[5]  # year.month.day?
    et.indent(tree, space="\t", level=0)
    root.insert(4, capture_date)
    # correct spacing for source (dataset name)
    et.indent(tree, space="\t", level=0)
    # correct filename and path to formatting with row/col coordinates
    for filename in root.iter('filename'):
        filename.text = chip_name
    for path in root.iter('path'):
        path.text = xml_path
    tree.write(xml_path)


def reclassify_narrow_closed_roof_and_closed_roof_tanks(xml_path): #used
    """ Reclassify Narrow Closed Roof and Closed Roof Tanks
    Args:
    xml_path(str): path to xml file
    """
    # load each xml
    class_ob = []
    tree = et.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter('object'):
        name = obj.find("name").text
        xmlbox = obj.find('bndbox')  # get the bboxes
        obj_xmin = xmlbox.find('xmin').text
        obj_ymin = xmlbox.find('ymin').text
        obj_xmax = xmlbox.find('xmax').text
        obj_ymax = xmlbox.find('ymax').text
        width = int(obj_xmax) - int(obj_xmin)
        height = int(obj_ymax) - int(obj_ymin)
        if (int(obj.find('difficult').text) == 0) and (int(obj.find('truncated').text) == 0):
            # if a closed roof tank is less than or equal to the narrow closed roof tank threshold
            # than reclassify as  narrow closed roof tank
            if name == "closed_roof_tank" and (width <= 15 or height <= 15):
                name = "narrow_closed_roof_tank"
            # if a narrow closed roof tank is greater than the closed roof tank threshold
            # than reclassify as closed roof tank
            if name == "narrow_closed_roof_tank" and (width > 15 or height > 15):
                name = "closed_roof_tank"

    tree.write(os.path.join(xml_path))
    
###################################################################################################################
##########################   Create dataframe of Image and Tile Characteristics  ##################################
###################################################################################################################   

def image_tile_characteristics(images_and_xmls_by_tile_dir, tile_dir, xml_folder_name, item_dim=int(512)): #used
    """ vectorize function"""
    #tile characteristics
    tile_names_by_tile = []
    tile_heights = []
    tile_widths = []
    tile_depths = []
    nw_x_utm_tile = [] #NW_coordinates
    nw_y_utm_tile = []  #NW_coordinates
    se_x_utm_tile = [] #SE_coordinates
    se_y_utm_tile = [] #SE_coordinates
    utm_projection_tile = [] 
    nw_lon_tile = [] #NW_coordinates
    nw_lat_tile = [] #NW_coordinates
    se_lon_tile = [] #SE_coordinates
    se_lat_tile = [] #SE_coordinates
    #image characteristics
    image_names = []
    tile_names_by_chip = []
    row_indicies = []
    col_indicies = []
    minx_pixel = []
    miny_pixel = []
    maxx_pixel = []
    maxy_pixel = []
    utm_projection_chip = [] 
    nw_x_utm_chip = [] #NW_coordinates
    nw_y_utm_chip = []  #NW_coordinates
    se_x_utm_chip = [] #SE_coordinates
    se_y_utm_chip = [] #SE_coordinates
    nw_lon_chip = [] #NW_coordinates
    nw_lat_chip = [] #NW_coordinates
    se_lon_chip = [] #SE_coordinates
    se_lat_chip = [] #SE_coordinates

    tile_names = [f for f in os.listdir(images_and_xmls_by_tile_dir) \
                  if len(glob(os.path.join(images_and_xmls_by_tile_dir, f, "**/*.xml"))) > 0]
    for tile_name in tqdm(tile_names):
        tile_names_by_tile.append(tile_name)
        #specify image/xml paths for each tile
        positive_image_dir = os.path.join(images_and_xmls_by_tile_dir, tile_name, "chips_positive")
        remove_thumbs(positive_image_dir)
        positive_xml_dir = os.path.join(images_and_xmls_by_tile_dir, tile_name, xml_folder_name)
        #load a list of images/xmls for each tile
        positive_images = os.listdir(positive_image_dir)
        positive_xmls = os.listdir(positive_xml_dir)
        #read in tile
        tile_path = os.path.join(tile_dir, tile_name + ".tif")
        utmx, utmy, tile_band, tile_height, tile_width = tile_dimensions_and_utm_coords(tile_path)
        #specify the utm coords for each tile 
        nw_x_utm_tile.append(utmx[0]) #NW_coordinates #min
        nw_y_utm_tile.append(utmy[0])  #NW_coordinates #max
        se_x_utm_tile.append(utmx[-1]) #SE_coordinates #max
        se_y_utm_tile.append(utmy[-1]) #SE_coordinates #min
        #specify utm proj
        utm_proj = get_utm_proj(tile_path)
        utm_projection_tile.append(utm_proj)
        #specify tile chracteristics
        tile_heights.append(tile_height)
        tile_widths.append(tile_width)
        tile_depths.append(tile_band)
        #specify lat lon 
        nw_lon, nw_lat = transform_point_utm_to_wgs84(utm_proj, utmx[0], utmy[0])
        se_lon, se_lat = transform_point_utm_to_wgs84(utm_proj, utmx[-1], utmy[-1]) 
        nw_lon_tile.append(nw_lon) #NW_coordinates #minlon
        nw_lat_tile.append(nw_lat) #NW_coordinates #maxlat
        se_lon_tile.append(se_lon) #SE_coordinates #maxlon
        se_lat_tile.append(se_lat) #SE_coordinates #minlat

        for positive_image in positive_images: #iterate over each image affiliated with a given tile
            #tile and chip names
            image_name = os.path.splitext(positive_image)[0]
            image_names.append(image_name) # The index is a six-digit number like '000023'.
            tile_names_by_chip.append(tile_name)
            #row/col indicies 
            y, x = image_name.split("_")[-2:] #name of tif with the extension removed; y=row;x=col
            y = int(y)
            x = int(x)
            row_indicies.append(y)
            col_indicies.append(x)
            #get the pixel coordinates (indexing starting at 0)
            minx = x*item_dim 
            miny = y*item_dim 
            maxx = (x+1)*item_dim - 1
            maxy = (y+1)*item_dim - 1
            if maxx > tile_width:
                maxx = tile_width - 1
            if maxy > tile_height:
                maxy = tile_height - 1
            minx_pixel.append(minx) #NW (max: Top Left) # used for numpy crop
            miny_pixel.append(miny) #NW (max: Top Left) # used for numpy crop
            maxx_pixel.append(maxx) #SE (min: Bottom right) 
            maxy_pixel.append(maxy) #SE (min: Bottom right) 
            #image utm coordinates
            nw_x_utm_chip.append(utmx[minx]) #NW_coordinates
            nw_y_utm_chip.append(utmy[miny]) #NW_coordinates
            se_x_utm_chip.append(utmx[maxx]) #SE_coordinates
            se_y_utm_chip.append(utmy[maxy]) #SE_coordinates
            utm_projection_chip.append(utm_proj)
            #determine the lat/lon
            nw_lon, nw_lat = transform_point_utm_to_wgs84(utm_proj, utmx[minx], utmy[miny])
            se_lon, se_lat = transform_point_utm_to_wgs84(utm_proj, utmx[maxx], utmy[maxy]) 
            nw_lon_chip.append(nw_lon) #NW (max: Top Left) # used for numpy crop
            nw_lat_chip.append(nw_lat) #NW (max: Top Left) # used for numpy crop
            se_lon_chip.append(se_lon) #SE (min: Bottom right) 
            se_lat_chip.append(se_lat) #SE (min: Bottom right)

    tile_characteristics = pd.DataFrame(data={'tile_name': tile_names_by_tile, 'tile_height': tile_heights, 
                                              'tile_width': tile_widths, 'tile_bands': tile_depths,
                                              'utm_projection': utm_projection_tile, 
                                              "nw_x_utm_tile_coord":nw_x_utm_tile, "nw_y_utm_tile_coord":nw_y_utm_tile, 
                                              "se_x_utm_tile_coord":se_x_utm_tile, "se_y_utm_tile_coord":se_y_utm_tile,
                                              "nw_lat_tile_coord":nw_lat_tile, "nw_lon_tile_coord":nw_lon_tile,
                                              "se_lat_tile_coord":se_lat_tile, "se_lon_tile_coord":se_lon_tile})
    image_characteristics = pd.DataFrame(data={'image_name': image_names, 'tile_name': tile_names_by_chip, 
                                               'row_index': row_indicies, 'col_index': col_indicies,
                                               'nw_x_pixel_image_coord': minx_pixel, 'nw_y_pixel_image_coord': miny_pixel, 
                                               'se_x_pixel_image_coord': maxx_pixel,'se_y_pixel_image_coord': maxy_pixel, 
                                               'utm_projection': utm_projection_chip, 
                                               'nw_x_utm_image_coord':nw_x_utm_chip, 'nw_y_utm_image_coord':nw_y_utm_chip,
                                               'se_x_utm_image_coord':se_x_utm_chip, 'se_y_utm_image_coord':se_y_utm_chip,
                                               'nw_lat_image_coord':nw_lat_chip, 'nw_lon_image_coord':nw_lon_chip,
                                               'se_lat_image_coord':se_lat_chip, 'se_lon_image_coord':se_lon_chip})
    return tile_characteristics, image_characteristics

###############################################################################################
###### tile_level_annotations.py: Create Tile level  XMLs; Merge tile level annotations; ######
###############  add additional information Write Tile Level Annotations; #####################
###############################################################################################


def create_tile_xml(tile_name, xml_directory, tile_resolution, tile_capture_date,
                    tile_width, tile_height, tile_band): #used
    tile_name_ext = tile_name + ".tif"
    root = et.Element("annotation")
    folder = et.Element("folder") #add folder to xml
    folder.text = "tiles" #folder
    root.insert(0, folder)
    filename = et.Element("filename") #add filename to xml
    filename.text = tile_name_ext #filename
    root.insert(1, filename)
    path = et.Element("path") #add path to xml
    path.text = os.path.join(xml_directory, tile_name_ext) #path
    root.insert(2, path)
    resolution = et.Element("resolution") #add resolution to xml
    resolution.text = tile_resolution #resolution
    root.insert(3, resolution)
    capture_date = et.Element("capture_date") #add capture_date to xml
    capture_date.text = tile_capture_date #capture_date
    root.insert(4, capture_date)
    source = et.Element("source") #add database to xml
    database = et.Element("database")
    database.text = "Tile Level Annotation"
    source.insert(0, database)
    root.insert(5, source)
    size = et.Element("size") #add size to xml
    width = et.Element("width")
    width.text = str(tile_width) #width
    size.insert(0, width)
    height = et.Element("height")
    height.text = str(tile_height) #height
    size.insert(1, height)
    depth = et.Element("depth")
    depth.text = str(tile_band) #depth
    size.insert(2, depth)
    root.insert(6, size)
    tree = et.ElementTree(root)
    et.indent(tree, space="\t", level=0)
    #tree.write("filename.xml")
    tree.write(os.path.join(xml_directory, tile_name +".xml"))     


def add_objects(xml_directory, tile_name, obj_class, obj_truncated, obj_difficult, obj_chip_name,
                obj_xmin, obj_ymin, obj_xmax, obj_ymax): #used
    tree = et.parse(os.path.join(xml_directory, tile_name + ".xml"))
    root = tree.getroot() 
    obj = et.Element("object") #add size to xml
    
    name = et.Element("name") #class
    name.text = str(obj_class) 
    obj.insert(0, name)
    
    pose = et.Element("pose") #pose
    pose.text = "Unspecified" 
    obj.insert(1, pose)
    
    truncated = et.Element("truncated")
    truncated.text = str(obj_truncated)
    obj.insert(2, truncated)

    difficult = et.Element("difficult")
    difficult.text = str(obj_difficult)
    obj.insert(3, difficult)
 
    chip_name = et.Element("chip_name")
    chip_name.text = str(obj_chip_name)
    obj.insert(4, chip_name)

    bndbox = et.Element("bndbox") #bounding box
    xmin = et.Element("xmin") #xmin
    xmin.text = str(obj_xmin) 
    bndbox.insert(0, xmin)
    ymin = et.Element("ymin") #ymin
    ymin.text = str(obj_ymin) 
    bndbox.insert(1, ymin)
    xmax = et.Element("xmax") #xmax
    xmax.text = str(obj_xmax) 
    bndbox.insert(2, xmax)
    ymax = et.Element("ymax") #ymax
    ymax.text = str(obj_ymax) 
    bndbox.insert(3, ymax)
    obj.insert(5, bndbox)
    
    root.append(obj)
    tree = et.ElementTree(root)
    et.indent(tree, space="\t", level=0)
    tree.write(os.path.join(xml_directory, tile_name +".xml"))   


def generate_tile_xmls(images_and_xmls_by_tile_dir, tile_dir, xml_folder_name, tiles_xml_dir, item_dim): #used
    """
    Args:
        images_and_xmls_by_tile_dir(str): path to directory containing folders holding annotations and images
        tile_dir(str): path to directory containing tiles
        
    Returns:
        object:
    """
    tile_names = [f for f in os.listdir(images_and_xmls_by_tile_dir) \
                  if len(glob(os.path.join(images_and_xmls_by_tile_dir, f, "**/*.xml"))) > 0]
    for tile_name in tqdm(tile_names):
        tile_name_ext = tile_name + ".tif"
        #get tile dimensions ##replace with information from tile characteristics
        da = rioxarray.open_rasterio(os.path.join(tile_dir, tile_name_ext))
        tile_band, tile_height, tile_width = da.shape[0], da.shape[1], da.shape[2]
        #specify xml paths for each tile
        positive_xml_dir = os.path.join(images_and_xmls_by_tile_dir, tile_name, xml_folder_name)
        #load a list of images/xmls for each tile
        positive_xmls = os.listdir(positive_xml_dir)
                       
        for index, chip_xml in enumerate(positive_xmls): #iterate over positive xmls
            #load each chipped image xml
            tree = et.parse(os.path.join(positive_xml_dir, chip_xml))
            root = tree.getroot()

            #create the tile xml
            if index == 0:
                resolution = root.find('resolution').text
                capture_date = root.find('capture_date').text
                create_tile_xml(tile_name, tiles_xml_dir, resolution, capture_date, tile_width, tile_height, tile_band)

            #identify rows and columns
            chip_name = os.path.splitext(chip_xml)[0]
            y, x = chip_name.split("_")[-2:] #name of tif with the extension removed; y=row;x=col
            # Each chip xml goes from 1 - 512, specify the "0", or the end point of the last xml
            minx = int(x)*item_dim
            miny = int(y)*item_dim

            #add the bounding boxes
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                obj_xmin = str(minx + int(xmlbox.find('xmin').text))
                obj_xmax = str(minx + int(xmlbox.find('xmax').text))
                obj_ymin = str(miny + int(xmlbox.find('ymin').text))
                obj_ymax = str(miny + int(xmlbox.find('ymax').text))
                # correct bboxes that extend past the bounds of the tile width/height
                if int(obj_xmin) > tile_width:
                    obj_xmin = str(tile_width)
                if int(obj_xmax) > tile_width:
                    obj_xmax = str(tile_width)
                if int(obj_ymin) > tile_height:
                    obj_ymin = str(tile_height)
                if int(obj_ymax) > tile_height:
                    obj_ymax = str(tile_height)
                add_objects(tiles_xml_dir, tile_name, obj.find('name').text, obj.find('truncated').text,
                            obj.find('difficult').text, chip_name, obj_xmin, obj_ymin, obj_xmax, obj_ymax)

                
def merge_boxes(bbox1, bbox2): #used
    """ 
    Generate a bounding box that covers two bounding boxes
    Called in merge_algo
    Arg:
    bbox1(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 1 
    bbox2(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 2
    Returns:
    merged_bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for the merged bbox

    """
    return [min(bbox1[0], bbox2[0]), 
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])]


def calc_sim(bbox1, bbox2, dist_limit): #used
    """Determine the similarity of distances between bboxes to determine whether bboxes should be merged
    Computer a Matrix similarity of distances of the text and object
    Called in merge_algo
    Arg:
    bbox1(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 1 
    bbox2(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 2
    dist_list(int): the maximum threshold (pixel distance) to merge bounding boxes
    Returns:
    (bool): to indicate whether the bboxes should be merged 
    """

    # text: ymin, xmin, ymax, xmax
    # obj: ymin, xmin, ymax, xmax
    bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax = bbox1
    bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax = bbox2
    x_dist = min(abs(bbox2_xmin-bbox1_xmax), abs(bbox2_xmax-bbox1_xmin))
    y_dist = min(abs(bbox2_ymin-bbox1_ymax), abs(bbox2_ymax-bbox1_ymin))
        
    #define distance if one object is inside the other
    if (bbox2_xmin <= bbox1_xmin) and (bbox2_ymin <= bbox1_ymin) and (bbox2_xmax >= bbox1_xmax) and (bbox2_ymax >= bbox1_ymax):
        return True
    elif (bbox1_xmin <= bbox2_xmin) and (bbox1_ymin <= bbox2_ymin) and (bbox1_xmax >= bbox2_xmax) and (bbox1_ymax >= bbox2_ymax):
        return True
    #determine if both bboxes are close to each other in 1d, and equal or smaller length in the other
    elif (x_dist <= dist_limit) and (bbox1_ymin <= bbox2_ymin) and (bbox1_ymax >= bbox2_ymax): #bb1 bigger
        return True
    elif (x_dist <= dist_limit) and (bbox2_ymin <= bbox1_ymin) and (bbox2_ymax >= bbox1_ymax): #bb2 bigger
        return True
    elif (y_dist <= dist_limit) and (bbox1_xmin <= bbox2_xmin) and (bbox1_xmax >= bbox2_xmax): #bb1 bigger
        return True
    elif (y_dist <= dist_limit) and (bbox2_xmin <= bbox1_xmin) and (bbox2_xmax >= bbox1_xmax): #bb2 bigger
        return True
    else: 
        return False


def merge_algo(characteristics, bboxes, dist_limit): #used
    merge_bools = [False] * len(characteristics)
    for i, (char1, bbox1) in enumerate(zip(characteristics, bboxes)):
        for j, (char2, bbox2) in enumerate(zip(characteristics, bboxes)):
            if j <= i:
                continue
            # Create a new box if a distances is less than disctance limit defined 
            merge_bool = calc_sim(bbox1, bbox2, dist_limit) 
            if merge_bool == True:
            # Create a new box  
                new_box = merge_boxes(bbox1, bbox2)   
                bboxes[i] = new_box
                #delete previous text boxes
                del bboxes[j]
                
                # Create a new text string
                ##chip_name list
                if char1[0] != char2[0]: #if the chip_names are not the same
                    #make chip_names into an array
                    if type(char1[0]) == str: 
                        chip_names_1 = np.array([char1[0]])
                    if type(char2[0]) == str:
                        chip_names_2 = np.array([char2[0]])
                    chip_names = np.concatenate((chip_names_1, chip_names_2), axis=0)
                    chip_names = np.unique(chip_names).tolist()
                else:
                    chip_names = np.unique(char1[0]).tolist()  #if the chip_names are not the same
                
                #get object type 
                if char1[1] != char2[1]:
                    object_type = 'undefined_object'
                object_type = char1[1]
                
                characteristics[i] = [chip_names, object_type, 'Unspecified', '1', '1']
                #delete previous text 
                del characteristics[j]
                
                #return a new boxes and new text string that are close
                merge_bools[i] = True
    return merge_bools, characteristics, bboxes


def calculate_diameter(bbox, resolution = 0.6): #used
    """ Calculate the diameter of a given bounding bbox (in Pascal Voc Format) for imagery of a given resolution
    Arg:
    bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box. Utm coordinates are provided as [nw_x_utm, se_y_utm, se_x_utm, nw_y_utm] to conform with Pascal Voc Format.
    resolution(float): the (gsd) resolution of the imagery
    Returns:
    (diameter): the diameter of the bbox of interest
    
                [#minx, #maxy, #maxx, #minx]
    """
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = bbox
    obj_width = obj_xmax - obj_xmin
    obj_height = obj_ymax - obj_ymin
    diameter = min(obj_width, obj_height) * resolution #meter
    return diameter


def merge_tile_annotations(tile_characteristics, tiles_xml_dir, tiles_xml_list=None, distance_limit=5): #used
    # References:
    # https: // answers.opencv.org / question / 231263 / merging - nearby - rectanglesedited /
    # https://stackoverflow.com/questions/55593506/merge-the-bounding-boxes-near-by-into-one
    # specify tiles_xml_list
    if tiles_xml_list is None: # if tiles_xml_list not provided, specify the tiles xml list
        tiles_xml_list = os.listdir(tiles_xml_dir)
    # lists for geosons/geodatabase
    tile_names = []
    chip_names = []
    object_class = []
    merged_bbox = []
    geometry = []  
    minx_pixels = []
    miny_pixels = []
    maxx_pixels = []
    maxy_pixels = []
    utm_projection = []
    nw_x_utms = []
    nw_y_utms = []
    se_x_utms = []
    se_y_utms = []
    centroid_lons = []
    centroid_lats = []
    nw_lats = []
    nw_lons = []
    se_lats = []
    se_lons = []
    diameter = []
    for tile_xml in tqdm(tiles_xml_list): # iterate over tiles
        # save bboxes and characteristics
        trunc_diff_objs_bboxes = []
        trunc_diff_objs_characteristics = []
        remaining_objs_bboxes = []
        remaining_objs_characteristics = []
        # get tile name / tile xml path
        tile_name = os.path.splitext(tile_xml)[0]
        tile_xml_path = os.path.join(tiles_xml_dir, tile_xml)
        # load tile characteristics
        tile_characteristics_subset = tile_characteristics[tile_characteristics.loc[:,"tile_name"] == tile_name]
        #get utm coords
        tile_utmx_array = np.linspace(float(tile_characteristics_subset["nw_x_utm_tile_coord"].values[0]), 
                                      float(tile_characteristics_subset["se_x_utm_tile_coord"].values[0]),
                                      int(tile_characteristics_subset["tile_width"].values[0]))
        tile_utmy_array = np.linspace(float(tile_characteristics_subset["nw_y_utm_tile_coord"].values[0]),
                                      float(tile_characteristics_subset["se_y_utm_tile_coord"].values[0]),
                                      int(tile_characteristics_subset["tile_height"].values[0]))
        utm_proj = tile_characteristics_subset["utm_projection"].values[0]
        # load each xml
        tree = et.parse(tile_xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox') #get the bboxes
            obj_xmin = xmlbox.find('xmin').text
            obj_ymin = xmlbox.find('ymin').text
            obj_xmax = xmlbox.find('xmax').text
            obj_ymax = xmlbox.find('ymax').text
            # get truncated bboxes/characteristics
            if (int(obj.find('difficult').text) == 1) or (int(obj.find('truncated').text) == 1):
                trunc_diff_objs_bboxes.append([obj_xmin, obj_ymin, obj_xmax, obj_ymax])
                trunc_diff_objs_characteristics.append([obj.find('chip_name').text, obj.find('name').text,
                                                        obj.find('pose').text, obj.find('truncated').text,
                                                        obj.find('difficult').text])
            else: #get remaining bboxes/characteristics
                remaining_objs_bboxes.append([obj_xmin, obj_ymin, obj_xmax, obj_ymax])
                remaining_objs_characteristics.append([obj.find('chip_name').text, obj.find('name').text,
                                                       obj.find('pose').text, obj.find('truncated').text,
                                                       obj.find('difficult').text])
        
        # Add merge bboxes
        trunc_diff_objs_bboxes = np.array(trunc_diff_objs_bboxes).astype(np.int32)
        trunc_diff_objs_bboxes = trunc_diff_objs_bboxes.tolist()
        merged_bools, merged_characteristics, merged_bboxes = merge_algo(trunc_diff_objs_characteristics,
                                                                         trunc_diff_objs_bboxes, distance_limit)

        for j, (merged_bool, char, bbox) in enumerate(zip(merged_bools, merged_characteristics, merged_bboxes)):
            tile_names.append(tile_name) 
            chip_names.append(char[0])
            object_class.append(char[1])
            # state whether bbox were merged
            merged_bbox.append(merged_bool)
            # pixel coordinates, 0 indexed
            minx = bbox[0] - 1
            miny = bbox[1] - 1
            maxx = bbox[2] - 1
            maxy = bbox[3] - 1 

            minx_pixels.append(minx)
            miny_pixels.append(miny)
            maxx_pixels.append(maxx)
            maxy_pixels.append(maxy)
            # geospatial data
            utm_projection.append(utm_proj)
            nw_x_utm = tile_utmx_array[minx]
            nw_y_utm = tile_utmy_array[miny]
            se_x_utm = tile_utmx_array[maxx]
            se_y_utm = tile_utmy_array[maxy]
            nw_x_utms.append(nw_x_utm)
            nw_y_utms.append(nw_y_utm)
            se_x_utms.append(se_x_utm)
            se_y_utms.append(se_y_utm)
            nw_lon, nw_lat = transform_point_utm_to_wgs84(utm_proj, nw_x_utm, nw_y_utm)
            se_lon, se_lat = transform_point_utm_to_wgs84(utm_proj, se_x_utm, se_y_utm)
            nw_lons.append(nw_lon)
            nw_lats.append(nw_lat)
            se_lons.append(se_lon)
            se_lats.append(se_lat)
            geometry.append(Polygon([(nw_lon, nw_lat), (nw_lon, se_lat), 
                                     (se_lon, se_lat), (se_lon, nw_lat)]))
            #add centroid
            utmcentroidx=nw_x_utm+(se_x_utm-nw_x_utm)/2
            utmcentroidy=se_y_utm+(nw_y_utm-se_y_utm)/2
            centroid_lon, centroid_lat = transform_point_utm_to_wgs84(utm_proj, utmcentroidx, utmcentroidy)
            centroid_lons.append(centroid_lon)
            centroid_lats.append(centroid_lat)
            #calculate diameter
            diameter.append(calculate_diameter(bbox))
            
        #Add remaining bboxes
        remaining_objs_bboxes = np.array(remaining_objs_bboxes).astype(np.int32)
        remaining_objs_bboxes = remaining_objs_bboxes.tolist()
        for j, (char, bbox) in enumerate(zip(remaining_objs_characteristics, remaining_objs_bboxes)):
            tile_names.append(tile_name)
            chip_names.append(char[0])
            object_class.append(char[1])
            # state whether bbox were merged
            merged_bbox.append(False)
            # pixel coordinates, 0 indexed
            minx = bbox[0] - 1
            miny = bbox[1] - 1
            maxx = bbox[2] - 1
            maxy = bbox[3] - 1
            minx_pixels.append(minx)
            miny_pixels.append(miny)
            maxx_pixels.append(maxx)
            maxy_pixels.append(maxy)
            #geospatial data
            utm_projection.append(utm_proj)
            nw_x_utm = tile_utmx_array[minx]
            nw_y_utm = tile_utmy_array[miny]
            se_x_utm = tile_utmx_array[maxx]
            se_y_utm = tile_utmy_array[maxy]
            nw_x_utms.append(nw_x_utm)
            nw_y_utms.append(nw_y_utm)
            se_x_utms.append(se_x_utm)
            se_y_utms.append(se_y_utm)
            nw_lon, nw_lat = transform_point_utm_to_wgs84(utm_proj, nw_x_utm, nw_y_utm)
            se_lon, se_lat = transform_point_utm_to_wgs84(utm_proj, se_x_utm, se_y_utm)
            nw_lons.append(nw_lon)
            nw_lats.append(nw_lat)
            se_lons.append(se_lon)
            se_lats.append(se_lat)
            geometry.append(Polygon([(nw_lon, nw_lat), (nw_lon, se_lat), (se_lon, se_lat), (se_lon, nw_lat)]))
            #add centroid
            utmcentroidx=nw_x_utm+(se_x_utm-nw_x_utm)/2
            utmcentroidy=se_y_utm+(nw_y_utm-se_y_utm)/2
            centroid_lon, centroid_lat = transform_point_utm_to_wgs84(utm_proj, utmcentroidx, utmcentroidy)
            centroid_lons.append(centroid_lon)
            centroid_lats.append(centroid_lat)
            #calculate diameter
            diameter.append(calculate_diameter(bbox))
            
    #create geodatabase
    gdf = gpd.GeoDataFrame({"object_class": object_class, 'tile_name': tile_names,'image_name': chip_names, 
            "nw_x_pixel_object_coord": minx_pixels, "nw_y_pixel_object_coord": miny_pixels, #min lon/lat
            "se_x_pixel_object_coord": maxx_pixels, "se_y_pixel_object_coord": maxy_pixels, #max lat
            "utm_projection": utm_projection,
            "nw_x_utm_object_coord": nw_x_utms, "nw_y_utm_object_coord": nw_y_utms, #utm min
            "se_x_utm_object_coord": se_x_utms, "se_y_utm_object_coord": se_y_utms, #utm max             
            "nw_lat_object_coord": nw_lats, "nw_lon_object_coord": nw_lons,#min lon/lat
            "se_lat_object_coord": se_lats, "se_lon_object_coord": se_lons, #min lon/lat
            "centroid_lon_object_coord": centroid_lons, "centroid_lat_object_coord": centroid_lats, #centroid
            'geometry': geometry, 'diameter': diameter, 'merged_bbox': merged_bbox})
    return gdf
      
def getFeatures(gdf): #used
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def identify_political_boundaries_for_each_tank(counties, tile_level_annotations): #used
    """ State Names for Tile Database 
    #https://gis.stackexchange.com/questions/251812/returning-percentage-of-area-of-polygon-intersecting-another-polygon-using-shape
    Add county and state data to tile dataframe
    Args:
        states_data_path:
        tile_level_annotations:

    Returns:

    """
    counties = counties.to_crs(epsg=4326) #reproject to lat lon
    #get counties for each polygon
    county = [None] * len(tile_level_annotations)
    county_fips = [None] * len(tile_level_annotations)
    state_fips = [None] * len(tile_level_annotations)

    for tank_index, tank_poly in tqdm(enumerate(tile_level_annotations["geometry"])): #iterate over the tank polygons
        for county_index, county_poly in enumerate(counties["geometry"]): #iterate over the county polygons
            if county_poly.intersects(tank_poly) or county_poly.contains(tank_poly): #check if tank is in polyon
                if county[tank_index] == None:
                    # change the counties if the polygon mainly resides in a different county
                    # add counties name, county fips, and state fips for each tank to list
                    county[tank_index] = counties.iloc[county_index]["NAME"]
                    county_fips[tank_index] = counties.iloc[county_index]["COUNTYFP"]
                    state_fips[tank_index] = counties.iloc[county_index]["STATEFP"]

                else:
                    # check percent of tank that intersects with current county
                    index, = np.where(counties["NAME"] == county[tank_index])
                    prev_county_poly = counties["geometry"][index[0]]
                    prev_county_poly_intersection_area = tank_poly.intersection(prev_county_poly).area/tank_poly.area #percent intersects with prev_county_poly
                    proposed_county_poly_intersection_area = tank_poly.intersection(county_poly).area/tank_poly.area #percent intersects with proposed county
                    if proposed_county_poly_intersection_area > prev_county_poly_intersection_area: 
                        # change the counties if the polygon mainly resides in a different county
                        # add counties name, county fips, and state fips for each tank to list
                        county[tank_index] = counties.iloc[county_index]["NAME"]
                        county_fips[tank_index] = counties.iloc[county_index]["COUNTYFP"]
                        state_fips[tank_index] = counties.iloc[county_index]["STATEFP"]
    #add data to dataframe
    tile_level_annotations['county'] = county
    tile_level_annotations['state_fips'] = state_fips
    tile_level_annotations['county_fips'] = county_fips
    tile_level_annotations['county_fips'] = tile_level_annotations['state_fips'] + tile_level_annotations['county_fips']

    return tile_level_annotations


def write_gdf(gdf, output_filepath, columns_to_drop_for_shp=["image_name"], output_filename = 'tile_level_annotations'): #used
    """
    Save tile level annotations 
    """
    gdf.crs = "EPSG:4326" #assign projection
    #save geodatabase as json
    with open(os.path.join(output_filepath, output_filename+".json"), 'w') as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as geojson 
    with open(os.path.join(output_filepath, output_filename+".geojson"), "w") as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as shapefile
    gdf_shapefile = gdf.drop(columns=columns_to_drop_for_shp)
    gdf_shapefile.to_file(os.path.join(output_filepath, output_filename+".shp"))
    
        
def get_img_xml_paths(img_paths_anno_paths): #used
    img_paths = []
    xml_paths = []
    for directory in tqdm(img_paths_anno_paths):
        #get all the image and xml paths in directory of annotated images
        remove_thumbs(directory[0])
        img_paths += sorted(glob(directory[0] + "/*.jpg", recursive = True))
        xml_paths += sorted(glob(directory[1] + "/*.xml", recursive = True))  
    return(img_paths, xml_paths)


def intersection_of_sets(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
      
    # Calculates intersection of sets on s1 and s2
    set1 = s1.intersection(s2)         #[80, 20, 100]
      
    # Calculates intersection of sets on set1 and s3
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    print(len(final_list))
    return(final_list)

def add_chips_to_chip_folders(rechipped_image_path, tile_name):
    """ 
    Args:
    remaining_chips_path(str): path to folder that will contain all of the remaining images that have not been labeled and correspond to tiles that have labeled images
    tile_name(str): name of tile without of extension
    Returns:
    """
    chips_path = os.path.join(rechipped_image_path, tile_name, "chips")
    os.makedirs(chips_path, exist_ok=True)
    
    item_dim = int(512)
    tile = cv2.imread(os.path.join(tiles_complete_dataset_path, tile_name + ".tif"),cv2.IMREAD_UNCHANGED)
    tile_height,  tile_width,  tile_channels = tile.shape #the size of the tile 
    row_index = math.ceil(tile_height/512) 
    col_index = math.ceil(tile_width/512)
    #print(row_index, col_index)

    count = 1            
    for y in range(0, row_index): #rows
        for x in range(0, col_index): #cols
            chip_img = tile_to_chip_array(tile, x, y, item_dim)
            #specify the chip names
            chip_name_correct_chip_name = tile_name + '_' + f"{y:02}"  + '_' + f"{x:02}" + '.jpg' # The index is a six-digit number like '000023'.
            if not os.path.exists(os.path.join(chips_path, chip_name_correct_chip_name)):
                cv2.imwrite(os.path.join(chips_path, chip_name_correct_chip_name), chip_img) #save images  