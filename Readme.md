# Above Ground Stroage Tank Data Pipeline
> This repository contains the necessary python files to download and process NAIP tiles; chip tiles in 512 x 512 images while conserving resolution, allocate images to annotators; seperate postive and negatives; standardize object labels; allocate images for verfication; compile a data summary. 

## Table of Contents
* [General Info](#general-information)
* [Overview](#technologies-used)
    * [0. Tile Identification](#tile-identification )
    * [1. Data Download](#data-download)
        * [Naming Convention](#naming-convention)
            * [Tile Naming Convention](#tile-naming-convention)
            * [Chip Naming Convention](#chip-naming-convention)
    * [2. Seperate Positive and Negative Images](#seperate-positive-and-negative-images)
    * [3. Record Annotator](#record-annotator)
    * [4. Verification and Tracking](#verification-and-tracking)
    * [5. Standardize Object Labels](#standardize-object-labels)
    * [6. Create Tank Inventory](#identify-missing-annotations)
    * [7. Create Complete Dataset](#create-complete-dataset)

    * [8. Data Summary](#data-summary)
## General Information
- Data Annotation: 
    - The Annotation tool is in labelImg.zip.
    - Data annotations are created using the LabelImg graphical image annotation tool. The tool has been developed in Python using Qt as its graphical interface. The annotations were created as XML files in the PASCAL VOC format. Unless otherwise noted, annotations and bbounding boxes in the source code are assumed to the in Pascal VOC Format [x_min, y_min, x_max, ymax], where x_min and y_min are coordinates of the top-left corner of the bounding box and x_max and y_max are the coordinates o the bottom right corner.
    - The installation infromation and guide can be found at https://github.com/tzutalin/labelImg. 
## 0. Tile Identification 
NAIP data is accessed using the Microsoft Planetary Computer STAC API. To download tiles of interest, the file pathway have been identified and collected using the EIA, HFID, and other datasources. The naip_pathways.ipynb jupyter notebook processess these datasources and creates/saves a numpy array of the tile names and tile urls.

### Naming Convention:
#### Tile Naming Convention:
**Quadrangle:** 
USGS quadrangle identifier, specifying a 7.5 minute x 7.5 minute area.
The filename component of the path (m_3008601_ne_16_1_20150804 in this example) is preserved from USDA’s original archive to allow consistent referencing across different copies of NAIP. Minor variation in file naming exists, but filenames are generally formatted as: 
m_[quadrangle]_[quarter-quad]_[utm zone]_[resolution]_[capture date].tif
For example, the above file is in USGS quadrangle 30086, in the NE quarter-quad, which is in UTM zone 16, with 1m resolution, and was first captured on 8/4/2015. 
#### Chip Naming Convention:
All chips' names include information: state_resolution_year_quadrangle_filename_index
The two npy files record the tiles have been labeled (tile_name_tile_url_labeled) and have not been labeled(tile_name_tile_url_remaining).
Each tile 
**Index**
The index of a 512x512 chips which is clipped from a tile. 

## 1. Data Download
download_distribution.py distributes tiles to annotators.

python download-distribution.py --number_of_tiles number
                                            --annotation_directory annotator_name
                                            --parent_directory \dir_containing_chips_and_annotations
                                            --tiles_remaining path_to_numpy_array
                                            --tiles_labeled path_to_numpy_array
 
## 2. Seperate Positive and Negative Images
After the annotators have annotated images, the positive (the images containing objects) and negative (the images that do not contain objects) must be placed in seperate folders.
This can be completed by running the following script in the command line.

seperate_positive-negative-images.py  --annotation_directory annotator_name
                                      --parent_directory \dir_containing_chips_and_annotations
                                             
Example:
python seperate-positive-negative-images.py  --annotation_directory unverified_images_not_reviewed_by_student31_Poonacha --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\labelwork

python seperate-positive-negative-images.py  --annotation_directory student_reviewed_images17_Sunny --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\unverified_images\student_reviewed_unverified_images_set7\Sunny

## 3. Record Annotator
After the annotators have reviewed their images to fix any small errors, the organizer relocates their images into the *Unverified* folder. This folder is organized by annotator, by annotation set. To record which annotations have been recorded by which annotator in a centralized location, the following script is run. This produces two outputs, a npy array and a csv which indicate the tile, chip, xml, and annotator. 

python track-annotator-draw.py  --parent_directory \dir_containing_all_chips_and_annotations
                                             
## 4. Verification and Tracking
After the annotators have reviewed their images to fix any small errors, the organizer relocates their images into the *Unverified* folder. This folder is organized by annotator, by annotation set. To record which annotations have been recorded by which annotator in a centralized location, the following script is run. This produces two outputs, a npy array and a csv which indicate the tile, chip, xml, and annotator. 

python verification-and-tracking.py     --tracker_file_path path_to_tracker_numpy_array
                                        --home_directory path_to_home_directory
                                        --verifiers coverage_quality_class
                                        --set_number the_set_number
                                        --annotator_allocation annotator1 annotator2


## 5. Standardize Object Labels 
While the predefined classes have been provided to annotators, it is possible for there to be inconsistencies in the label names. The object label names are standardized based on known errors using the compiled imagery and annotation datasets.

python correct_incorrect_labels.py --parent_dir compiled_imagery_dataset

## 6. Create Tank Inventory
The annotations for each imagery are merged by each time to create a national invetory of above ground storage tanks.

tile-level-annotation.py --parent_dir compiled_imagery_dataset --xml_folder_name chips_positive_corrected_xml 
                         --tile_dir tile_dir --tile_level_annotation_dir output_dir --tile_level_annotation_dataset_filename file_name
                         --county_gpd_path zipped_county_data

## 7. Create Complete Dataset
make_complete_dataset.py

## 8. Data Summary
- data_clean_descrip.py
