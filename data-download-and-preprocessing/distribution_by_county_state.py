"""
Data Summary of Labeled Images (To - Date)
"""

"""
Import Packages
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Creates a table of the number of distribution of tanks by class by county and state.')
    parser.add_argument('--output_dir', type=str, default="/hpc/group/borsuklab/ast/distribution",
                        help='Directory to store output files.')
    parser.add_argument('--tile_level_annotation_dataset_path', type=str, 
                        default="/hpc/group/borsuklab/ast/tile-level-annotations/tile_level_annotations.shp", 
                        help='path to tif path.')
    args = parser.parse_args()
    return args


def main(args): 
    tile_level_annotations = gpd.read_file(args.tile_level_annotation_dataset_path)
    tile_level_annotations = tile_level_annotations.to_crs(epsg=4326) #reproject to gps coords
    tile_level_annotations["county_id"] =  tile_level_annotations.state_fips + tile_level_annotations.county_fip
    
    #State dist
    state_count = pd.crosstab(tile_level_annotations.state_fips, tile_level_annotations.object_cla, 
                              margins=True, margins_name="Total")
    state_percent = pd.crosstab(tile_level_annotations.state_fips, tile_level_annotations.object_cla, 
                                margins=True, normalize = True,  margins_name="Total")
    
    #county dist
    county_id_count = pd.crosstab(tile_level_annotations.county_id, tile_level_annotations.object_cla,
                                  margins=True, margins_name="Total")
    county_id_percent = pd.crosstab(tile_level_annotations.county_id, tile_level_annotations.object_cla,
                                    margins=True, normalize = True, margins_name="Total")
    
    #Write
    os.makedirs(args.output_dir, exist_ok = True)
    state_count.sort_values(by=['Total'], ascending=False).to_csv('AST distribution by state count.csv')
    state_percent.sort_values(by=['Total'], ascending=False).to_csv('AST distribution by state percent.csv')
    county_id_count.sort_values(by=['Total'], ascending=False).to_csv('AST distribution by county count.csv')
    county_id_percent.sort_values(by=['Total'], ascending=False).to_csv('AST distribution by county percentage.csv')

if __name__ == '__main__':
    args = get_args_parse()
    main(args)    