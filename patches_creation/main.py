# import all necessary libraries
import os
import concurrent.futures

import rasterio
import rasterio.mask
import torch
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests
import cv2 as cv

from utils import plot_grid
from shapely.geometry import Polygon, box
import os
import shutil
import warnings
from multiprocessing import Pool
from bs4 import BeautifulSoup
from shapely.geometry import Polygon, Point
from read_data import read_data, printc
from dataset_creation import map_coords_datasets, crop_images, cut_image, cut_raster_into_patches
from utils import print_color, add_index_column
from dir_utils import getListOfFiles
from tqdm import tqdm
from dataset_creation import create_df
from dir_utils import getListOfFiles
from rasterio.mask import mask
from rasterio.plot import show
from pyproj import CRS


def remove_non_directories(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isdir(file_path):
            os.remove(file_path)


def refactorDirs(new_ign_coords_df, cur_dir, target_dir):
    """
    For each data patch creates an independent folder of the following format:
    |-- Patch
    |    |-- RGB_from_orthophoto.tif (or jp2)
    |    |-- S2_B01.tif (or jp2)
    |    |-- S2_B02.tif (or jp2)
    |    |-- and so on…
    |    |-- Metadata.json (with info such as when were the images acquired, original name of the image, which sensor                (S2A or B), …)
    |    |-- VQA.json (with ground truth for VQA task)
    Input:
    :ign_sentinel_df:
    :x_df:
    """
    # collect BDOrtho files
    for idx, row in enumerate(new_ign_coords_df.iterrows()):
        path = row[1]["data_path"]
        new_folder_path = target_dir + "/" + str(row[0])
        if not os.path.isdir(new_folder_path):
            os.mkdir(new_folder_path)
        file_name = path.split("\\")[-1]

        shutil.move(path, new_folder_path + "/" + file_name)

    # collect IGN files
    all_files = [file for file in os.listdir(cur_dir) if "Sentinel-2" in file]
    ign_files = {}
    ign_files = {
        int(file.split("_")[2]): file for file in all_files
    }
    for idx, path in ign_files.items():
        new_folder_path = target_dir + "/" + str(idx)
        file_name = path.split("/")[-1]
        shutil.move(path, new_folder_path + "/" + file_name)
def process_shp_file(shp_file,rasters_gdf):
    print(shp_file)
    gdf = gpd.read_file(shp_file)

    intersection = gpd.sjoin(gdf, rasters_gdf, how='inner', op='intersects')
    grouped = intersection.groupby("data_path")

    for name, group in grouped:
        tg_dir = os.path.join(target_dir, str(group.index_right.values[0]))

        if not os.path.isdir(os.path.join(target_dir, tg_dir, bdt_dir)):
            os.makedirs(os.path.join(target_dir, tg_dir, bdt_dir), exist_ok=True)

        df1 = group.drop('data_path', axis=1)
        gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
        gdf1.to_file(os.path.join(target_dir, tg_dir, bdt_dir, os.path.basename(shp_file)))


# set up all the directories

def main():
    # DATA_DIR = '/home/boussaid/Downloads/STAGE/_stage_LIPADE/data'
    DATA_DIR = 'D:\\data'
    #DATA_DIR = 'E:\\Dataset\\ORTHO'

    gf_dir = 'GeoFabrik_Ile_de_France_10_02'  # GeoFabrik Ile-de-France directory
    GF_DIR = os.path.join(DATA_DIR, gf_dir)

    bdt_dir = 'BDTOPO_3-0_TOUSTHEMES_SHP_LAMB93_D075_2021-03-15/BDTOPO'
    bdt_donnees_dir = "1_DONNEES_LIVRAISON_2021-03-00272/BDT_3-0_SHP_LAMB93_D075-ED2021-03-15"
    bdt_sup_dir = "2_METADONNEES_LIVRAISON_2021-03-00272/BDT_3-0_SHP_LAMB93_D075-ED2021-03-15/EMPRISE"
    bdt_metadon_dir = "3_SUPPLEMENTS_LIVRAISON_2021-03-00272"
    BDT_DIR = os.path.join(DATA_DIR, bdt_dir, bdt_donnees_dir)

    bdo_dir = 'ORTHOHR_1-0_RVB-0M20_JP2-E080_LAMB93_D075_2021-01-01/BDORTHO'  # BDORTHO data directory
    bdo_donnees_dir = '1_DONNEES_LIVRAISON_2022-06-00123/OHR_RVB_0M20_JP2-E080_LAMB93_D75-2021'
    bdo_sup_dir = '3_SUPPLEMENTS_LIVRAISON_2022-06-00123/OHR_RVB_0M20_JP2-E080_LAMB93_D75-2021'
    bdo_metadon_dir = '2_METADONNEES_LIVRAISON_2022-06-00123'

    IGN_DIR = os.path.join(DATA_DIR, bdo_dir, bdo_donnees_dir)
    # We store all these directories as a dictionary
    depts_not_to_look_into = ['74','34']

    for dept in os.listdir(os.path.join(DATA_DIR, "BD_Ortho")):
    #for dept in os.listdir(DATA_DIR):
        """if dept == "75" or dept == "50":
            continue"""
        if dept not in depts_not_to_look_into:
            continue

        print(dept)
        IGN_DIR = os.path.join(DATA_DIR, 'BD_Ortho', dept)
        year = [x for x in os.listdir(IGN_DIR) if "comment" not in x][0]
        IGN_DIR = os.path.join(IGN_DIR, year)
        bdo_dir = os.path.join([i for i in os.listdir(IGN_DIR) if os.path.isdir(os.path.join(IGN_DIR, i))][0],
                               "ORTHOHR")
        dirs = os.listdir(os.path.join(IGN_DIR, bdo_dir))
        bdo_donnees_dir = [d for d in dirs if d[0:2] == "1_"][0]
        bdo_metadon_dir = [d for d in dirs if d[0:2] == "2_"][0]
        bdo_sup_dir = [d for d in dirs if d[0:2] == "3_"][0]
        bdo_donnees_dir = os.path.join(bdo_donnees_dir, os.listdir(os.path.join(IGN_DIR, bdo_dir, bdo_donnees_dir))[0])
        bdo_metadon_dir = os.path.join(bdo_metadon_dir, os.listdir(os.path.join(IGN_DIR, bdo_dir, bdo_metadon_dir))[0])
        bdo_sup_dir = os.path.join(bdo_sup_dir, os.listdir(os.path.join(IGN_DIR, bdo_dir, bdo_sup_dir))[0])

        bdo_dir = os.path.join(dept, year, bdo_dir)

        bdt_dir = os.path.join("BD_Topo", dept, "BDTOPO")
        dirs = os.listdir(os.path.join(DATA_DIR, "BD_Topo", dept, "BDTOPO"))
        # dirs = os.listdir(os.path.join(DATA_DIR, bdt_dir))
        bdt_donnees_dir = [d for d in dirs if d[0:2] == "1_"][0]
        bdt_metadon_dir = [d for d in dirs if d[0:2] == "2_"][0]
        bdt_sup_dir = [d for d in dirs if d[0:2] == "3_"][0]
        bdt_donnees_dir = os.path.join(bdt_donnees_dir, os.listdir(os.path.join(DATA_DIR, bdt_dir, bdt_donnees_dir))[0])
        bdt_metadon_dir = os.path.join(bdt_metadon_dir, os.listdir(os.path.join(DATA_DIR, bdt_dir, bdt_metadon_dir))[0])
        bdt_sup_dir = os.path.join(bdt_sup_dir, os.listdir(os.path.join(DATA_DIR, bdt_dir, bdt_sup_dir))[0])
        BDT_DIR = os.path.join(DATA_DIR, bdt_dir, bdt_donnees_dir)

        # bdt_dir = os.path.join("BD_Topo", dept)
        # print(bdo_dir)
        dirs = {
            "Sentinel-2": '',
            "IGN": {"files": IGN_DIR, "coords": os.path.join(DATA_DIR, "BD_Ortho", bdo_dir, bdo_sup_dir)},
            "OSM": GF_DIR,
            "BDT": BDT_DIR
        }
        # We also set up the data streams we want to read
        to_read_data = {
            "BDT": False,
            "OSM": False,
            "IGN": True,
            "Sentinel-2": False
        }  # IGN stands for BDORTHO

        data = read_data(dirs, to_read_data)
        # bdt_coords_df = data["BDT"]
        ign_jp2_files = data["IGN"]["Filenames"]
        ign_coords_df = data["IGN"]["Coordinates"]

        # === Set up directory to save the data ===
        version = "test_version8"  # to work properly should be a novel paramter
        subdir_name = "BDOrtho+Sentinel2"
        SUBDIR_PATH = os.path.join(DATA_DIR, subdir_name)

        # create directory if necessary
        try:
            if not os.path.exists(SUBDIR_PATH):
                os.mkdir(SUBDIR_PATH)
            print("Directory sucessfuly created.")
        except:
            print("Directory already exists.")
        # set up directory
        subdir_name = os.path.join(subdir_name, version)
        SUBDIR_PATH = os.path.join(DATA_DIR, subdir_name)

        # create a new directory
        try:
            if not os.path.exists(SUBDIR_PATH):
                os.mkdir(SUBDIR_PATH)
            print("Directory sucessfuly created.")
        except ValueError:
            print(f"Subdirectory {version} already exists. Please, change the name of the subdirectory.")

        dept_name = os.path.join(subdir_name, dept)
        DEPT_PATH = os.path.join(DATA_DIR, subdir_name, dept)

        # create a new directory
        try:
            if not os.path.exists(DEPT_PATH):
                os.mkdir(DEPT_PATH)
            print("Directory sucessfuly created.")
        except ValueError:
            print(f"Subdirectory {version} already exists. Please, change the name of the subdirectory.")

        os.chdir(DEPT_PATH)
        print(f"Current directory is {os.getcwd()}.")
        dept_shp = os.path.join(BDT_DIR, "ADMINISTRATIF", "DEPARTEMENT.shp")

        gdf = gpd.read_file(dept_shp)
        for row in gdf.iterrows():
            if row[1].INSEE_DEP == dept:
                geometry_of_departement = row[1].geometry
        df = ign_coords_df
        size = 1000
        new_images_bdortho = []


        for idx in tqdm(range(len(df))):

            print(df.iloc[idx]["data_path"])
            #with rasterio.open(df.iloc[idx]["data_path"]) as image:
            # show(image)
            cut_raster_into_patches(df.iloc[idx]["data_path"], DEPT_PATH, geometry_of_departement, size, name = f"BDOrtho_{idx}")


        # new_images_bdortho.append(new_images)

        # print(f"Dataset length: {len(new_images_bdortho)}")

        list_of_files = [file_name for file_name in tqdm(getListOfFiles(DEPT_PATH)) \
                         if "BDOrtho_" in file_name and ".aux.xml" not in file_name]
        new_ign_coords_df = create_df(list_of_files)
        # === Refactor directories ===
        # from dir_utils import refactorDirs

        # gdf_filtered = new_ign_coords_df[new_ign_coords_df.geometry.within(geometry_of_departement)]

        # new_ign_coords_df = gdf_filtered

        # === Set up directory to save the data ===

        cur_dir = os.path.join(DATA_DIR, DEPT_PATH, version, dept)
        target_dir = os.path.join(DATA_DIR, subdir_name, version, dept)

        refactorDirs(new_ign_coords_df, DEPT_PATH, DEPT_PATH)
        # remove_non_directories(DEPT_PATH)
        warnings.simplefilter('ignore')
        from rtree import index

        print("goes here")
        rasters_gdf = gpd.GeoDataFrame(new_ign_coords_df, geometry='geometry')
        print(rasters_gdf)
        version = "test_version8"  # to work properly should be a novel paramter
        version_2 = "test_version_v2"
        subdir_name = "BDOrtho+Sentinel2"

        cur_dir = DEPT_PATH
        target_dir = DEPT_PATH
        data = {}
        bdTopoDirs = [f.name for f in os.scandir(BDT_DIR) if f.is_dir()]

        for bdt_dir in tqdm(bdTopoDirs):

            dir_path = os.path.join(BDT_DIR, bdt_dir)
            shp_files = [filename for filename in getListOfFiles(dir_path)
                         if filename.endswith('.shp') and (not (filename.split("\\")[-1].startswith(".")))]
            for shp_file in tqdm(shp_files):
                # print(shp_file.split("\\")[-1])
                gdf = gpd.read_file(shp_file)

                intersection = gpd.sjoin(gdf, rasters_gdf, how='inner', op='intersects')

                test = (intersection.groupby("data_path"))
                for name, group in test:
                    tg_dir = os.path.join(target_dir, str(group.index_right.values[0]))

                    if not os.path.isdir(os.path.join(target_dir, tg_dir, bdt_dir)):
                        if not os.path.isdir(os.path.join(target_dir, tg_dir)) :
                            os.mkdir(os.path.join(target_dir, tg_dir))
                        os.mkdir(os.path.join(target_dir, tg_dir, bdt_dir))

                    # df1 = group.drop('data_path',axis=1)
                    df1 = group.drop('data_path', axis=1)

                    gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')

                    gdf1.to_file(os.path.join(target_dir, tg_dir, bdt_dir, shp_file.split("\\")[-1]))
        depts_not_to_look_into.append(dept)





        """#with Pool(processes=os.cpu_count()) as pool:
        for bdt_dir in tqdm(bdTopoDirs):
            dir_path = os.path.join(BDT_DIR, bdt_dir)
            shp_files = [filename for filename in getListOfFiles(dir_path)
                         if filename.endswith('.shp') and not filename.startswith(".")]
            #pool.map(process_shp_file, shp_files, rasters_gdf)

            for shp_file in shp_files:

                gdf = gpd.read_file(shp_file)
                #print(gdf)
                intersection = gpd.sjoin(gdf, rasters_gdf, how='inner', op='intersects')
                grouped = intersection.groupby("data_path")

                for name, group in grouped:
                    tg_dir = os.path.join(target_dir, str(group.index_right.values[0]))

                    if not os.path.isdir(os.path.join(target_dir, tg_dir, bdt_dir)):
                        os.makedirs(os.path.join(target_dir, tg_dir, bdt_dir), exist_ok=True)

                    df1 = group.drop('data_path', axis=1)
                    gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
                    gdf1.to_file(os.path.join(target_dir, tg_dir, bdt_dir, os.path.basename(shp_file)))"""

if __name__ == '__main__':
    main()