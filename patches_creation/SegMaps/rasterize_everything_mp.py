import math
import os
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import glob
from tqdm import tqdm
import geopandas as gpd
from PIL import Image
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import ColorInterp
from rasterio.crs import CRS
import json
from rasterio.merge import merge
import glob
from PIL import Image
from multiprocessing import Pool, cpu_count

def rasterize_bdortho_parallel(args):
    # Open the BD Ortho image
    bdortho_image_path, bdtopo_shapefiles, out_tif_path, objs, list_of_all_shps, folder = args
    mapJson = {}
    crs = "EPSG:2154"
    # Create an empty array with the same shape as the BD Ortho image
    with rasterio.open(bdortho_image_path) as src:
        transform = src.transform
        left, bottom, right, top = src.bounds
        bbox = box(left, bottom, right, top)
        # Get the shape of the BD Ortho image
        out_shape = src.shape
    rasterized_array = np.zeros((1000, 1000), dtype=np.int32)
    allRasters = []
    latestIndex = 0
    # Loop through the list of BD Topo shapefiles
    for i, shpfile in enumerate(list_of_all_shps):
        temp_bool = False
        for i, shapefile in (enumerate(bdtopo_shapefiles)):
            shp_path = shapefile.split("\\")[:-1]
            shpname = shapefile.split("\\")[-1]
            shp_path = "\\".join(shp_path)

            if shpfile == shpname and os.path.exists(os.path.join(shp_path, shpfile)):
                temp_bool = True
                shapefile = os.path.join(shp_path, shpfile)
                break
        if temp_bool:
            # Read the shapefile into a GeoDataFrame
            gdf = gpd.read_file(shapefile)

            intersection = gpd.overlay(gdf, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:2154"),
                                       keep_geom_type=False, how='intersection')
            del gdf
            # Iterate over the geometries in the shapefile
            """for index, row in gdf.iterrows():
                # Check if the geometry is entirely within the bounding box of the raster
                if row['geometry'].within(bbox):
                    # If it is, keep it as is
                    pass
                else:
                    if not row["geometry"].is_valid:
                        row["geometry"] = row["geometry"].buffer(0)
                    # If it isn't, crop it to the bounding box of the raster
                    gdf.at[index, 'geometry'] = row['geometry'].intersection(bbox)
            # Rasterize the GeoDataFrame and add it to the rasterized array"""
            if intersection.is_empty.all():
                data = np.zeros((1000, 1000), dtype="uint8")
                mapJson[shpfile.split(".")[0]] = f"{len(allRasters)}"

                allRasters.append(data)
            else:
                shapes = ((geom, value + 1) for geom, value in zip(intersection.geometry, intersection.index) if geom.is_valid and not geom.is_empty)
                burned = features.rasterize(shapes=shapes, out_shape=out_shape, transform=transform, all_touched=False)


                mask = (burned > 0)
                # if not (mask == 0).all():
                mapJson[shpfile.split(".")[0]] = f"{len(allRasters)}"
                allRasters.append(burned)


        else:

            data = np.zeros((1000, 1000), dtype="uint8")
            mapJson[shpfile.split(".")[0]] = f"{len(allRasters)}"

            allRasters.append(data)

    if len(allRasters) > 0:

        rastersDir = os.path.join(out_tif_path, "rasters")
        if not os.path.isdir(rastersDir):
            os.mkdir(rastersDir)
        else:
            files = glob.glob(rastersDir + '/*')
            for f in files:
                os.remove(f)

        out_meta = {
            'driver': 'GTiff',
            'height': out_shape[0],
            'width': out_shape[1],
            'count': len(allRasters),
            'dtype': 'uint32',  # Use an unsigned 16-bit integer data type to store values between 0 and 65535
            'crs': crs,
            'transform': transform,
            'compress':  'lzw'
        }

        with rasterio.open(os.path.join(rastersDir, "raster" + folder + ".tiff"), 'w', **out_meta) as dst:
            dst.write(np.array(allRasters))
        del allRasters
        """with rasterio.open(os.path.join(rastersDir, "raster" + folder + ".tiff"), 'w', **out_meta) as dst:
            dst.write(np.array(allRasters))"""
        """with open(os.path.join(rastersDir,"raster"+folder+".json"), 'w') as dst:
            json.dump(mapJson, dst)"""

if __name__ == '__main__':
    categories = ["BATI", "HYDROGRAPHIE", "OCCUPATION_DU_SOL", "TRANSPORT", "ZONE_REGLEMENTEES",
                  "SERVICES_ET_ACTIVITES"]
    list_of_all_shps = ["BATIMENT.shp", "CIMETIERE.shp", "TERRAIN_DE_SPORT.shp", "RESERVOIR.shp", "PYLONE.shp",
                        "CONSTRUCTION_SURFACIQUE", "ZONE_D_ESTRAN.shp", "ZONE_DE_VEGETATION.shp",
                        "SURFACE_HYDROGRAPHIQUE.shp", "AERODROME.shp", "EQUIPEMENT_DE_TRANSPORT.shp",
                        "TRONCON_DE_ROUTE.shp", "TRONCON_DE_VOIE_FERREE.shp", "FORET_PUBLIQUE.shp",
                        "PARC_OU_RESERVE.shp", "TOPONYMIE_SERVICES_ET_ACTIVITES.shp"]
    DATA_DIR = 'D:\\data'
    for dept in ["34"]:

        sub_dir = "BDOrtho+Sentinel2/test_version8/" + dept
        patch_name = dept
        SUBDIRS_PATH = f"/gpfsssd/scratch/rech/pen/uwh84qh/DatasetParis/{dept}/{dept}"
        #SUBDIRS_PATH = "E:\\Hichem\\74"
        SUBDIR_PATH = f"/gpfsssd/scratch/rech/pen/uwh84qh/DatasetParis/{dept}\\BDTopo_new"
        Folders = next(os.walk(SUBDIRS_PATH))[1]
        args_list = []
        step = math.floor(len(Folders)/5000)
        folders_to_process =[]
        for i in tqdm(range(0,len(Folders),step)):
            folder = Folders[i]
            folders_to_process.append(folder)
            if folder == "Q&A":
                continue
            SUBDIR_PATH1 = os.path.join(SUBDIRS_PATH, folder)
            allFiles = os.listdir(SUBDIR_PATH)
            alLFilesInFolder = os.listdir(SUBDIR_PATH1)
            list_of_categories = [cat for cat in allFiles if cat in categories]

            rst_fn = os.path.join(SUBDIR_PATH1,
                                  [f for f in alLFilesInFolder if "BDOrtho_" in f and f.split(".")[-1] != "xml"][0])
            list_of_categories.sort()
            list_of_files = []
            for cat in list_of_categories:
                for dirpath, _, filenames, in os.walk(os.path.join(SUBDIR_PATH, cat)):
                    for f in filenames:
                        list_of_files.append(os.path.join(dirpath, f))
            list_of_shp_files = [os.path.join(SUBDIR_PATH, f) for f in list_of_files if
                                 ".shp" in f and not "BASSIN_VERSANT" in f]
            list_of_shp_files.sort()

            args_list.append((rst_fn, list_of_shp_files, SUBDIR_PATH1, categories, list_of_all_shps, folder))
        print(len(folders_to_process))
        with open(f"/gpfsssd/scratch/rech/pen/uwh84qh/DatasetParis\\{dept}\\{dept}\\Q&A\\folders_rasterized.json", "w") as fp:
        #with open(f"E:\\Hichem\\74\\Q&A\\folders_rasterized.json", "w") as fp:
            json.dump(folders_to_process, fp)
        print(len(args_list))
        with Pool(processes=10) as pool:
            results = list(tqdm(pool.imap(rasterize_bdortho_parallel, args_list, chunksize=2), total=len(folders_to_process)))
