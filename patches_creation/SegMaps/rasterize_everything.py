# import all necessary libraries
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


def rasterize_bdortho(bdortho_image_path, bdtopo_shapefiles, out_tif_path, objs, list_of_all_shps, folder):
    # Open the BD Ortho image
    mapJson = {}
    crs = "EPSG:2154"
    # Create an empty array with the same shape as the BD Ortho image
    with rasterio.open(bdortho_image_path) as src:
        transform = src.transform

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
            bounds = src.bounds
            bbox = box(*bounds)

            # Iterate over the geometries in the shapefile
            for index, row in gdf.iterrows():
                # Check if the geometry is entirely within the bounding box of the raster
                if row['geometry'].within(bbox):
                    # If it is, keep it as is
                    pass
                else:
                    if not row["geometry"].is_valid:
                        row["geometry"] = row["geometry"].buffer(0)
                    # If it isn't, crop it to the bounding box of the raster
                    gdf.at[index, 'geometry'] = row['geometry'].intersection(bbox)
            # Rasterize the GeoDataFrame and add it to the rasterized array

            shapes = ((geom, value + 1) for geom, value in zip(gdf.geometry, gdf.index) if geom.is_valid and not geom.is_empty)
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
        """with rasterio.open(os.path.join(rastersDir, "raster" + folder + ".tiff"), 'w', **out_meta) as dst:
            dst.write(np.array(allRasters))"""
        """with open(os.path.join(rastersDir,"raster"+folder+".json"), 'w') as dst:
            json.dump(mapJson, dst)"""


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
    SUBDIRS_PATH = os.path.join(DATA_DIR, sub_dir)
    SUBDIR_PATH = f"D:\\data\\BD_Topo\\{dept}\\BDTopo_new"
    Folders = next(os.walk(SUBDIRS_PATH))[1]
    allFiles = os.listdir(SUBDIR_PATH)
    list_of_categories = [cat for cat in allFiles if cat in categories]

    list_of_categories.sort()
    list_of_files = []
    for cat in list_of_categories:
        for dirpath, _, filenames, in os.walk(os.path.join(SUBDIR_PATH, cat)):
            for f in filenames:
                list_of_files.append(os.path.join(dirpath, f))
    list_of_shp_files = [os.path.join(SUBDIR_PATH, f) for f in list_of_files if
                         ".shp" in f and not "BASSIN_VERSANT" in f]
    list_of_shp_files.sort()
    for folder in tqdm(Folders):
        if folder == "Q&A":
            continue
        SUBDIR_PATH1 = os.path.join(SUBDIRS_PATH, folder)

        alLFilesInFolder = os.listdir(SUBDIR_PATH1)

        rst_fn = os.path.join(SUBDIR_PATH1, [f for f in alLFilesInFolder if "BDOrtho_" in f and f.split(".")[-1] != "xml"][0])

        """for cat in list_of_categories:
            list_of_files = os.listdir(os.path.join(SUBDIR_PATH,cat))
            list_of_shp_files = [os.path.join(SUBDIR_PATH,cat,f) for f in list_of_files if ".shp" in f and not "BASSIN_VERSANT" in f]
            list_of_shp_files.sort()"""
        rasterize_bdortho(rst_fn, list_of_shp_files, SUBDIR_PATH, categories, list_of_all_shps, folder)
