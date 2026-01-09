import os
import rasterio
import rasterio.mask
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm

from shapely.geometry import Polygon, box
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.windows import Window
from pyproj import CRS
from utils import get_coordinates
# from Dataset_Creation.archive_utils import get_coordinates


def map_coords_datasets(df1, df2, dataset_name1=None, dataset_name2=None):
    """
    Creates a dataframe with the corresponding index of df2 for the object in df1,
    i.e. for each object in df1 we output the image in which this object is present.
    Note: the order of the dataframes is important!
    """
    coords_dct = {}

    for idx, df1_row in (df1.iterrows()):
        obj = df1_row["geometry"]
        # get coordinates of the object
        obj_x_coords = np.array(obj.exterior.coords.xy[0])
        obj_y_coords = np.array(obj.exterior.coords.xy[1])

        # get coordinates of mosaic piece
        for img_idx, row in (df2.iterrows()):
            img_boundaries = row["geometry"]
            img_x_coords = np.array(img_boundaries.exterior.coords.xy[0])
            img_y_coords = np.array(img_boundaries.exterior.coords.xy[1])

            # check if the object is inside the image
            is_inside = obj_x_coords.max() >= img_x_coords.min() and \
                        obj_x_coords.min() >= img_x_coords.min() and \
                        obj_x_coords.max() <= img_x_coords.max() and \
                        obj_x_coords.min() <= img_x_coords.max() and \
                        obj_y_coords.max() >= img_y_coords.min() and \
                        obj_y_coords.min() >= img_y_coords.min() and \
                        obj_y_coords.max() <= img_y_coords.max() and \
                        obj_y_coords.min() <= img_y_coords.max()
            if is_inside:
                coords_dct[idx] = img_idx
                break

    # get coordinates of corresponding objects:
    coords1 = []
    coords2 = []
    for idx, img_idx in coords_dct.items():
        coords1.append(df1.loc[idx]["geometry"])
        coords2.append(df2.loc[img_idx]["geometry"])

    coords_df = pd.DataFrame({dataset_name1 + "_idx": list(coords_dct.keys()),
                              dataset_name1 + "_coords": coords1,
                              dataset_name2 + "_idx": list(coords_dct.values()),
                              dataset_name2 + "_coords": coords2})

    return coords_df


def create_df(data_path_list: list, new_crs=2154):
    """
    Given a data path with the .jp2 images, the function creates a GeoPandas
    dataframe with two columns: - name of the file,
                                - Polygon obtained from its bounding boxes.
    It maps the dataframe to a new coordinate reference system (CRS).
    ------------------------------------------------------------------------
    :param data_path: list with the the data paths of the images
    :param new_crs: EPSG number of the target coordinate reference system
    :return: dataframe with two columns
    """
    # prepare the data for the DataFrame
    data = {"data_path": data_path_list,
            "geometry": [get_coordinates(path)[1] for path in tqdm.tqdm(data_path_list)]}

    # create DataFrame
    
    crs = set([get_coordinates(path)[2] for path in tqdm.tqdm(data_path_list)])  # get crs
    print(len(crs))
    if len(crs) == 1:
        crs = crs.pop()
    else:
        raise ValueError(f"Provided images have different coordinate reference systems: {crs}")
    df = gpd.GeoDataFrame(data, crs=crs)

    # change crs
    if new_crs is not None:
        df = df.to_crs(epsg=new_crs)

    return df


def transform_to_coords(coordinates):
    return [{"type": Polygon,
           "coordinates": list(coordinates.exterior.coords)}]



def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def transform_to_box(bbox_coords, crs_out, crs_in=2154, scale=(10, 20)):
    """
    Transforms the corners coordinates of the image into a suitable format
    :param bbox_coords: get the coordinates of the file
    :param crs_in: input EPSG (coordinate reference system)
    :param crs_out: output EPSG (coordinate reference system)
    :return: coordinates in suitable format
    """
    x, y = bbox_coords.exterior.xy
    bbox = box(min(x), min(y), max(x), max(y))
    
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=CRS(crs_in))
    geo = geo.to_crs(crs=crs_out)
    coords = getFeatures(geo)

    return coords
def crop_images(data_path: str,
                output_df: gpd.GeoDataFrame,
                x_df: gpd.GeoDataFrame,
                input_idx: int,
                out_idx: int,
                debug=False,
               size=(2000,2000)):
    """
    TO DO:
        - add overlapping and boundaries (perhaps in the previous step) ?

    Cuts image into patches the dataframe.
    :param data_path: path of the image to cut
    :param output_df: dataset to process
    :param x_df:
    :param input_idx: row number of the reference dataset
    :param out_idx: row number of the dataset to process
    :param debug: whether to print several important parameters of the algorithm or not
    :return:
        :out_img: the cutted image
        :out_transform: transformation of the image (metadata that comes from the mask function)
        :rasterized image saved as .tif file
    """
    # ========================
    input_name, output_name = x_df.columns[[0, 2]]
    input_name, output_name = input_name[:-3], output_name[:-3]
    
    # === Get coordinates ===
    entry = x_df[x_df[input_name + "idx"] == input_idx]
    input_coords = entry[input_name + "coords"].values[0]
    out_file = output_df[output_df["index"] == out_idx]
    
    src_path = out_file["data_path"].values[0]

    with rasterio.open(src_path) as src:
        # change epsg of bounding box
        epsg_code = int(src.crs.data['init'][5:])
        transformed_coords = transform_to_box(input_coords, epsg_code)

        if debug:
            print("EPSG_code", epsg_code)
            print(transformed_coords)
        
        out_img, out_transform = mask(src, transformed_coords,crop=True)

        
        
        # === Update the metadata ===
        # Copy the metadata
        out_meta = src.meta.copy()
        if debug:
            print("\n", out_meta)

        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": CRS(epsg_code)})

        # Output raster
        out_tif_name = f"{input_name}{input_idx}_{output_name}{out_idx}.tif"
        out_tif_name = os.path.join(data_path, out_tif_name)
        with rasterio.open(out_tif_name, "w", **out_meta) as dest:
            dest.write(out_img)
            print(f"File {out_tif_name} successfully created!")
        with rasterio.open(out_tif_name) as src1 :
            xsize, ysize = size


            # Generate a random window origin (upper left) that ensures the window 
            # doesn't go outside the image. i.e. origin can only be between 
            # 0 and image width or height less the window width or height
            xmin, xmax = 1, src1.width
            ymin, ymax = 1, src1.height

            # Create a Window and calculate the transform from the source dataset    
            window = Window(xmin, ymin, xsize, ysize)
            transform = src1.window_transform(window)

            # Create a new cropped raster to write to
            profile = src1.profile
            profile.update({
                "driver": "GTiff",
                'height': xsize,
                'width': ysize,
                'transform': transform,
                "crs": CRS(epsg_code)})

            with rasterio.open(out_tif_name, 'w', **profile) as dst:
                # Read the data from the window and write it to the output raster
                dst.write(src1.read(window=window))
             

    if debug:
        # Open the clipped raster file
        with rasterio.open(out_tif_name) as clipped:
            # Show statistics
            print("\nCRS of the output image:", clipped.crs)
            print(f"\nHeight & width: {clipped.width} x {clipped.height}")

            # Visualize
            show(clipped, cmap='terrain')

    return out_img, out_transform


"""def cut_raster_into_patches(src, output_dir, geometry, patch_size=1000,name="BDOrtho_"):
        idx = 0
        for x in range(0, src.width, patch_size):
            for y in range(0, src.height, patch_size):
                window = Window(x, y, patch_size, patch_size)
                transform = src.window_transform(window)
                patch = src.read(window=window)
                profile = src.profile
                profile.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': transform
                })
                # Check if the patch is within the given geometry
                patch_bounds = src.window_bounds(window)
                patch_box = box(*patch_bounds)
                if patch_box.within(geometry):
                    # Save the patch to a new file
                    output_file = os.path.join(output_dir, name + f"_{idx}.tif")
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(patch)
                        if idx == 0:
                            print(output_file)
                    idx+=1"""
import os
import rasterio
from shapely.geometry import box
from rasterio.windows import Window
from multiprocessing import Pool


def process_patch(args):
    input_raster_path, output_dir, geometry, patch_size, name, idx, x, y = args

    with rasterio.open(input_raster_path) as src:
        epsg_code = int(src.crs.data['init'][5:])
        window = Window(x, y, patch_size, patch_size)
        transform = src.window_transform(window)
        patch = src.read(window=window)
        profile = src.profile
        profile.update({"driver": "GTiff",
            'height': patch_size,
            'width': patch_size,
            'transform': transform,
            "crs": CRS(epsg_code)
        })
        patch_bounds = src.window_bounds(window)
        patch_box = box(*patch_bounds)

    if patch_box.within(geometry):
        output_file = os.path.join(output_dir, name + f"_{idx}.tif")
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(patch)
        return output_file
    return None


def cut_raster_into_patches(input_raster_path, output_dir, geometry, patch_size=1000, name="BDOrtho_"):
    args_list = []
    idx = 0
    with rasterio.open(input_raster_path) as input_raster:
        for x in range(0, input_raster.width, patch_size):
            for y in range(0, input_raster.height, patch_size):
                args_list.append((input_raster_path, output_dir, geometry, patch_size, name, idx, x, y))
                idx += 1

    with Pool() as pool:
        result_files = pool.map(process_patch, args_list)

    """for result_file in result_files:
        if result_file:"""
    print(len(result_files))


def cut_image(image, size=1024, debug=False,
              data_path=None, name=None):
    """
    Cuts the image into patches and updates the metadata of every such patch.
    Starts from the bottom ...
    See also:
      https://rasterio.readthedocs.io/en/latest/quickstart.html#spatial-indexing
    TO DO:
      - add overlapping of the images ?
    :param image: object / path of the image to cut
    :param size: int or tuple, size of the cutted image
    :param debug: prints several parameters while processing the code, debugging mode
    :param data_path: path to which we save obtained rasterized images, if None the images are not saved
    :param name: name of the files
    :return cutted_images: list of images with metadata. Each image is a numpy.array of shape 3 x size x size
    """
    # === process size ===
    if isinstance(size, tuple):
        size_x, size_y = size
    elif isinstance(size, int):
        size_x, size_y = size, size
    else:
        raise TypeError("Size should be int or tuple")

    # === get image shape ===
    width, height = image.width, image.height
    print(width,height,"Here")
    cutted_images = []
    epsg_code = int(image.crs.data['init'][5:])
    # === create bounding box ===
    idx = 0
    for x in range(size_x, width + 1, size_x):
        for y in range(size_y, height + 1, size_y):
            # get the coordinates of bbox in cartesian coordinates
            minx, maxx = x - size_x , x - 1
            miny, maxy = y - size_y, y - 1
            # get the coordinates of bbox in crs coordinates
            minx, miny = image.xy(minx, miny)
            maxx, maxy = image.xy(maxx, maxy)
            bbox = box(minx, miny, maxx, maxy)
            if debug:
                print("Coordinates", minx, maxx, miny, maxy, width, x, y)
                print(bbox)

            # === crop the image ===
            """epsg_code = int(image.crs.data['init'][5:])
            transformed_coords = transform_to_box(bbox, epsg_code)"""
            geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=CRS(epsg_code))
            transformed_coords = getFeatures(geo)
            if debug:
                print("EPSG_code", epsg_code)
                print("Transformed coords", transformed_coords)
            out_img, out_transform = mask(image, transformed_coords, all_touched=True, crop=True)
            # add updated image
            cutted_images.append(out_img)
            # === Save the image ===
            if data_path is not None:
                # === Update the metadata ===
                # Copy the metadata
                out_meta = image.meta.copy()
                if debug:
                    print("\n", out_meta)

                out_meta.update({"driver": "GTiff",
                                 "height": out_img.shape[1],
                                 "width": out_img.shape[2],
                                 "transform": out_transform,
                                 "crs": CRS(epsg_code)})

                # === Save as a raster file ===
                out_tif_name = f"_{idx}.tif" if name is None else name + f"_{idx}.tif"
                out_tif_name = os.path.join(data_path, out_tif_name)
                with rasterio.open(out_tif_name, "w", **out_meta) as dest:
                    dest.write(out_img)
                    if idx== 0:
                        print(f"File {out_tif_name} successfully created!")

                idx += 1

    return cutted_images
