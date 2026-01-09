import os
import geopandas as gpd
from dir_utils import getListOfFiles
from utils import print_color, add_index_column
from dataset_creation import create_df


def printc(dir, read=True):
    """Auxiliary function for the read_data function"""
    if read:
        print(print_color.BOLD + dir + print_color.END +" data has been sucessfully read.")
    else:
        print("We do not read " + print_color.BOLD + dir + print_color.END + \
              " in the current session.\nIf you wish to read it, please, change the parameter of " + \
              print_color.BOLD + f"to_read_data[{dir}]" + print_color.END + " to True.")


def read_data(directories, to_read_data: dict, limit=None, out_epsg=2154) -> dict:
    """
    Reads the data coming from several different sources. Currently the sources include:
    - Sentinel-2 images ("Sentinel-2"
    - BDOrtho images ("IGN")
    - Open Street Map data ("OSM")
    TBA:
    - Sentinel-1 images
    - BDTopo images
    :param directories: paths to files to read
    :param to_read_data: specifies which data to read
    :param limit: limits the number of files to read
    :param out_epsg: EPSG (coordinate referance system) of the target file.
        Default: EPSG:2154. RGF93 / Lambert-93: In France this is commonly used format.
    :return: directories with the processed data. In particular:
    *Open Street Map data.*
        - Returns ```osm_coords_df``` with the coordinates of *Open Street Map data* (objects of interest)

    *IGN.*
        - Returns ```ign_jp2_files``` with the file names of aerial images data obtained from *IGN*, stored in .jp2 format.
        - Returns ```ign_coords_df``` with their coordinates obtained from *IGN*.

    *Sentinel-2*
        - Returns ```sentinel_jp2_files``` has the file names of the Sentinel-2 images.
        - Returns ```sentinel_coords_df``` with their coordinates obtained from *IGN*.

    _jp2_files include the file names of the images
    _coords_df is a dataframe with three columns:
        - data_path - specifies the path to the data
        - geometry - coordinates of the images corners
        - index - index of the file in the dataframe
    """
    output_data = {}
    for name, path_dir in directories.items():
        if name == "OSM":
            if to_read_data["OSM"]:
                # Set filepath
                FILEPATH = os.path.join(path_dir, 'gis_osm_buildings_a_free_1.shp')

                # Read file using gpd.read_file()
                osm_buildings_data = gpd.read_file(FILEPATH)

                # As the data from GeoFabrik as it is stored in the WGS-84 format.
                # We next map the data to Lambert-93 CRS system.
                osm_coords_df = osm_buildings_data[:limit].to_crs(
                    epsg=out_epsg)  # it takes around 7 minutes to run on Colab for the whole dataset
                output_data["OSM"] = add_index_column(osm_coords_df)

                printc("Open Street Map Data")
            else:
                printc("Open Street Map Data", False)
        elif name == "BDT":
            if to_read_data["BDT"]:
                output_data["BDT"]= {}
                bdTopoDirs = [ f.name for f in os.scandir(path_dir) if f.is_dir() ]
                for bdTopoDir in bdTopoDirs:
                  # Set filepath
                    dir_path = os.path.join(path_dir, bdTopoDir)
                    hydro_shp_files = [filename for filename in getListOfFiles(dir_path)
                                    if filename.endswith('.shp') and (not (filename.split("\\")[-1].startswith("."))) ]
                    output_data["BDT"][bdTopoDir] = {}
                    for FILEPATH in hydro_shp_files:
                      #FILEPATH = os.path.join(path_dir, 'HYDROGRAPHIE')
                      key = FILEPATH.split("\\")[-1].split(".")[-2]
                      # Read file using gpd.read_file()
                      bdt_buildings_data = gpd.read_file(FILEPATH)
                      osm_coords_df = bdt_buildings_data[:limit].to_crs(
                          epsg=out_epsg)  # it takes around 7 minutes to run on Colab for the whole dataset
                      output_data["BDT"][bdTopoDir][key] = add_index_column(osm_coords_df)


                printc("BD TOPO Data")
            else:
                printc("BD TOPO Data", False)
                
        elif name == "IGN":
            if to_read_data["IGN"]:
                # === Read IGN images ===
                files_path = path_dir["files"]
                coords_path = path_dir["coords"]

                # get the IGN filenames
                ign_jp2_files = [filename for filename in getListOfFiles(files_path)
                                 if filename.endswith('.jp2')]

                # === Read coordinates of the IGN images ===
                # set filepath
                FILEPATH = os.path.join(coords_path, "dalles.shp")

                # Read the file using gpd.read_file()
                ign_coords_df = gpd.read_file(os.path.join(files_path, FILEPATH))
                ign_coords_df["NOM"] = ign_coords_df["NOM"].apply(
                    lambda name: files_path + name[1:])
                ign_coords_df = ign_coords_df.rename(columns={"NOM": "data_path"})
                ign_coords_df = add_index_column(ign_coords_df)
                ign_coords_df = ign_coords_df[["index", "data_path", "geometry"]]

                # save the data
                output_data["IGN"] = {
                    "Filenames": ign_jp2_files,
                    "Coordinates": ign_coords_df
                }

                printc("IGN BDOrtho")
            else:
                printc("IGN BDOrtho", False)

        elif name == "Sentinel-2":
            if to_read_data["Sentinel-2"]:
                print(getListOfFiles(path_dir))
                # get the files by walking through the directory
                sentinel_jp2_files = [filename for filename in getListOfFiles(path_dir)
                                      if filename.endswith('jp2')
                                      and 'GRANULE' in filename
                                      and 'IMG_DATA' in filename
                                     and ('R20m' in filename
                                     or 'R10m' in filename)]
                print(sentinel_jp2_files)
                # create dataframe with the coordinates stored
                sentinel_coords_df = create_df(sentinel_jp2_files)
                sentinel_coords_df = add_index_column(sentinel_coords_df)
                sentinel_coords_df = sentinel_coords_df[["index", "data_path", "geometry"]]

                output_data["Sentinel-2"] = {
                    "Filenames": sentinel_jp2_files,
                    "Coordinates": sentinel_coords_df
                }

                printc("Sentinel-2")
            else:
                printc("Sentinel-2", False)

    return output_data
