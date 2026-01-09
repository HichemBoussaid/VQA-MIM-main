"""Functions to work with directories"""
import os
import shutil

import tqdm


def ListAllSubDir(root):
    dirs = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            dirs.append(os.path.join(path, name))
    return dirs


"""def getListOfFiles(dirName):
  
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles"""
def getListOfFiles(dirName):
    """For the given path, get the List of all files in the directory tree"""
    # create a list of file
    allFiles = list()
    # Iterate over all the entries
    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(dirName)):
        # Create full path
        for filename in filenames:
            fullPath = os.path.join(dirpath, filename)
            allFiles.append(fullPath)

    return allFiles

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
    # create target directory if necessary
    try:
        os.mkdir(target_dir)
        print("Target directory sucessfuly created.")
    except:
        print("Target directory already exists.")

    # collect BDOrtho files
    for idx, path in enumerate(new_ign_coords_df["data_path"]):
        new_folder_path = target_dir + "/" + str(idx)
        os.mkdir(new_folder_path)
        file_name = path.split("/")[-1]
        shutil.move(path, new_folder_path + "/" + file_name)

    # collect IGN files
    all_files = [file for file in os.listdir(cur_dir) if "Sentinel-2" in file]
    ign_files = {
        int(file.split("_")[2]): file for file in all_files
    }
    for idx, path in ign_files.items():
        new_folder_path = target_dir + "/" + str(idx)
        file_name = path.split("/")[-1]
        shutil.move(path, new_folder_path + "/" + file_name)
