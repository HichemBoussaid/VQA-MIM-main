import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
import torch
from shapely.geometry import Polygon


def convert_to_tensor(img_path):
    # to do: check data type
    with rasterio.open(img_path) as rast_img:
        tensor_img = torch.tensor(rast_img.read().astype('float32')) / 255
    return tensor_img


def plot_img(tensor_img, title=None, vbounds=[200, 400], hbounds=[0, 200],
             cmap='gray'):
    """Input image has shape C x N x M, where C -- number of channels"""
    f = plt.figure(figsize=(8, 8))
    if tensor_img.shape[0] == 1:
        plt.imshow(tensor_img[:, vbounds[0]:vbounds[1], hbounds[0]:hbounds[1]].permute(1, 2, 0).squeeze(-1),
                   cmap=cmap)
    else:
        plt.imshow(tensor_img[:, vbounds[0]:vbounds[1], hbounds[0]:hbounds[1]].permute(1, 2, 0))
    if title:
        plt.title(title, size=10)
    plt.tight_layout()


class print_color:
    """Useful class for printing.
    Example: print(print_color.BOLD + 'Hello World !' + print_color.END)
    Source: https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_coordinates(data_path: str):
    """
    The function takes the data path and returns the bounding box coordinates of
    the object of interest.
    ---------------------------------------
    :param data_path: data path of a single image
    :return: - dictionary with the GeographicBoundingBox coordinates
              - Polygon obtained with these coordinates
    """
    # Read the data
    dataset = rasterio.open(data_path)

    # Getting coordinates stored as dict
    coords = {}
    index = ["left", "bottom", "right", "top"]
    for i, value in enumerate(dataset.bounds):
        coords[index[i]] = value

    # Create Polygon
    bbox_coords = [(coords["left"], coords["bottom"]),
                   (coords["right"], coords["bottom"]),
                   (coords["right"], coords["top"]),
                   (coords["left"], coords["top"])]
    bbox_polygon = Polygon(bbox_coords)

    # Coordinate reference system
    crs = dataset.crs.data["init"]

    return coords, bbox_polygon, crs


def add_index_column(df):
    df["index"] = range(0, len(df))
    return df


def find_size(length: int):
    """
    Auxiliary function for plot_grid to find the shape of the grid
    :param length: length of the grid
    :return: shape of the grid
    """
    if length == 1:
        return 1, 1
    if length in [3, 5, 7]:
        return 1, length
    if length in [2, 4, 6, 8, 10, 12, 14]:
        return 2, length // 2
    c = 0
    while c < 5:
        for div in range(7, 2, -1):
            if length % div == 0:
                return length // div, div
        length += 1
        c += 1
    raise ValueError("Algorithm does not converge")


def plot_grid(images: list, size="auto", title=None, scale=None):
    """
    TO DO:
    - add support of 1-channel images.

    Plots the images as a N x M grid. If size not specified, defines it automatically
    :param images: list of images to be plotted
    :param size: if size of the grid is not specified define the size
    :return: None
    """
    if size == "auto":
        size = find_size(len(images))

    n, m = size

    if not scale:
        SCALE_DCT = {3: 6, 4: 5, 5: 4, 6: 4}
        scale = SCALE_DCT[n]

    fig, axs = plt.subplots(*size, figsize=(scale * m, scale * n))

    for idx, img in enumerate(images):
        axs[idx // m][idx % m].imshow(np.transpose(img, [1, 2, 0]))
        axs[idx // m][idx % m].axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.plot()
