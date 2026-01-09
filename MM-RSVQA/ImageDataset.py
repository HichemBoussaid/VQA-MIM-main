import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
import rasterio
import rasterio.mask
from tqdm import tqdm
from rasterio.windows import Window
from sar_display import *
from transformers import DistilBertTokenizer
from scipy import interpolate
import gc
import json
import torch
import tifffile as tiff
import os



def BDOrthopatchfounder(pathBdO, nameBdO, idbdO, plot=False):
    dept, id_ = idbdO.split("-")
    path = pathBdO + "/" + dept + "/" + dept + "/" + id_ + '/' + nameBdO
    with rasterio.open(path) as img:
        w1 = img.read(1)
        w2 = img.read(2)
        w3 = img.read(3)
        imgBDO = np.dstack((w1, w2, w3))
    if plot == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(imgBDO)
        plt.title('BdOrtho')
        plt.show()
    return (imgBDO)


def Sentinel1patchfoundergetit(pathS1, namevh, namevv, an, rn, orbit_directionswath, look_directionswath,
                               height_size=500, threshold=233, plot=False, sarch=2,
                               dept=None):  # orbit_direction,look_direction,
    with rasterio.open(pathS1 + dept + "/S1S2/" + namevh) as src:
        left = (rn) - height_size // 2
        top = (an) - height_size // 2
        wvh = src.read(1, window=Window(left, top, height_size, height_size))
    with rasterio.open(pathS1 + dept + "/S1S2/" + namevv) as src:
        # left = int(rn) - height_size // 2
        # top = int(an) - height_size // 2
        wvv = src.read(1, window=Window(left, top, height_size, height_size))
    with rasterio.open(pathS1 + dept + "/S1S2/" + 'ratio' + namevv) as src:
        # left = int(rn) - height_size // 2
        # top = int(an) - height_size // 2
        ratio = src.read(1, window=Window(left, top, height_size, height_size))
    if threshold != 9999:
        wvh = apply_threshold(wvh, threshold)
        wvv = apply_threshold(wvv, threshold)
    rotatedvh = geoTransform(wvh, orbit_directionswath, look_directionswath)
    rotatedvv = geoTransform(wvv, orbit_directionswath, look_directionswath)
    rotatedratio = geoTransform(ratio, orbit_directionswath, look_directionswath)
    if plot == True:
        if sarch == 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(rotatedvv, cmap='gray')
            plt.title('Sentinel 1 VV')
            plt.show()
        elif sarch == 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(rotatedvh, cmap='gray')
            plt.title('Sentinel 1 VH')
            plt.show()
        else:
            image_3ch = np.zeros((rotatedvv.shape[0], rotatedvv.shape[1], 3), dtype=np.uint8)
            image_3ch[:, :, 0] = rotatedvv  # Canale rosso: im1
            image_3ch[:, :, 1] = rotatedvh  # Canale verde: im2
            image_3ch[:, :, 2] = rotatedratio
            plt.figure(figsize=(10, 10))
            plt.imshow(image_3ch, cmap='gray')
            plt.title('Sentinel 1')
            plt.show()
    return (rotatedvh, rotatedvv, rotatedratio)


def upsample_patch(patch):
    original_height, original_width = patch.shape
    target_height = original_height * 2
    target_width = original_width * 2
    # Create grid coordinates for original and target sizes
    x = np.linspace(0, original_width - 1, original_width)
    y = np.linspace(0, original_height - 1, original_height)
    xnew = np.linspace(0, original_width - 1, target_width)
    ynew = np.linspace(0, original_height - 1, target_height)

    # Create interpolation function
    f = interpolate.interp2d(x, y, patch, kind='linear')

    # Upsample the patch
    upsampled_patch = f(xnew, ynew)

    return upsampled_patch


def Sentinel2patchfoundergetit(pathS2, nameS2, pixx, pixy, height_size=500, plot=False, dept=None):
    with rasterio.open(pathS2 + dept + "/S1S2/" + nameS2) as dataset:
        left = int(pixy - height_size // 2)
        top = int(pixx - height_size // 2)
        w1 = dataset.read(1, window=Window(left, top, height_size, height_size))
        w2 = dataset.read(2, window=Window(left, top, height_size, height_size))
        w3 = dataset.read(3, window=Window(left, top, height_size, height_size))
        patch = np.dstack((w1, w2, w3))
    if plot == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(patch)
        plt.title(nameS2[23:-4])
        plt.show()
    return (patch)


def Sentinel210patchfoundergetit(pathS2, nameS2, pixx, pixy, height_size=500, plot=False, dept=None):
    with rasterio.open(pathS2 + dept + "/S1S2/" + nameS2) as dataset:
        # For images with one channel, read just that channel
        left = int(pixy - height_size // 2)
        top = int(pixx - height_size // 2)
        patch = dataset.read(1, window=Window(left, top, height_size, height_size))
        # patch = np.expand_dims(w1, axis=-1)

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(patch)
        plt.title(nameS2[23:-4])
        plt.show()

    return patch


def Sentinel220patchfoundergetit(pathS2, nameS2, pixx, pixy, height_size=500, plot=False, dept=None):
    with rasterio.open(pathS2 + dept + "/S1S2/" + nameS2) as dataset:
        # For images with one channel, read just that channel
        left = int(pixy // 2 - height_size // 4)
        top = int(pixx // 2 - height_size // 4)
        patch = dataset.read(1, window=Window(left, top, height_size // 2, height_size // 2))
        patch = upsample_patch(patch)
        # patch = np.expand_dims(w1, axis=-1)
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(patch)
        plt.title(nameS2[23:-4])
        plt.show()

    return patch


def apply_threshold(image_data, threshold):
    # Calcola il modulo dell'immagine
    modulus = np.abs(image_data)

    # Applica la logica di thresholding
    thresholded_modulus = np.where(modulus > threshold, threshold, modulus)

    return thresholded_modulus


def process_tensor(tensor):
    t1, t2, t3 = torch.split(tensor, [683, 683, 682], dim=1)
    padding_sizes = [85, 85, 86]
    t1 = torch.cat((t1, torch.zeros_like(t1[:, :padding_sizes[0]])), dim=1)
    # print(t1)
    t2 = torch.cat((t2, torch.zeros_like(t2[:, :padding_sizes[1]])), dim=1)
    t3 = torch.cat((t3, torch.zeros_like(t3[:, :padding_sizes[2]])), dim=1)
    divided_tensor = torch.stack((t1, t2, t3), dim=1)
    return divided_tensor


import matplotlib.pyplot as plt
import numpy as np


def save_images(imgBDO=None, imgS2=None, imgS1=None, save_path="output/", batch_index="0_0", save_as_tif=False):
    # Ensure the save path exists
    # os.makedirs(os.path.join(save_path,batch_index), exist_ok=True)
    imgBDO = None
    # Save BDO image
    if imgBDO is not None:
        tiff.imwrite(os.path.join(save_path, f"BDO_before.tif"), imgBDO)

        """imgBDO_pil = Image.fromarray(imgBDO)

        imgBDO_pil.save(os.path.join(save_path, 'BDO_before.png'))"""

    # Save S2 image (Assume saving only the first 3 channels for RGB representation)
    if imgS2 is not None:
        tiff.imwrite(os.path.join(save_path, f"S2.tif"), imgS2)

    if imgS1 is not None:
        tiff.imwrite(os.path.join(save_path, f"S1.tif"), imgS1)


MAX_ANSWERS = 4332
LEN_QUESTION = 20

encoder_answers = None
print("Loading DistilBert...")
model_path = "distilbert/distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)  # TOKENIZERS_PARALLELISM=False
print("Done.", flush=True)


# Dataset construction working, on getitem
class ImageDataset(Dataset):
    def __init__(self, jsonimg, encoder_answers, jsonquestions, jsonanswers, pathBdO, pathS2, pathS1,
                 patch_sizeBDO=1000,
                 patch_sizeS2=200, patch_sizeS1=300, train=True, ratio_images_to_use=1, transformS2=None,
                 transform=None,
                 number_outputs=1000, selected_answers=None,
                 tokenizer=tokenizer, activate_bdo=True, activate_s1=True,
                 activate_s2=True, part=-1):  # encoder_questions, encoder_answers,
        self.jsonimg = jsonimg
        self.encoder_answers = encoder_answers
        self.jsonquestions = jsonquestions
        self.jsonanswers = jsonanswers
        self.pathBdO = pathBdO
        self.patch_sizeBDO = patch_sizeBDO
        self.pathS2 = pathS2
        self.pathS1 = pathS1
        self.patch_sizeS1 = patch_sizeS1
        self.patch_sizeS2 = patch_sizeS2
        self.train = train
        self.ratio_images_to_use = ratio_images_to_use
        self.transformS2 = transformS2
        self.transform = transform
        if not (activate_bdo or activate_s1 or activate_s2):
            raise ValueError("You need at least one image modality")
        self.activate_bdo = activate_bdo
        self.activate_s1 = activate_s1
        self.activate_s2 = activate_s2
        self.save = True
        self.part = part

        LEN_QC = 128  # no idea
        with open(jsonquestions) as file1:
            questionsJSON = json.load(file1)
            # questions=self.questions['questions']
        with open(jsonanswers) as file2:
            answersJSON = json.load(file2)
            # answers=self.answers['answers']
        with open(jsonimg) as f:
            # Load the JSON data
            imagesJSON = json.load(f)

        images_idx = []
        images = []
        nameBdO = []
        nameS2 = []
        pixx = []
        pixy = []
        nameS1vh = []
        nameS1vv = []
        rn = []
        an = []
        orbit_directionswath = []
        look_directionswath = []

        cont = 0
        for idx, img in enumerate(imagesJSON['images']):
            if img['active']:
                images_idx.append(idx)
                images.append(img['id'])
                nameBdO.append(img['BdOrthoname'])
                nameS2.append(img['S2name'][:-7])
                pixx.append(img['S2centerpos'][0])
                pixy.append(img['S2centerpos'][1])
                nameS1vh.append(img['S1namevh'])
                nameS1vv.append(img['S1namevv'])
                rn.append(img['S1centerpos'][0])
                an.append(img['S1centerpos'][1])
                orbit_directionswath.append(img['orbit_directionswath'])
                look_directionswath.append(img['look_directionswath'])
        if self.part == -1:
            self.images_idx = images_idx[:int(len(images_idx) * ratio_images_to_use)]
            images = images[:int(len(images) * ratio_images_to_use)]
            self.nameBdO = nameBdO[:int(len(nameBdO) * ratio_images_to_use)]

            self.nameS2 = nameS2[:int(len(nameS2) * ratio_images_to_use)]

            self.pixx = pixx[:int(len(pixx) * ratio_images_to_use)]

            self.pixy = pixy[:int(len(pixy) * ratio_images_to_use)]
            self.nameS1vh = nameS1vh[:int(len(nameS1vh) * ratio_images_to_use)]
            self.nameS1vv = nameS1vv[:int(len(nameS1vv) * ratio_images_to_use)]
            self.rn = rn[:int(len(rn) * ratio_images_to_use)]
            self.an = an[:int(len(an) * ratio_images_to_use)]
            self.orbit_directionswath = orbit_directionswath[:int(len(orbit_directionswath) * ratio_images_to_use)]
            self.look_directionswath = look_directionswath[:int(len(look_directionswath) * ratio_images_to_use)]
            self.img_ids = images

        else:
            divide_by = 2
            self.images_idx = images_idx[
                              int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            images = images[int(len(images) / divide_by * self.part):int(len(images) / divide_by * (self.part + 1))]
            self.nameBdO = nameBdO[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]

            self.nameS2 = nameS2[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]

            self.pixx = pixx[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]

            self.pixy = pixy[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.nameS1vh = nameS1vh[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.nameS1vv = nameS1vv[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.rn = rn[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.an = an[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.orbit_directionswath = orbit_directionswath[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.look_directionswath = look_directionswath[int(len(images_idx) / divide_by * self.part):int(len(images_idx) / divide_by * (self.part + 1))]
            self.img_ids = images

        del images_idx

        del nameBdO, nameS2, pixy, pixx, nameS1vh, nameS1vv, rn, an, orbit_directionswath, look_directionswath
        gc.collect()

        self.images_questions_answers = []
        self.processed_paths = []

        if train:
            self.freq_dict = {}
            for i, image in enumerate(tqdm(images)):
                for questionid in imagesJSON['images'][self.images_idx[i]]['questions_ids']:
                    question = questionsJSON['questions'][questionid]
                    if type(answersJSON['answers'][questionid]['answer']) == list:
                        answer_str = (answersJSON['answers'][questionid]['answer'][0])
                    else:
                        answer_str = answersJSON['answers'][questionid]['answer']

                    if answer_str not in self.freq_dict:
                        self.freq_dict[answer_str] = 1
                    else:
                        self.freq_dict[answer_str] += 1

            self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
            self.selected_answers = []
            self.non_selected_answers = []
            coverage = 0
            total_answers = 0

            for i, key in enumerate(self.freq_dict):
                if i < number_outputs:
                    self.selected_answers.append(key[0])
                    coverage += key[1]
                else:
                    self.non_selected_answers.append(key[0])
                total_answers += key[1]

            print(
                f"The {number_outputs} most frequent answers cover {coverage / total_answers * 100}% of the total answers.")
        else:
            self.selected_answers = selected_answers

        for i, image in enumerate(tqdm(images)):
            for questionid in imagesJSON['images'][self.images_idx[i]]['questions_ids']:
                question = questionsJSON['questions'][questionid]
                question_str = question["question"]
                type_str = question["type"]
                if type(answersJSON['answers'][questionid]['answer']) == list:
                    answer_str = (answersJSON['answers'][questionid]['answer'][0])
                else:
                    answer_str = answersJSON['answers'][questionid]['answer']
                question_tokens = tokenizer(question_str, return_tensors="pt", padding='max_length', max_length=26,
                                            add_special_tokens=True)  # 20 is not enought #1,25
                '''
                for key, tensor in question_tokens.items():
                    print(f"Dimensions of {key}: {tensor.size()}")
                '''
                # Print the size of the tenso
                question_tokens['input_ids'] = question_tokens['input_ids'].squeeze()  # 25
                question_tokens['attention_mask'] = question_tokens['attention_mask'].squeeze()
                if self.selected_answers == None:
                    self.images_questions_answers.append(
                        [question_tokens, self.encoder_answers.encode(answer_str), i, image, type_str])
                else:
                    if answer_str in self.selected_answers:
                        self.images_questions_answers.append(
                            [question_tokens, self.selected_answers.index(answer_str), i, image, type_str])
        del questionsJSON, imagesJSON, answersJSON
        gc.collect()

    def __len__(self):
        return len(self.images_questions_answers)

    def __getitem__(self, index):
        question = self.images_questions_answers[index]
        # imgBDO = BDOrthopatchfounder(self.pathBdO, question[3], plot=False)
        band_names = [self.nameS2[question[2]] + 'TCI.jp2', self.nameS2[question[2]] + 'B05.jp2',
                      self.nameS2[question[2]] + 'B06.jp2', self.nameS2[question[2]] + 'B07.jp2',
                      self.nameS2[question[2]] + 'B08.jp2', self.nameS2[question[2]] + 'B8A.jp2',
                      self.nameS2[question[2]] + 'B11.jp2', self.nameS2[question[2]] + 'B12.jp2']
        # Load images and other data based on the given index
        dept, id_ = question[3].split("-")
        path = self.pathBdO + "/" + dept + "/" + dept + "/" + id_ + '/'

        if self.activate_bdo:
            imgBDO = BDOrthopatchfounder(self.pathBdO, self.nameBdO[question[2]], question[3], plot=False)

        if self.activate_s2:
            if os.path.isfile(path + "S2.tif"):
                try:
                    imgS2 = tiff.imread(path + "S2.tif")
                except  tiff.TiffFileError:
                    imgS2_TCI = Sentinel2patchfoundergetit(self.pathS2, band_names[0], self.pixx[question[2]],
                                                           self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                           dept=dept)
                    imgS2_B05 = Sentinel220patchfoundergetit(self.pathS2, band_names[1], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B06 = Sentinel220patchfoundergetit(self.pathS2, band_names[2], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B07 = Sentinel220patchfoundergetit(self.pathS2, band_names[3], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B08 = Sentinel210patchfoundergetit(self.pathS2, band_names[4], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B8A = Sentinel220patchfoundergetit(self.pathS2, band_names[5], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B11 = Sentinel220patchfoundergetit(self.pathS2, band_names[6], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2_B12 = Sentinel220patchfoundergetit(self.pathS2, band_names[7], self.pixx[question[2]],
                                                             self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                             dept=dept)
                    imgS2 = np.dstack(
                        (imgS2_TCI, imgS2_B05, imgS2_B06, imgS2_B07, imgS2_B08, imgS2_B8A, imgS2_B11, imgS2_B12))
            else:
                imgS2_TCI = Sentinel2patchfoundergetit(self.pathS2, band_names[0], self.pixx[question[2]],
                                                       self.pixy[question[2]], self.patch_sizeS2, plot=False, dept=dept)
                imgS2_B05 = Sentinel220patchfoundergetit(self.pathS2, band_names[1], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B06 = Sentinel220patchfoundergetit(self.pathS2, band_names[2], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B07 = Sentinel220patchfoundergetit(self.pathS2, band_names[3], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B08 = Sentinel210patchfoundergetit(self.pathS2, band_names[4], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B8A = Sentinel220patchfoundergetit(self.pathS2, band_names[5], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B11 = Sentinel220patchfoundergetit(self.pathS2, band_names[6], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2_B12 = Sentinel220patchfoundergetit(self.pathS2, band_names[7], self.pixx[question[2]],
                                                         self.pixy[question[2]], self.patch_sizeS2, plot=False,
                                                         dept=dept)
                imgS2 = np.dstack(
                    (imgS2_TCI, imgS2_B05, imgS2_B06, imgS2_B07, imgS2_B08, imgS2_B8A, imgS2_B11, imgS2_B12))

        if self.activate_s1:
            if os.path.isfile(path + "S1.tif"):
                try:
                    imgS1 = tiff.imread(path + "S1.tif")
                except tiff.TiffFileError:
                    print(f"{path + 'S1.tif'} is corrupted or not a valid TIFF file.")
                    imgS1vh, imgS1vv, imgS1ratio = Sentinel1patchfoundergetit(self.pathS1, self.nameS1vh[question[2]],
                                                                              self.nameS1vv[question[2]],
                                                                              self.an[question[2]],
                                                                              self.rn[question[2]],
                                                                              self.orbit_directionswath[question[2]],
                                                                              self.look_directionswath[question[2]],
                                                                              self.patch_sizeS1, threshold=9999,
                                                                              plot=False,
                                                                              sarch=2, dept=dept)
                    imgS1 = np.dstack((imgS1vv, imgS1vh, imgS1ratio)).astype(np.float32)
            else:
                imgS1vh, imgS1vv, imgS1ratio = Sentinel1patchfoundergetit(self.pathS1, self.nameS1vh[question[2]],
                                                                          self.nameS1vv[question[2]],
                                                                          self.an[question[2]],
                                                                          self.rn[question[2]],
                                                                          self.orbit_directionswath[question[2]],
                                                                          self.look_directionswath[question[2]],
                                                                          self.patch_sizeS1, threshold=9999,
                                                                          plot=False,
                                                                          sarch=2, dept=dept)
                imgS1 = np.dstack((imgS1vv, imgS1vh, imgS1ratio)).astype(np.float32)

        if path not in self.processed_paths and self.activate_s1 and self.activate_s2 and self.activate_bdo and self.save:
            save_images(imgBDO, imgS2, imgS1, save_path=path,
                        batch_index=f"{index}", save_as_tif=False)
            self.processed_paths.append(path)

        if self.transform:
            if self.activate_bdo:
                imgBDO = self.transform(imgBDO.copy())  # imageT here i have error referenced before assignment
            if self.activate_s2:
                imgS2 = self.transformS2(imgS2.copy())
            if self.activate_s1:
                imgS1 = self.transform(imgS1.copy())
        data = (
            question[0],
            question[1],
            *([imgBDO] if self.activate_bdo else []),
            *([imgS2] if self.activate_s2 else []),
            *([imgS1] if self.activate_s1 else []),
            question[4]
        )

        return data, index


del tokenizer
