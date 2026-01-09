import csv
import os
import time

import pandas as pd
from tqdm import tqdm
import json
import ConstructImageQuestion
from pathlib import Path
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')

from collections import defaultdict


def get_stats(questions_dict, answers_dict):
    # Initialize counters
    question_type_counter = defaultdict(int)
    answer_type_counter = defaultdict(int)
    questions_per_image_counter = defaultdict(int)

    # Count question types
    for question in questions_dict:
        question_type_counter[question['type']] += 1
        questions_per_image_counter[question['patch_id']] += 1

    # Count answer types
    for answer in answers_dict:
        answer_type_counter[answer['type']] += 1

    # Print stats
    print("Question Type Statistics:")
    for question_type, count in question_type_counter.items():
        print(f"{question_type}: {count}")

    print("\nAnswer Type Statistics:")
    for answer_type, count in answer_type_counter.items():
        print(f"{answer_type}: {count}")

    print("\nQuestions per Image Statistics:")
    for patch_id, count in questions_per_image_counter.items():
        print(f"{patch_id}: {count}")


# Call the function with your data

def check_all_zero(d):
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if v != 0:
                    return False
        else:
            if value != 0:
                return False
    return True


DATA_DIR = 'D:\\data\\BDOrtho+Sentinel2'
urban_shp = gpd.read_file("D:\\data\\SIG\\ZAU\\com_uu2020_2023\\com_uu2020_2023.shp").to_crs(epsg=2154)
mountains_shp = gpd.read_file("D:\\data\\SIG\\Mountains\\m_massifs_v1.shp").to_crs(epsg=2154)

urban_csv = []
with open("D:\\data\\SIG\\ZAU\\UU2020_au_01-01-2023.csv", encoding="latin1") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        row1 = row[0].split(";")
        urban_csv.append((row1[0], row1[-3]))

for dept in ["75"]:
    question_by_patch = {}
    print(dept)
    clc_shp = gpd.read_file(f'D:\\data\\SIG\\CLC_PNE_RG\\{dept}\\clc.shp').to_crs(
        epsg=2154)
    version = "test_version8\\" + dept
    floods_path = "D:\\data\\SIG\\Floods\\" + dept
    files_names = os.listdir(floods_path)
    shp_names = [x for x in files_names if "inondable" in x and ".shp" in x]
    tri_fai = gpd.GeoDataFrame(pd.concat(
        [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in [x for x in shp_names if "fai" in x]],
        ignore_index=True))
    # tri_fai = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "fai" in x][0])).to_crs(epsg=2154)
    tri_moy = gpd.GeoDataFrame(pd.concat(
        [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in [x for x in shp_names if "moy" in x]],
        ignore_index=True))
    # tri_moy = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "moy" in x][0])).to_crs(epsg=2154)
    tri_for = gpd.GeoDataFrame(pd.concat(
        [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in [x for x in shp_names if "for" in x]],
        ignore_index=True))
    # tri_for = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "for" in x][0])).to_crs(epsg=2154)
    target_dir = os.path.join(DATA_DIR, version)

    questions_dict = []
    answers_dict = []
    total_questions = 0
    with open("BD_TOPO_Dict.json", encoding='utf-8') as fp:
        bd_topo_dict = json.load(fp)
    if not os.path.exists(target_dir + "/Q&A"):
        os.mkdir(target_dir + "/Q&A")
    listPatches = [f for f in os.listdir(target_dir) if os.path.isdir(target_dir + "/" + f) and f != "Q&A"]
    i = 0
    stats = {}
    big_GPD = {}
    for folder, v in bd_topo_dict.items():
        if folder not in big_GPD:
            big_GPD[folder] = {}
        for shapefile, value in v.items():
            shp_path = Path(os.path.join("D:\\data\\BD_Topo", dept, "BDTopo_new", folder, shapefile))
            if shp_path.exists():
                gdf = gpd.read_file(shp_path)
                big_GPD[folder][shapefile] = gdf

    number_of_questions = int(0.1 * len(listPatches))

    target_answers = {"less_more_equal": {"yes": 5 * number_of_questions, "no": 5 * number_of_questions},
                      "presence": {"yes": 5 * number_of_questions, "no": 5 * number_of_questions},
                      "pres_flooding": {"yes": 5 * number_of_questions, "no": 5 * number_of_questions},
                      "pres_mountains": {"yes": 5 * number_of_questions, "no": 5 * number_of_questions},
                      "lvl_flooding": {"high": int(3.33 * number_of_questions),
                                       "medium": int(3.33 * number_of_questions),
                                       "low": int(3.33 * number_of_questions)},
                      "urban": {"Outside Urban Unit": int(2.5 * number_of_questions),
                                "City Center": int(2.5 * number_of_questions),
                                "Suburbs": int(2.5 * number_of_questions),
                                "Isolated City": int(2.5 * number_of_questions)},
                      "type_flooding": {"River overflows": int(2.5 * number_of_questions),
                                        "Runoff": int(2.5 * number_of_questions),
                                        "Sea flooding": int(2.5 * number_of_questions),
                                        "Groundwater overflows": int(2.5 * number_of_questions)},
                      "clc": {
                          'Continuous urban fabric': number_of_questions,
                          'Discontinuous urban fabric': number_of_questions,
                          'Industrial or commercial units and public facilities': number_of_questions,
                          'Road and rail networks and associated land': number_of_questions,
                          'Port areas': number_of_questions,
                          'Airports': number_of_questions,
                          'Mineral extraction sites': number_of_questions,
                          'Dump sites': number_of_questions,
                          'Construction sites': number_of_questions,
                          'Green urban areas': number_of_questions,
                          'Sport and leisure facilities': number_of_questions,
                          'Non-irrigated arable land': number_of_questions,
                          'Permanently irrigated land': number_of_questions,
                          'Rice fields': number_of_questions,
                          'Vineyards': number_of_questions,
                          'Fruit trees and berry plantations': number_of_questions,
                          'Olive groves': number_of_questions,
                          'Pastures, meadows and other permanent grasslands under agricultural use': number_of_questions,
                          'Annual crops associated with permanent crops': number_of_questions,
                          'Complex cultivation patterns': number_of_questions,
                          'Land principally occupied by agriculture, with significant areas of natural vegetation': number_of_questions,
                          'Agro-forestry areas': number_of_questions,
                          'Broad-leaved forest': number_of_questions,
                          'Coniferous forest': number_of_questions,
                          'Mixed forest': number_of_questions,
                          'Natural grasslands': number_of_questions,
                          'Moors and heathland': number_of_questions,
                          'Sclerophyllous vegetation': number_of_questions,
                          'Transitional woodland-shrub': number_of_questions,
                          'Beaches, dunes, sands': number_of_questions,
                          'Bare rocks': number_of_questions,
                          'Sparsely vegetated areas': number_of_questions,
                          'Burnt areas': number_of_questions,
                          'Glaciers and perpetual snow': number_of_questions,
                          'Inland marshes': number_of_questions,
                          'Peat bogs': number_of_questions,
                          'Coastal salt marshes': number_of_questions,
                          'Salines': number_of_questions,
                          'Intertidal flats': number_of_questions,
                          'Water courses': number_of_questions,
                          'Water bodies': number_of_questions,
                          'Coastal lagoons': number_of_questions,
                          'Estuaries': number_of_questions,
                          'Sea and ocean': number_of_questions,
                      },
                      "vegetation": {"banana plantation": number_of_questions,
                                     "wooded area": number_of_questions,
                                     "sugar cane plantation": number_of_questions,
                                     "coniferous forest": number_of_questions,
                                     "deciduous forest": number_of_questions,
                                     "mixed forest": number_of_questions,
                                     "young forest plantation": number_of_questions,
                                     "hedge": number_of_questions,
                                     "field of hop plants": number_of_questions,
                                     "woody plant": number_of_questions,
                                     "poplar plantation": number_of_questions,
                                     "rice plantation": number_of_questions,
                                     "vineyard": number_of_questions},
                      "water": {"canal": number_of_questions, 'river delta': number_of_questions,
                                "estuary": number_of_questions, "lake": number_of_questions,
                                "lagoon": number_of_questions, "swamp": number_of_questions,
                                "pond": number_of_questions, "reservoir": number_of_questions},
                      "department": {"Hérault": 2*number_of_questions, "Manche": 2*number_of_questions,
                                     "Paris": 2*number_of_questions, "Hauts-de-Seine": 2*number_of_questions,
                                     "Seine-Saint-Denis": 2*number_of_questions,
                                     "Val-de-Marne": 2*number_of_questions},
                      "region": {"Occitanie": int(3.33 * number_of_questions), "Normandie": int(3.33 * number_of_questions),
                                 "Île-de-France": int(3.33 * number_of_questions)},
                      "smallest_largest": {"top-left of the image": 1 * number_of_questions,
                                           "top-right of the image": 1 * number_of_questions,
                                           "top of the image": 1 * number_of_questions,
                                           'left of the image': 1 * number_of_questions,
                                           "right of the image": 1 * number_of_questions,
                                           "center of the image": 1 * number_of_questions,
                                           "bottom-left of the image": 1 * number_of_questions,
                                           "bottom-right of the image": 1 * number_of_questions,
                                           "bottom of the image": 1 * number_of_questions},
                      "nearest": {"top-left of the image": 1 * number_of_questions,
                                  "top-right of the image": 1 * number_of_questions,
                                  "top of the image": 1 * number_of_questions,
                                  'left of the image': 1 * number_of_questions,
                                  "right of the image": 1 * number_of_questions,
                                  "center of the image": 1 * number_of_questions,
                                  "bottom-left of the image": 1 * number_of_questions,
                                  "bottom-right of the image": 1 * number_of_questions,
                                  "bottom of the image": 1 * number_of_questions},
                      "spatial_relation": {"top-left": 1 * number_of_questions, "top-right": 1 * number_of_questions,
                                           "above": 1 * number_of_questions,
                                           'left': 1 * number_of_questions, "right": 1 * number_of_questions,
                                           "inside": 1 * number_of_questions, "contains": 1 * number_of_questions,
                                           "bottom-left": 1 * number_of_questions,
                                           "bottom-right": 1 * number_of_questions,
                                           "below": 1 * number_of_questions, "intersecting": 1 * number_of_questions},
                      "name_mountains": {},

                      "count": {},
                      "area": {},
                      "density": {},
                      'distance': {},
                      'percent_clc': {}

                      }

    tries = 0
    questions_dict = []
    answers_dict = []
    canonical_dict = []
    images_dict = []
    list_images_ids = []
    start = time.time()

    dict_to_keep_max_num = {"count": 0, "distance": 0, "area": 0, "density": 0, "name_mountains": 0, "percent_clc": 0}

    while not check_all_zero(target_answers) and tries < 6:
        tries += 1
        print(target_answers)
        for ids, patch_id in tqdm(enumerate(listPatches)):
            if patch_id not in question_by_patch:
                question_by_patch[patch_id] = 50
            elif question_by_patch[patch_id] == 0:  # TODO: limit number of question per patch
                continue

            if check_all_zero(target_answers) or tries > 6:
                break
            img_path = os.path.join(target_dir, patch_id)
            QuestionBuilder = ConstructImageQuestion.ConstructImageQuestion(img_path, bd_topo_dict, big_GPD,
                                                                            (urban_shp, urban_csv), dept,
                                                                            (tri_fai, tri_moy, tri_for), mountains_shp,
                                                                            clc_shp)
            returned = QuestionBuilder.askQuestion1()
            for item in returned:
                if len(item) == 0:
                    continue
                quest, answer, can_quest = item[0]
                if (quest, answer, can_quest) == ('', '', ''):
                    continue
                else:
                    if can_quest in canonical_dict:
                        continue
                    if isinstance(answer, list):
                        answer = answer[0]
                    type = can_quest[0]

                    idx = len(questions_dict)
                    answ = answer
                    if answer not in target_answers[type].keys() and type not in ["distance", "area"]:
                        target_answers[type][answer] = number_of_questions
                    elif answer not in target_answers[type].keys() and type in ['distance', 'area']:
                        if type == "distance":
                            answ = str(round(float(answer[:-1]))) + "m"
                        elif type == "area":
                            answ = str(round(float(answer[:-2]))) + "m2"

                        target_answers[type][answ] = number_of_questions
                    if target_answers[type][answ] > 0:
                        target_answers[type][answ] -= 1
                        if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                                dict_to_keep_max_num[
                                    type] <= 10 * number_of_questions:
                            dict_to_keep_max_num[type] += 1
                    else:
                        continue

                    if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                            dict_to_keep_max_num[
                                type] > 10 * number_of_questions:
                        continue
                    question_by_patch[patch_id] -= 1

                    questions_dict.append(
                        {"id": idx, "patch_id": patch_id, "question": quest,
                         "answer_id": idx, "type": can_quest[0], "active": True})
                    answers_dict.append({"id": idx, "answer": answer, "question_id": idx,
                                         "type": can_quest[0], "active": True})
                    canonical_dict.append(can_quest)
                    if patch_id in list_images_ids:
                        images_dict[ids]["questions_ids"].append(idx)
                    else:
                        images_dict.append({"id": patch_id, "active": True, "questions_ids": [idx]})
                    if sum(question_by_patch.values()) == 0:
                        break

        with open(target_dir + "/Q&A/allQuestions25.json", "w") as fp:
            json.dump({"questions": questions_dict}, fp)
        with open(target_dir + "/Q&A/allAnswers25.json", "w") as fp:
            json.dump({"answers": answers_dict}, fp)
        print(total_questions)
        #######ça a prit 3846.40815281868 : 64.11 min
        with open(target_dir + "/Q&A/stats25.json", "w") as fp:
            json.dump({"stats": target_answers}, fp)
    #get_stats(questions_dict, answers_dict)


    end = time.time()

    print(f"ça a prit {end - start}")
