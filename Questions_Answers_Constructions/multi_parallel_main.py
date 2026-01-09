import csv
import os
import time
import json
from collections import defaultdict
from multiprocessing import Pool, Manager
from pathlib import Path

import numpy as np
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import ConstructImageQuestion
import warnings

warnings.filterwarnings('ignore')


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


def process_patch(args):
    ids, patch_id, question_by_patch, target_answers, bd_topo_dict, big_GPD, urban_shp, urban_csv, dept, \
        tri_fai, tri_moy, tri_for, mountains_shp, clc_shp, target_dir, number_of_questions, questions_dict, \
        answers_dict, canonical_dict, images_dict, list_images_ids, dict_to_keep_max_num = args

    # target_answers = args[3]

    if patch_id not in question_by_patch:
        question_by_patch[patch_id] = 50
    elif question_by_patch[patch_id] == 0:
        return

    img_path = os.path.join(target_dir, patch_id)
    QuestionBuilder = ConstructImageQuestion.ConstructImageQuestion(img_path, bd_topo_dict, big_GPD,
                                                                    (urban_shp, urban_csv), dept,
                                                                    (tri_fai, tri_moy, tri_for), mountains_shp,
                                                                    clc_shp)

    returned = QuestionBuilder.askQuestion4()
    """print(patch_id)
    print(returned)
    print("----")"""
    del QuestionBuilder
    """print(target_answers["area"])
    print("==")"""
    for item in returned:
        if not item:
            continue

        if len(item) != 3:
            print(f"Unexpected number of elements in 'item': {len(item)}")
            continue

        quest, answer, can_quest = item

        if (quest, answer, can_quest) == ('', '', ''):
            continue
        else:
            if can_quest in canonical_dict:
                continue
            if isinstance(answer, list):
                answ = answer[0]
            else:
                answ = answer

            type = can_quest[0]
            if type == "distance":
                answ = str(round(float(answer[:-1]))) + "m"
            elif type == "area":
                answ = str(round(float(answer[:-2]))) + "m2"
            idx = len(questions_dict)
            # with lock:
            if answ not in target_answers[type].keys():
                if type in ["percent_clc","density"]:
                    target_answers[type][answer] = int(number_of_questions / np.log1p((float(answ[:-1])) + 2))
                elif type == "count":
                    target_answers[type][answer] = int(number_of_questions / np.log1p((float(answ)) + 2))
                elif type == "distance":
                    target_answers[type][answ] = int(number_of_questions / np.log1p((float(answ[:-1])) + 2))
                elif type == "area":
                    target_answers[type][answ] = int(number_of_questions / np.log1p((float(answ[:-2])) + 2))
                else:
                    target_answers[type][answer] = int(number_of_questions)
            if target_answers[type][answ] > 0:

                target_answers[type][answ] -= 1
                if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                        dict_to_keep_max_num[type] <= 10 * number_of_questions:
                    dict_to_keep_max_num[type] += 1
            else:
                continue

            if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                    dict_to_keep_max_num[type] > 10 * number_of_questions:
                continue
            question_by_patch[patch_id] -= 1

            questions_dict.append(
                {"id": idx, "patch_id": patch_id, "question": quest,
                 "answer_id": idx, "type": can_quest[0], "active": True, "canonical": can_quest})
            answers_dict.append({"id": idx, "answer": answer, "question_id": idx,
                                 "type": can_quest[0], "active": True})
            canonical_dict.append(can_quest)
            if patch_id in list_images_ids:
                images_dict[ids]["questions_ids"].append(idx)
            else:
                images_dict.append({"id": patch_id, "active": True, "questions_ids": [idx]})
            if sum(question_by_patch.values()) == 0:
                break
    del returned
    return questions_dict, answers_dict, target_answers
    #print(target_answers["area"])
    # args[3] = target_answers


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

def save_results(results, output_dir, batch_number, questions_dict, answers_dict):
    # Define a function to save results, questions_dict, and answers_dict to a file
    """output_file = os.path.join(output_dir, f'results_batch_{batch_number}.txt')

    with open(output_file, 'w') as f:
        for result in results:
            f.write(str(result) + '\n')"""
    questions_dict1 = list(questions_dict)
    answers_dict1 = list(answers_dict)
    # Save questions_dict to JSON
    with open(os.path.join(output_dir, f'questions_dict_batch.json'), 'w') as fp:
        json.dump({"questions": questions_dict1}, fp)

    # Save answers_dict to JSON
    with open(os.path.join(output_dir, f'answers_dict_batch.json'), 'w') as fp:
        json.dump({"answers": answers_dict1}, fp)
    del questions_dict1,answers_dict1
if __name__ == "__main__":
    DATA_DIR = 'D:\\data\\BDOrtho+Sentinel2'
    urban_shp = gpd.read_file("D:\\data\\SIG\\ZAU\\com_uu2020_2023\\com_uu2020_2023.shp").to_crs(epsg=2154)
    import psutil

    total_threads = psutil.cpu_count() / psutil.cpu_count(logical=False)
    print('You can run {} processes per CPU core simultaneously'.format(total_threads))
    print(psutil.Process().cpu_affinity())
    urban_csv = []
    with open("D:\\data\\SIG\\ZAU\\UU2020_au_01-01-2023.csv", encoding="latin1") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            row1 = row[0].split(";")
            urban_csv.append((row1[0], row1[-3]))

    for dept in ["34"]:
        print(dept)
        if dept not in ["75", "92", "93", "94"]:
            mountains_shp = gpd.read_file(f"D:\\data\\SIG\\Mountains\\{dept}\\m_massiv.shp").to_crs(epsg=2154)
        else:
            mountains_shp = None
        version = "test_version8\\" + dept
        clc_shp = gpd.read_file(f'D:\\data\\SIG\\CLC_PNE_RG\\{dept}\\clc.shp').to_crs(
            epsg=2154)
        floods_path = "D:\\data\\SIG\\Floods\\" + dept
        files_names = os.listdir(floods_path)
        shp_names = [x for x in files_names if "inondable" in x and ".shp" in x]
        tri_fai = gpd.GeoDataFrame(pd.concat(
            [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in
             [x for x in shp_names if "fai" in x]],
            ignore_index=True))
        # tri_fai = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "fai" in x][0])).to_crs(epsg=2154)
        tri_moy = gpd.GeoDataFrame(pd.concat(
            [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in
             [x for x in shp_names if "moy" in x]],
            ignore_index=True))
        # tri_moy = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "moy" in x][0])).to_crs(epsg=2154)
        tri_for = gpd.GeoDataFrame(pd.concat(
            [gpd.read_file(os.path.join(floods_path, i)).to_crs(epsg=2154) for i in
             [x for x in shp_names if "for" in x]],
            ignore_index=True))
        # tri_for = gpd.read_file(os.path.join(floods_path, [x for x in shp_names if "for" in x][0])).to_crs(epsg=2154)
        target_dir = os.path.join(DATA_DIR, version)
        manager = Manager()
        questions_dict = manager.list()
        answers_dict = manager.list()
        canonical_dict = manager.list()
        """questions_dict = []
        answers_dict = []"""
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

        target_answers = manager.dict(
            {"less_more_equal": manager.dict({"yes": 5 * number_of_questions, "no": 5 * number_of_questions}),
             "presence": manager.dict({"yes": 5 * number_of_questions, "no": 5 * number_of_questions}),
             "pres_flooding": manager.dict({"yes": 5 * number_of_questions, "no": 5 * number_of_questions}),
             "pres_mountains": manager.dict({"yes": 5 * number_of_questions, "no": 5 * number_of_questions}),
             "lvl_flooding": manager.dict({"high": int(3.33 * number_of_questions),
                                           "medium": int(3.33 * number_of_questions),
                                           "low": int(3.33 * number_of_questions)}),
             "urban": manager.dict({"Outside Urban Unit": int(2.5 * number_of_questions),
                                    "City Center": int(2.5 * number_of_questions),
                                    "Suburbs": int(2.5 * number_of_questions),
                                    "Isolated City": int(2.5 * number_of_questions)}),
             "type_flooding": manager.dict({"River overflows": int(2.5 * number_of_questions),
                                            "Runoff": int(2.5 * number_of_questions),
                                            "Sea flooding": int(2.5 * number_of_questions),
                                            "Groundwater overflows": int(2.5 * number_of_questions)}),
             "clc": manager.dict({
                 'Continuous urban fabric': int(number_of_questions),
                 'Discontinuous urban fabric': int(number_of_questions),
                 'Industrial or commercial units and public facilities': int(number_of_questions),
                 'Road and rail networks and associated land': int(number_of_questions),
                 'Port areas': int(number_of_questions),
                 'Airports': int(number_of_questions),
                 'Mineral extraction sites': int(number_of_questions),
                 'Dump sites': int(number_of_questions),
                 'Construction sites': int(number_of_questions),
                 'Green urban areas': int(number_of_questions),
                 'Sport and leisure facilities': int(number_of_questions),
                 'Non-irrigated arable land': int(number_of_questions),
                 'Permanently irrigated land': int(number_of_questions),
                 'Rice fields': int(number_of_questions),
                 'Vineyards': int(number_of_questions),
                 'Fruit trees and berry plantations': int(number_of_questions),
                 'Olive groves': int(number_of_questions),
                 'Pastures, meadows and other permanent grasslands under agricultural use': int(number_of_questions),
                 'Annual crops associated with permanent crops': int(number_of_questions),
                 'Complex cultivation patterns': int(number_of_questions),
                 'Land principally occupied by agriculture, with significant areas of natural vegetation': int(
                     number_of_questions),
                 'Agro-forestry areas': int(number_of_questions),
                 'Broad-leaved forest': int(number_of_questions),
                 'Coniferous forest': int(number_of_questions),
                 'Mixed forest': int(number_of_questions),
                 'Natural grasslands': int(number_of_questions),
                 'Moors and heathland': int(number_of_questions),
                 'Sclerophyllous vegetation': int(number_of_questions),
                 'Transitional woodland-shrub': int(number_of_questions),
                 'Beaches, dunes, sands': int(number_of_questions),
                 'Bare rocks': int(number_of_questions),
                 'Sparsely vegetated areas': int(number_of_questions),
                 'Burnt areas': int(number_of_questions),
                 'Glaciers and perpetual snow': int(number_of_questions),
                 'Inland marshes': int(number_of_questions),
                 'Peat bogs': int(number_of_questions),
                 'Coastal salt marshes': int(number_of_questions),
                 'Salines': int(number_of_questions),
                 'Intertidal flats': int(number_of_questions),
                 'Water courses': int(number_of_questions),
                 'Water bodies': int(number_of_questions),
                 'Coastal lagoons': int(number_of_questions),
                 'Estuaries': int(number_of_questions),
                 'Sea and ocean': int(number_of_questions),
             }),
             "vegetation_smallest": manager.dict({"banana plantation": number_of_questions,
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
                                         "vineyard": number_of_questions}),
             "vegetation_largest": manager.dict({"banana plantation": number_of_questions,
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
                                         "vineyard": number_of_questions}),
             "water_smallest": manager.dict({"canal": int(1.25*number_of_questions), 'river delta': int(1.25*number_of_questions),
                                    "estuary": int(1.25*number_of_questions), "lake": int(1.25*number_of_questions),
                                    "lagoon": int(1.25*number_of_questions), "swamp": int(1.25*number_of_questions),
                                    "pond": int(1.25*number_of_questions), "reservoir": int(1.25*number_of_questions)}),
             "water_largest": manager.dict({"canal": int(1.25*number_of_questions), 'river delta': int(1.25*number_of_questions),
                                    "estuary": int(1.25*number_of_questions), "lake": int(1.25*number_of_questions),
                                    "lagoon": int(1.25*number_of_questions), "swamp": int(1.25*number_of_questions),
                                    "pond": int(1.25*number_of_questions), "reservoir": int(1.25*number_of_questions)}),
             "department": manager.dict({"Hérault": 2473, "Manche":2473,
                                         "Paris": 2473, "Hauts-de-Seine": 2473,
                                         "Seine-Saint-Denis": 2473,
                                         "Val-de-Marne": 2473}),
             "region": manager.dict({"Occitanie": 16274,
                                     "Normandie": 16274,
                                     "Île-de-France": 16274}),
             "smallest_largest": manager.dict({"top-left of the image": int(1.11 * number_of_questions),
                                               "top-right of the image": int(1.11 * number_of_questions),
                                               "top of the image":int(1.11 * number_of_questions),
                                               'left of the image': int(1.11 * number_of_questions),
                                               "right of the image":int(1.11 * number_of_questions),
                                               "center of the image": int(1.11 * number_of_questions),
                                               "bottom-left of the image": int(1.11 * number_of_questions),
                                               "bottom-right of the image": int(1.11 * number_of_questions),
                                               "bottom of the image": int(1.11 * number_of_questions)}),
             "nearest": manager.dict({"top-left of the image": int(1.11 * number_of_questions),
                                      "top-right of the image": int(1.11 * number_of_questions),
                                      "top of the image": int(1.11 * number_of_questions),
                                      'left of the image': int(1.11 * number_of_questions),
                                      "right of the image": int(1.11 * number_of_questions),
                                      "center of the image":int(1.11 * number_of_questions),
                                      "bottom-left of the image": int(1.11 * number_of_questions),
                                      "bottom-right of the image": int(1.11 * number_of_questions),
                                      "bottom of the image": int(1.11 * number_of_questions)}),
             "spatial_relation": manager.dict({"top-left": int(1.25 * number_of_questions),
                                               "top-right": int(1.25 * number_of_questions),
                                               "above": int(1.25 * number_of_questions),
                                               'left': int(1.25 * number_of_questions),
                                               "right": int(1.25 * number_of_questions),
                                               "bottom-left":int(1.25 * number_of_questions),
                                               "bottom-right": int(1.25 * number_of_questions),
                                               "below":int(1.25 * number_of_questions),
                                               }),
             "name_mountains": manager.dict({}),

             "count": manager.dict({}),
             "area": manager.dict({}),
             "density": manager.dict({}),
             'distance': manager.dict({}),
             'percent_clc': manager.dict({})

             })

        tries = 0
        # questions_dict = []
        # answers_dict = []
        # canonical_dict = []
        images_dict = []
        list_images_ids = []
        start = time.time()
        question_by_patch = manager.dict({})

        dict_to_keep_max_num = manager.dict({"count": 0, "distance": 0, "area": 0, "density": 0, "name_mountains": 0,
                                             "percent_clc": 0})
        #lock = manager.Lock()
        print("CPU COUNT : ", os.cpu_count())
        #pool = Pool(processes=6)
        patches_used = []
        with open("D:\\data\\BDOrtho+Sentinel2\\test_version8\\34\\Q&A\\answers_dict_batch_corrected.json") as fp:
            answers = json.load(fp)["answers"]
        with open("D:\\data\\BDOrtho+Sentinel2\\test_version8\\34\\Q&A\\questions_dict_batch_corrected.json") as fp:
            questions = json.load(fp)["questions"]
        for item in tqdm(zip(questions, answers)):
            quest, answer = item
            patch_id = quest["patch_id"]
            if patch_id not in patches_used:
                patches_used.append(patch_id)
            if patch_id not in question_by_patch:
                question_by_patch[patch_id] = 50
            elif question_by_patch[patch_id] == 0:
                continue
            can_quest = quest["canonical"]
            type = quest["type"]
            answer = answer["answer"]
            quest = quest["question"]
            if isinstance(answer, list):
                answ = answer[0]
            else:
                answ = answer
            if type == "distance":
                answ = str(round(float(answer[:-1]))) + "m"
            elif type == "area":
                answ = str(round(float(answer[:-2]))) + "m2"
            idx = len(questions)
            # with lock:
            if answ not in target_answers[type].keys():
                if type in ["percent_clc", "density"]:
                    target_answers[type][answer] = int(number_of_questions / np.log1p((float(answ[:-1])) + 2))
                elif type == "count":
                    target_answers[type][answer] = int(number_of_questions / np.log1p((float(answ)) + 2))
                elif type == "distance":
                    target_answers[type][answ] = int(number_of_questions / np.log1p((float(answ[:-1])) + 2))
                elif type == "area":
                    target_answers[type][answ] = int(number_of_questions / np.log1p((float(answ[:-2])) + 2))
                else:
                    target_answers[type][answ] = int(number_of_questions)
            if target_answers[type][answ] > 0:

                target_answers[type][answ] -= 1
                if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                        dict_to_keep_max_num[type] <= 10 * number_of_questions:
                    dict_to_keep_max_num[type] += 1
            else:
                continue

            if type in ["count", "density", "distance", "area", "name_mountains", "percent_clc"] and \
                    dict_to_keep_max_num[type] > 10 * number_of_questions:
                continue
            question_by_patch[patch_id] -= 1
        questions_dict += questions
        answers_dict += answers
        new_list_patches =[item for item in listPatches if item not in patches_used]
        print(len(new_list_patches))
        print(len(patches_used))
        print("--")
        args_list = [
            (ids, patch_id, question_by_patch, target_answers, bd_topo_dict, big_GPD, urban_shp, urban_csv, dept,
             tri_fai, tri_moy, tri_for, mountains_shp, clc_shp, target_dir, number_of_questions, questions_dict,
             answers_dict, canonical_dict, images_dict, list_images_ids, dict_to_keep_max_num)
            for ids, patch_id in enumerate(new_list_patches)
        ]
        #TODO rather than doing + thing, check the patches that don't have q&a !!!! for ids, patch_id in enumerate(listPatches[25500+15810+2055+780+2716+61095+6000:])

        #results = list(tqdm(pool.imap(process_patch, args_list, chunksize=15)))

        output_directory = target_dir + "/Q&A/"
        batch_size = 1000
        current_batch_number = 0
        results = []

        # Use a pool for multiprocessing
        with Pool(5) as pool:
            for args in tqdm(pool.imap(process_patch, args_list, chunksize=15), total=len(args_list)):
                results.append(args)

                # Check if it's time to save the results
                if len(results) % batch_size == 0:
                    save_results(results, output_directory, current_batch_number, questions_dict, answers_dict)
                    current_batch_number += 1
                    results = []  # Clear the results for the next batch


        pool.close()
        pool.join()
        questions_dict = list(questions_dict)
        answers_dict = list(answers_dict)
        target_answers = dict(target_answers)

        with open(target_dir + "/Q&A/allQuestions_final1.json", "w") as fp:
            json.dump({"questions": questions_dict}, fp)
        with open(target_dir + "/Q&A/allAnswers_final1.json", "w") as fp:
            json.dump({"answers": answers_dict}, fp)

        """with open(target_dir + "/Q&A/stats30.json", "w") as fp:
            json.dump({"stats": target_answers}, fp)"""

        get_stats(questions_dict, answers_dict)
