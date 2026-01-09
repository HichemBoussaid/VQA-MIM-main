import os
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
from shapely.geometry import box
from shapely import distance
import geopandas as gpd
import Objects
import cv2
from multiprocessing import Pool
from random import choice as randchoice
from shapely.geometry import Point, LineString, Polygon

key_clc = {
    '111': 'Continuous urban fabric',
    '112': 'Discontinuous urban fabric',
    '121': 'Industrial or commercial units and public facilities',
    '122': 'Road and rail networks and associated land',
    '123': 'Port areas',
    '124': 'Airports',
    '131': 'Mineral extraction sites',
    '132': 'Dump sites',
    '133': 'Construction sites',
    '141': 'Green urban areas',
    '142': 'Sport and leisure facilities',
    '211': 'Non-irrigated arable land',
    '212': 'Permanently irrigated land',
    '213': 'Rice fields',
    '221': 'Vineyards',
    '222': 'Fruit trees and berry plantations',
    '223': 'Olive groves',
    '231': 'Pastures, meadows and other permanent grasslands under agricultural use',
    '241': 'Annual crops associated with permanent crops',
    '242': 'Complex cultivation patterns',
    '243': 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    '244': 'Agro-forestry areas',
    '311': 'Broad-leaved forest',
    '312': 'Coniferous forest',
    '313': 'Mixed forest',
    '321': 'Natural grasslands',
    '322': 'Moors and heathland',
    '323': 'Sclerophyllous vegetation',
    '324': 'Transitional woodland-shrub',
    '331': 'Beaches, dunes, sands',
    '332': 'Bare rocks',
    '333': 'Sparsely vegetated areas',
    '334': 'Burnt areas',
    '335': 'Glaciers and perpetual snow',
    '411': 'Inland marshes',
    '412': 'Peat bogs',
    '421': 'Coastal salt marshes',
    '422': 'Salines',
    '423': 'Intertidal flats',
    '511': 'Water courses',
    '512': 'Water bodies',
    '521': 'Coastal lagoons',
    '522': 'Estuaries',
    '523': 'Sea and ocean',
}


class ConstructImageQuestion:
    def __init__(self, patch_id, bd_topo_dict, big_GPD, urban, dept, tri, mountains, clc_shp):
        self.image = None
        self.image_path = None
        self.image_shape = (1000, 1000)
        self.image_resolution = 0.2
        self.patch_id = patch_id
        self.dept = dept
        self.mountains_shp = mountains
        self.clc_shp = clc_shp
        self.bd_topo_dict = bd_topo_dict
        self.big_gpd = big_GPD
        self.features = {}
        self.questions = []
        self.answers = []
        self.number_of_categories = 8
        self.tri_fai, self.tri_moy, self.tri_for = tri
        # Self.excludes is the exclude part in the json
        self.excludes = {}
        self.canonical_questions = []
        self.all_objects = []
        self.current_question_type = None
        self.size_attributes = ["smallest", "largest"]
        self.position_attributes = ["top", "top-left", "top-right", "bottom", "bottom-right", "bottom-left", "left",
                                    "right", "center"]
        # Le type est défini ainsi [proba, function, args] (voir exemples ci-dessous)
        """self.QUESTION_TYPES = {"clc": [1, self.clc_questions, 1],
                               "percent_clc": [1,self.percent_clc_questions,1]
                               }"""

        self.QUESTION_TYPES = {'area': [1, self.new_area, 1], "nearest": [1, self.generic_nearest, 1],
                               'presence': [1, self.presence, 1],
                               'count': [1, self.count, 1], 'less_more_equal': [0.5, self.less_more_equal, 2],
                               'density': [1, self.density, 1], 'spatial_relation': [1, self.spatial_relation, 1],
                               "smallest_largest": [1, self.smallest_largest, 1],
                               "distance": [1, self.distance_question, 1],
                               "clc": [1, self.clc_questions, 1],
                               "percent_clc": [1, self.percent_clc_questions, 1],
                               "region": [1, self.region_question, 1],
                               "department": [1, self.department_question, 1],
                               "urban": [1, self.urban_questions, 1],
                               "water_sm": [1, self.water_questions_sm, 1],
                               "water_lg": [1, self.water_questions_lg, 1],
                               "vegetation_sm": [1, self.vegetation_questions_sm, 1],
                               "vegetation_lg": [1, self.vegetation_questions_lg, 1],
                               "pres_flooding": [1, self.presence_floods_questions, 1],
                               "lvl_flooding": [1, self.level_floods_questions, 1],
                               "type_flooding": [1, self.type_floods_questions, 1],
                               "pres_mountains": [1, self.presence_mountains_questions, 1],
                               "name_mountains": [1, self.name_mountains_questions, 1],
                               }
        self.urban_shp, self.urban_csv = urban

        # Activer ou desactiver selon le type de question choisi
        # self.QUESTION_TYPES['rural_urban'] = [0, self.ruralUrban, 0]
        self.getFeatures()
        self.constructAllObjects()
        self.water_objects = [x for x in self.all_objects if x.parent == "water area" and x.count > 0]
        self.vegetation_objects = [x for x in self.all_objects if (x.parent == "vegetation zone" or x.string in
                                                                   ["public forest", "national park"]) and x.count > 0]
        src = self.image
        left, bottom, right, top = src.bounds
        bbox = box(left, bottom, right, top)
        gdf_img = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:2154")
        self.tri_fai = gpd.overlay(self.tri_fai, gdf_img,
                                   keep_geom_type=False, how='intersection')
        self.tri_moy = gpd.overlay(self.tri_moy, gdf_img,
                                   keep_geom_type=False, how='intersection')
        self.tri_for = gpd.overlay(self.tri_for, gdf_img,
                                   keep_geom_type=False, how='intersection')

        self.clc_intersection = gpd.overlay(self.clc_shp, gdf_img,
                                            keep_geom_type=False, how='intersection')

        self.mountains_intersection = gpd.overlay(self.mountains_shp, gdf_img,
                                                  keep_geom_type=False, how='intersection') \
            if self.dept not in ["75", "92", "93", "94"] else None

        self.urban_intersection = gpd.overlay(self.urban_shp, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:2154"),
                                              keep_geom_type=False, how='intersection')
        del self.big_gpd
    def clc_questions(self):
        question = "Which land cover category occupies the largest area"

        max_area_index = self.clc_intersection.geometry.area.idxmax()
        answer = key_clc[self.clc_shp.loc[max_area_index, 'code_18']]
        return question, answer, ["clc", answer, self.patch_id]

    def percent_clc_questions(self):
        random_key = random.choice(list(key_clc.keys()))

        question = f"What percentage of the area is {key_clc[random_key]}"
        total_area = self.clc_intersection.loc[self.clc_shp['code_18'] == random_key, 'geometry'].area.sum()

        image_area = self.image_shape[0] * self.image_shape[
            1] * self.image_resolution * self.image_resolution  # in square units

        percentage = (total_area / image_area) * 100
        answer = "{:.2f}%".format(percentage)
        return question, answer, ["percent_clc", key_clc[random_key], answer, self.patch_id]

    def presence_mountains_questions(self):
        if self.dept in ["75", "92", "93", "94"]:
            return '', '', ''

        answer = "no" if self.mountains_intersection.is_empty.all() else "yes"
        question = "Are there mountains"
        return question, answer, ["pres_mountains", answer, self.patch_id]

    def name_mountains_questions(self):

        if self.dept in ["75", "92", "93", "94"] or self.mountains_intersection.is_empty.all():
            return '', '', ''
        question = "What is the name of the mountain range"
        answer = self.mountains_intersection.name_mm[0]
        return question, answer, ["name_mountains", answer, self.patch_id]

    def urban_questions(self):
        key_to_classif = {"H": "Outside Urban Unit", "C": "City Center", "B": "Suburbs", "I": "Isolated City"}
        codegeo = self.urban_intersection['codgeo'].tolist()
        if len(codegeo) > 0:
            codegeo = self.urban_intersection['codgeo'].tolist()[0]
        else:
            return '', '', ''
        matching_tuple = next((t for t in self.urban_csv if codegeo in t), None)
        if matching_tuple is None:
            return '', '', ''
        question = "What is the urban classification of the area"
        answer = key_to_classif[matching_tuple[1]]
        return question, answer, ['urban', answer, self.patch_id]

    def askQuestion2(self, type=None):
        to_return = []
        tries = 0
        while tries < 20:
            question, answer, canonical_question = self.QUESTION_TYPES[type][1]()
            if (question, answer, canonical_question) == ('', '', ''):
                tries += 1
                continue
            if canonical_question[0] not in ["region", "department", "pres_flooding"]:
                question += ' in the image'
            question += '?'
            to_return.append((question, answer, canonical_question))
            break
        return to_return

    def askQuestion1(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.askQuestion2, self.QUESTION_TYPES.keys()))
        return results


    def askQuestion4(self, type=None, answerWanted=None):
        if answerWanted is None:
            to_return = []
            for i in range(7):
                for type in self.QUESTION_TYPES:
                    tries = 0
                    if tries >= 10:
                        break
                    else:
                        question, answer, canonical_question = self.QUESTION_TYPES[type][1]()
                        if (question, answer, canonical_question) == ('', '', ''):
                            tries += 1
                            continue
                        if canonical_question[0] not in ["region", "department", "pres_flooding"]:
                            question += ' in the image'
                        question += '?'
                        to_return.append((question, answer, canonical_question))
            return to_return

    def department_question(self):
        dept_keys = {"34": "Hérault", "50": "Manche", "75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
                     "94": "Val-de-Marne", "74": "Haute-Savoie"}
        question = "To which department does the area in the image belong to"
        answer = dept_keys[self.dept]
        return question, answer, ["department", self.patch_id]

    def region_question(self):
        region_keys = {"34": "Occitanie", "50": "Normandie", "75": "Île-de-France", "92": "Île-de-France",
                       "93": "Île-de-France", "94": "Île-de-France", "74": "Auvergne-Rhône-Alpes"}
        question = "To which region does the area in the image belong to"
        answer = region_keys[self.dept]
        return question, answer, ["region", self.patch_id]

    def water_questions_lg(self):
        if len(self.water_objects) >= 1:
            #choice = randchoice([i for i in range(len(self.size_attributes))])
            choice = 1
            chosen_attribute = self.size_attributes[choice]
            question = f"What type of water body is the {chosen_attribute}"
            mini = 10000
            index = 0
            maxi = -1000
            obj_to_keep = None
            for obj in self.water_objects:
                idx = self.get_object_with_attribute(obj, choice, "size")
                if idx is None:
                    return '', '', ''
                """if choice == 0 and mini > obj.features[index][1].geometry.area:
                    obj_to_keep = obj
                    mini = obj.features[index][1].geometry.area"""
                if choice == 1 and maxi < obj.features[index][1].geometry.area:
                    obj_to_keep = obj
                    maxi = obj.features[index][1].geometry.area
            if obj_to_keep is None:
                return '','',''
            answer = obj_to_keep.string

            return question, answer, ["water_largest", answer, self.patch_id]
            """elif len(self.water_objects) == 1:
            question = f"What type of water body is present"
            answer = self.water_objects[0].string
            return question, answer, ["water_largest", answer, self.patch_id]"""

        else:
            return '', '', ''

    def water_questions_sm(self):
        if len(self.water_objects) >= 1:
            #choice = randchoice([i for i in range(len(self.size_attributes))])
            choice = 0
            chosen_attribute = self.size_attributes[choice]
            question = f"What type of water body is the {chosen_attribute}"
            mini = 10000
            index = 0
            maxi = -1000
            obj_to_keep = None
            for obj in self.water_objects:
                idx = self.get_object_with_attribute(obj, choice, "size")
                if idx is None:
                    return '', '', ''
                if choice == 0 and mini > obj.features[index][1].geometry.area:
                    obj_to_keep = obj
                    mini = obj.features[index][1].geometry.area
                elif choice == 1 and maxi < obj.features[index][1].geometry.area:
                    obj_to_keep = obj
                    maxi = obj.features[index][1].geometry.area
            if obj_to_keep is None:
                return '', '', ''
            answer = obj_to_keep.string
            return question, answer, ["water_smallest", answer, self.patch_id]
            """elif len(self.water_objects) == 1:
            question = f"What type of water body is present"
            answer = self.water_objects[0].string
            return question, answer, ["water_smallest", answer, self.patch_id]"""

        else:
            return '', '', ''

    def presence_floods_questions(self):
        question = "Is this area prone to flooding"

        if not (self.tri_for.is_empty.all() and self.tri_fai.is_empty.all() and self.tri_moy.is_empty.all()):
            answer = "yes"
        else:
            answer = "no"

        return question, answer, ["pres_flooding", answer, self.patch_id]

    def level_floods_questions(self):
        if not (self.tri_for.is_empty.all() and self.tri_fai.is_empty.all() and self.tri_moy.is_empty.all()):
            areas = {"high": self.tri_for.geometry.area.sum(),
                     "medium": self.tri_moy.geometry.area.sum(),
                     "low": self.tri_fai.geometry.area.sum()}
            question = "What is the flood risk level"
            answer = ""
            for area in ['high', 'medium', 'low']:
                if areas[area] != 0:
                    answer = area
                    break
            if answer == "":
                return '', '', ''
            return question, answer, ["lvl_flooding", answer, self.patch_id]
        else:
            return '', '', ''

    def type_floods_questions(self):
        if not (self.tri_for.is_empty.all() and self.tri_fai.is_empty.all() and self.tri_moy.is_empty.all()):
            areas = {"high": self.tri_for.geometry.area.sum(),
                     "medium": self.tri_moy.geometry.area.sum(),
                     "low": self.tri_fai.geometry.area.sum()}

            key_nature = {"01": "River overflows", "02": "Runoff", "03": "Sea flooding", "04": "Groundwater overflows"}
            risk_mapping = {
                "high": (key_nature[self.tri_for.typ_inond[0]] if not self.tri_for.is_empty.all() else None),
                "medium": key_nature[self.tri_moy.typ_inond[0]] if not self.tri_moy.is_empty.all() else None,
                "low": key_nature[self.tri_fai.typ_inond[0]] if not self.tri_fai.is_empty.all() else None}
            question = "What is the nature of flood risk"
            geom1 = self.tri_fai.unary_union
            geom2 = self.tri_moy.unary_union
            if geom1 is not None and geom2 is not None:
                intersection = geom1.intersection(geom2)

                intersection_area = intersection.area
                areas["low"] -= intersection_area
            geom1 = self.tri_moy.unary_union
            geom2 = self.tri_for.unary_union
            if geom1 is not None and geom2 is not None:
                # Calculate the intersection of the combined geometries
                intersection = geom1.intersection(geom2)
                if intersection is not None:
                    intersection_area = intersection.area
                    areas["medium"] -= intersection_area
            risk = max(areas, key=areas.get)
            answer = risk_mapping.get(risk)

            # Remove the max risk from the dictionary
            # areas.pop(risk, None)

            # Now, areas contains the rest of the risks
            rest_of_the_values = {risk: risk_mapping.get(risk) for risk in areas.keys() if
                                  risk_mapping.get(risk) is not None}
            if answer is None:
                return '', '', ''
            return question, list(set(rest_of_the_values.values())), ["type_flooding", answer, self.patch_id]
        else:
            return '', '', ''

    def vegetation_questions_lg(self):
        if len(self.vegetation_objects) > 1:
            #choice = randchoice([i for i in range(len(self.size_attributes))])
            choice = 1
            chosen_attribute = self.size_attributes[choice]
            question = f"Which vegetation type occupies the {chosen_attribute} area"
            mini = 10000
            index = 0
            maxi = -1000
            obj_to_keep = None
            for obj in self.vegetation_objects:
                idx = self.get_object_with_attribute(obj, choice, "size")
                if idx is None:
                    return '', '', ''
                if choice == 0 and mini > obj.features[idx][1].geometry.area:
                    obj_to_keep = obj
                    mini = obj.features[idx][1].geometry.area
                elif choice == 1 and maxi < obj.features[idx][1].geometry.area:
                    obj_to_keep = obj
                    maxi = obj.features[idx][1].geometry.area
            if obj_to_keep is None:
                return '', '', ''
            answer = obj_to_keep.string
            return question, answer, ["vegetation_largest", answer, self.patch_id]
            """elif len(self.vegetation_objects) == 1:
            question = f"What type of vegetation is present"
            answer = self.vegetation_objects[0].string
            return question, answer, ["vegetation", answer, self.patch_id]"""

        else:
            return '', '', ''

    def vegetation_questions_sm(self):
        if len(self.vegetation_objects) > 1:
            #choice = randchoice([i for i in range(len(self.size_attributes))])
            choice =0
            chosen_attribute = self.size_attributes[choice]
            question = f"Which vegetation type occupies the {chosen_attribute} area"
            mini = 10000
            index = 0
            maxi = -1000
            obj_to_keep = None
            for obj in self.vegetation_objects:
                idx = self.get_object_with_attribute(obj, choice, "size")
                if idx is None:
                    return '', '', ''
                if choice == 0 and mini > obj.features[idx][1].geometry.area:
                    obj_to_keep = obj
                    mini = obj.features[idx][1].geometry.area
                elif choice == 1 and maxi < obj.features[idx][1].geometry.area:
                    obj_to_keep = obj
                    maxi = obj.features[idx][1].geometry.area
            if obj_to_keep is None:
                return '', '', ''
            answer = obj_to_keep.string
            return question, answer, ["vegetation_smallest", answer, self.patch_id]
            """elif len(self.vegetation_objects) == 1:
            question = f"What type of vegetation is present"
            answer = self.vegetation_objects[0].string
            return question, answer, ["vegetation", answer, self.patch_id]"""

        else:
            return '', '', ''

    def getFeatures(self):
        for folder, v in self.bd_topo_dict.items():
            self.image_path = [x for x in os.listdir(self.patch_id) if "BDOrtho" in x and not "aux.xml" in x][0]

            with rasterio.open(os.path.join(self.patch_id, self.image_path)) as src:
                self.image = src
                self.image_shape = src.shape
            for shapefile, value in v.items():
                # shp_path = Path(os.path.join("D:\\data\\BD_Topo", dept, "BDTopo_new", folder, shapefile))
                # if shp_path.exists():
                # gdf = gpd.read_file(shp_path)
                if folder not in self.big_gpd or shapefile not in self.big_gpd[folder]:
                    continue

                gdf = self.big_gpd[folder][shapefile]
                left, bottom, right, top = src.bounds
                bbox = box(left, bottom, right, top)
                intersection = gpd.overlay(gdf, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:2154"),
                                           keep_geom_type=False, how='intersection')

                # gdf.to_crs(epsg=3395)
                rows = []
                if "base" in value.keys():
                    for row in intersection.iterrows():
                        rows.append(row)
                    self.features[value["base"]] = rows
                    self.excludes[value["base"]] = value["exclude"]
                if len(value.keys()) >= 2:
                    for key, nature in value.items():
                        if key in ["base", "exclude"]:
                            continue

                        for row in intersection.iterrows():
                            if row[1][key] in nature.keys():
                                self.excludes[nature[row[1][key]]] = value["exclude"]
                                if nature[row[1][key]] in self.features.keys():
                                    self.features[nature[row[1][key]]].append(row)
                                else:
                                    self.features[nature[row[1][key]]] = [row]

    def chooseObject(self, exclude=None, choose_in=None, count_condition=1):
        tried = []
        if choose_in is None:
            choice = randchoice([i for i in range(len(self.all_objects)) if i not in tried])
            tried.append(choice)
            objects = self.all_objects[choice]
            while self.current_question_type in objects.exclude or (
                    exclude is not None and objects.string == exclude and len(tried) < len(self.all_objects)):
                choice = randchoice([i for i in range(len(self.all_objects)) if i not in tried])
                tried.append(choice)
                objects = self.all_objects[choice]
            if len(tried) == len(self.all_objects):
                return None
            return objects
        else:
            indexes_to_choose_from = []
            for i, obj in enumerate(self.all_objects):
                if obj.string in choose_in:
                    indexes_to_choose_from.append(i)
            choice = randchoice(indexes_to_choose_from)
            tried.append(choice)
            objects = self.all_objects[choice]
            while self.current_question_type in objects.exclude or (
                    exclude is not None and objects.string == exclude and len(tried) < len(self.all_objects)) \
                    or objects.count < count_condition:
                choice = randchoice(indexes_to_choose_from)
                tried.append(choice)
                objects = self.all_objects[choice]
                if len(tried) == len(self.all_objects):
                    return None
            return objects

    def constructAllObjects(self):
        all_labels = []
        parents = {}
        for v in self.bd_topo_dict.values():
            for key, value in v.items():
                exclude = value["exclude"]
                parent = None
                for key_nature, nature in value.items():
                    if isinstance(nature, str):
                        all_labels.append(nature)
                        parents[nature] = parent
                        if nature not in self.excludes.keys():
                            self.excludes[nature] = exclude
                    elif isinstance(nature, dict):
                        if "base" in value.keys():
                            parent = value["base"]
                        for nat_2 in nature.values():
                            all_labels.append(nat_2)
                            parents[nat_2] = parent
                            if nat_2 not in self.excludes.keys():
                                self.excludes[nat_2] = exclude
        for k in all_labels:
            features = self.features[k] if k in self.features.keys() else []
            self.all_objects.append(Objects.Objects(k, features, self.excludes[k], parents[k]))

    def presence(self):

        objects = self.chooseObject()
        # print(objects.string)

        if objects is None:
            return '', '', ''
        answer = 'no'
        if objects.count > 0:
            answer = 'yes'
        if objects.string[0] in ['u', 'a', 'i', 'e', 'o']:
            choices = ['Is there an ' + objects.string,
                       'Is an ' + objects.string + ' present']
        else:
            choices = ['Is there a ' + objects.string,
                       'Is a ' + objects.string + ' present']
        question = choices[random.randint(0, len(choices) - 1)]
        return question, answer, ['presence', objects.plural_string, self.patch_id]

    def count(self):
        objects = self.chooseObject()

        if objects is None:
            return '', '', ''
        answer = str(objects.count)
        choices = ['How many ' + objects.plural_string + ' are',
                   'What is the number of ' + objects.plural_string,
                   'What is the amount of ' + objects.plural_string]
        question = choices[random.randint(0, len(choices) - 1)]
        return question, answer, ['count', objects.plural_string, self.patch_id]

    def less_more_equal(self):
        # print('less more')

        obj1 = self.chooseObject()
        tries = 0
        while obj1.count < 1 and tries < 20:
            obj1 = self.chooseObject()
            tries += 1
        if obj1.count < 1:
            return '', '', ''
        obj2 = self.chooseObject(exclude=obj1.string)
        if obj1 is None or obj2 is None:
            return '', '', ''

        choice = random.randint(0, 2)
        answer = 'no'
        if choice == 0:  # less
            question = 'Are there less ' + obj1.plural_string + ' than ' + obj2.plural_string
            if obj1.count < obj2.count:
                answer = 'yes'
        elif choice == 1:  # more
            question = 'Are there more ' + obj1.plural_string + ' than ' + obj2.plural_string
            if obj1.count > obj2.count:
                answer = 'yes'
        else:  # equal
            question = 'Is the number of ' + obj1.plural_string + ' equal to the number of ' + obj2.plural_string
            if obj1.count == obj2.count:
                answer = 'yes'
        return question, answer, ['less_more_equal', obj1.plural_string, obj2.plural_string, choice, self.patch_id]

    def area(self):
        objects = self.chooseObject()
        if objects is None:
            return '', '', ''
        area = 0
        for obj in objects.features:
            area += obj[1].geometry.area
        question = 'What is the area covered by ' + objects.plural_string
        answer = str(int(area)) + 'm2'
        return question, answer, ['area', objects.plural_string]

    def new_area(self):
        choose_in = ["building", "cemetery", "park", "sports field", "water area", "airport", "public forrest",
                     "vegetation zone"]

        attribute = self.size_attributes
        obj = self.chooseObject()

        if obj is None:
            return '', '', ''

        if obj.count > 1:
            choice = randchoice([i for i in range(len(attribute))])
            chosen_attribute = attribute[choice]
            index = self.get_object_with_attribute(obj, choice, "size")
            if index is None:
                return '', '', ''
        elif obj.count == 1:
            chosen_attribute = ""
            index = 0
        else:
            return '', '', ''

        question = f"What is the area of the {chosen_attribute} {obj.string}" if chosen_attribute != "" else f"What is the area of the {obj.string}"
        answer = f"{obj.features[index][1].geometry.area:.2f}m2" if index is not None else "0m2"
        return question, answer, ['area', obj.plural_string, chosen_attribute, self.patch_id]

    def density(self):
        objects = self.chooseObject()

        if objects is None:
            return '', '', ''
        count = 0
        for obj in objects.features:
            count += obj[1].geometry.area
        answer = f'{(100 * count) / (self.image_shape[0] * self.image_shape[1] * self.image_resolution ** 2) :.2f}%'

        question = "What is the " + objects.string + " density"
        return question, answer, ['density', objects.plural_string, self.patch_id]

    def smallest_largest(self):
        # choose_in = ["building", "cemetery", "park", "sports field", "water area", "airport", "public forrest",
        #             "vegetation zone"]

        attribute = self.size_attributes
        objects = self.chooseObject()
        tries = 1
        while objects.count < 2 and tries < 20:
            objects = self.chooseObject()
            tries += 1

        if objects is None or tries == 20:
            return '', '', ''

        choice = randchoice([i for i in range(len(attribute))])
        chosen_attribute = attribute[choice]
        question = f"Where is the {chosen_attribute} {objects.string}"

        index = self.get_object_with_attribute(objects, choice, "size")
        if index is None:
            return '', '', ''
        position = self.get_polygon_position(objects.features[index][1].geometry, self.image_shape[0],
                                             self.image_shape[1])
        xmin, ymin, xmax, ymax = objects.features[index][1].geometry.bounds
        col_min, row_min = self.image.index(xmin, ymin)
        col_max, row_max = self.image.index(xmax, ymax)
        answer = f"{position} of the image"
        return question, [answer, (row_min, col_min, row_max, col_max)], ['smallest_largest', objects.plural_string,
                                                                          chosen_attribute,
                                                                          (col_min, row_min, col_max, row_max)]

    def spatial_relation(self):
        choose_in = ["building", "cemetery", "park", "sports field", "water area", "airport", "public forrest",
                     "vegetation zone"]

        attribute = self.size_attributes
        obj1 = self.chooseObject(choose_in=choose_in, count_condition=1)
        if obj1 is None:
            return '', '', ''
        if obj1.count > 1:
            choice = randchoice([i for i in range(len(attribute))])
            chosen_attribute1 = attribute[choice]
            index1 = self.get_object_with_attribute(obj1, choice, "size")

        else:
            chosen_attribute1 = ""
            index1 = 0
        obj2 = self.chooseObject(exclude=obj1.string, choose_in=choose_in, count_condition=1)
        if obj2 is None:
            return '', '', ''
        if obj2.count > 1:
            choice = randchoice([i for i in range(len(attribute))])
            chosen_attribute2 = attribute[choice]
            index2 = self.get_object_with_attribute(obj2, choice, "size")
        else:
            chosen_attribute2 = ""
            index2 = 0
        if index1 is None or index2 is None or distance(obj1.features[index1][1].geometry, obj2.features[index2][1].geometry) == 0:
            return '', '', ''
        question = f"What is the relative position of the {chosen_attribute2} {obj2.string} with respect to the {chosen_attribute1} {obj1.string}"

        answer = get_spatial_relationship_between_two_objects(obj1.features[index1][1].geometry,
                                                              obj2.features[index2][1].geometry)
        xmin, ymin, xmax, ymax = obj2.features[index2][1].geometry.bounds
        col_min, row_min = self.image.index(xmin, ymin)
        col_max, row_max = self.image.index(xmax, ymax)

        if answer is None:
            return '', '', ''
        return question, [answer, (row_min, col_min, row_max, col_max)], ['spatial_relation', obj1.plural_string,
                                                                          obj2.plural_string, chosen_attribute1,
                                                                          chosen_attribute2, self.patch_id]

    def precise_nearest(self):
        choose_in1 = ["monument", "museum", "city hall"]
        choose_in2 = ["hospital", "high school", "city hall", "nursing home", "monument", "museum", "courthouse",
                      "police station"]
        obj1 = self.chooseObject(choose_in=choose_in1)
        if obj1 is None:
            return '', '', ''
        obj2 = self.chooseObject(exclude=obj1.string, choose_in=choose_in2)
        if obj2 is None:
            return '', '', ''
        choice = randchoice([i for i in range(len(obj1.features))])
        precise_object1 = obj1.features[choice][1]
        question = f"What is the closest {obj2.string} to {capitalize_first_letter(precise_object1.GRAPHIE)}"
        min_distance = 100000
        min_index = 0
        for i, f in enumerate(obj2.features):
            d = distance(precise_object1.geometry, f[1].geometry)
            if d < min_distance and f[1].GRAPHIE[:3] != "sep":
                min_distance = d
                min_index = i
        precise_object2 = obj2.features[min_index][1]
        answer = capitalize_first_letter(
            precise_object2.GRAPHIE) + f" {self.image.index(precise_object2.geometry.x, precise_object2.geometry.y)}"
        return question, answer, ['density', obj1.plural_string, obj2.plural_string]

    def generic_nearest(self):
        obj1 = self.chooseObject()
        tries = 1
        while obj1.count < 1 and tries < 20:
            obj1 = self.chooseObject()
            tries += 1
        if tries == 10 or obj1 is None:
            return '', '', ''
        obj2 = self.chooseObject(exclude=obj1.string)
        tries = 1
        while (obj2.count < 1 or obj2.parent == obj1.string or obj1.parent == obj2.string) and tries < 20:
            obj2 = self.chooseObject(exclude=obj1.string)
            tries += 1
        if obj2 is None or tries == 20:
            return '', '', ''
        if obj1.count > 1:
            precise_object1 = randchoice(obj1.features)[1]
            coordinates_of_obj1 = self.image.index(precise_object1.geometry.centroid.x,
                                                   precise_object1.geometry.centroid.y)
            question = f"Where is the closest {obj2.string} to {coordinates_of_obj1}"
        elif obj1.count == 1:
            precise_object1 = obj1.features[0][1]
            question = f"Where is the closest {obj2.string} to the {obj1.string}"
        else:
            return '', '', ''

        min_distance = 100000
        min_index = 0
        for i, f in enumerate(obj2.features):
            d = distance(precise_object1.geometry, f[1].geometry)
            if d < min_distance and (
                    "GRAPHIE" not in f[1].keys() or "GRAPHIE" in f[1].keys() and f[1].GRAPHIE[:3] != "sep"):
                min_distance = d
                min_index = i
        if min_distance == 0:
            return '', '', ''
        precise_object2 = obj2.features[min_index][1]
        position = self.get_polygon_position(precise_object2.geometry, self.image_shape[0], self.image_shape[1])
        xmin, ymin, xmax, ymax = precise_object2.geometry.bounds
        col_min, row_min = self.image.index(xmin, ymin)
        col_max, row_max = self.image.index(xmax, ymax)

        answer = f"{position} of the image"

        return question, [answer, (row_min, col_min, row_max, col_max)], ['nearest', obj1.plural_string,
                                                                          obj2.plural_string,
                                                                          (row_min, col_min, row_max, col_max)]

    def distance_question(self):

        obj1_attributes = ["smallest", "largest", "point"]
        choice = randchoice([i for i in range(len(obj1_attributes))])
        if choice == 2:
            obj1 = None
            row = random.randint(0, self.image_shape[0] - 1)
            col = random.randint(0, self.image_shape[1] - 1)
            obj1_question = (col, row)
            obj1_coordinates = Point(self.image.xy(row, col))
            # Chose random point
        else:

            obj1 = self.chooseObject()
            tries = 1
            while obj1.count < 1 and tries < 20:
                obj1 = self.chooseObject()
                tries += 1
            if obj1 is None or tries == 20:
                return '', '', ''
            if obj1.count > 1:
                chosen_attribute1 = obj1_attributes[choice]
                index1 = self.get_object_with_attribute(obj1, choice, "size")
                if index1 is None:
                    return '', '', ''
                obj1_coordinates = obj1.features[index1][1].geometry
                obj1_question = chosen_attribute1 + " " + obj1.string
            else:
                chosen_attribute1 = ""
                index1 = 0
                obj1_question = f"the {obj1.string}"
                obj1_coordinates = obj1.features[0][1].geometry
        if obj1 is not None:
            exclude = obj1.string
        else:
            exclude = None

        obj2_attributes = ["smallest", "largest", "closest"]
        choice = randchoice([i for i in range(len(obj2_attributes))])
        obj2 = self.chooseObject(exclude=exclude)
        tries = 1
        while (obj2.count < 1 or (
                exclude is not None and (obj2.parent == obj1.string or obj1.parent == obj2.string))) and tries < 20:
            obj2 = self.chooseObject(exclude=exclude)
            tries += 1
        if obj2 is None or tries == 20:
            return '', '', ''
        if choice == 2:
            min_distance = 100000
            min_index = 0
            for i, f in enumerate(obj2.features):
                d = distance(obj1_coordinates, f[1].geometry)
                if d < min_distance and (
                        "GRAPHIE" not in f[1].keys() or "GRAPHIE" in f[1].keys() and f[1].GRAPHIE[:3] != "sep"):
                    min_distance = d
                    min_index = i
            precise_object2 = obj2.features[min_index][1]
            answer = f"{min_distance:.2f}m"
            obj2_question = f"closest {obj2.string}"
            # Chose random point
        else:

            if obj2.count > 1:
                chosen_attribute2 = obj2_attributes[choice]
                index2 = self.get_object_with_attribute(obj2, choice, "size")
                if index2 is None:
                    return '', '', ''
                obj2_question = chosen_attribute2 + " " + obj2.string
                answer = f"{distance(obj1_coordinates, obj2.features[index2][1].geometry):.2f}m"
            else:
                chosen_attribute2 = ""
                index2 = 0
                obj2_question = f"the {obj2.string}"
                answer = f"{distance(obj1_coordinates, obj2.features[0][1].geometry):.2f}m"
                question = f"What is the distance of {obj2_question} with respect to {obj1_question}"
        if answer == "0m":
            return '', '', ''
        question = f"What is the distance of {obj2_question} with respect to {obj1_question}"
        return (question, answer, ["distance", obj1.plural_string, obj2.plural_string]) if obj1 is not None else (
            question, answer, ["distance", "point", obj2.plural_string, self.patch_id])

    def get_object_with_attribute(self, obj, choice, type_of_attributes):
        if type_of_attributes == "size":
            list_of_areas = [f[1].geometry.area for f in obj.features]
            if sum(list_of_areas) == 0:
                return None
            if choice == 0:
                result = min(list_of_areas)
            elif choice == 1:
                result = max(list_of_areas)
            else:
                return None
            return list_of_areas.index(result)
        if type_of_attributes == "position":
            for i, feature in enumerate(obj.features):
                if self.position_attributes[choice] == self.get_polygon_position(
                        feature[1].geometry, self.image_shape[0], self.image_shape[1]):
                    return i
            return None

    def get_polygon_position(self, polygon, image_width, image_height):
        # Calculate the centroid of the polygon
        centroid = polygon.centroid
        y, x = self.image.index(centroid.x, centroid.y)

        # Calculate the size of each zone
        zone_width = image_width / 3
        zone_height = image_height / 3

        # Determine the position of the polygon based on its centroid
        if x < zone_width:
            if y < zone_height:
                return "top-left"
            elif y < 2 * zone_height:
                return "left"
            else:
                return "bottom-left"
        elif x < 2 * zone_width:
            if y < zone_height:
                return "top"
            elif y < 2 * zone_height:
                return "center"
            else:
                return "bottom"
        else:
            if y < zone_height:
                return "top-right"
            elif y < 2 * zone_height:
                return "right"
            else:
                return "bottom-right"


def capitalize_first_letter(string: str) -> str:
    return re.sub(r"(\b\w)|('\w)|(/\w)", lambda x: x.group().upper(), string)


def get_spatial_relationship_between_two_objects(geom1, geom2):
    # spatial_relationship = ["inside", "intersecting", ]
    spatial = []
    if geom2.contains(geom1):
        return "contains"
    if geom1.contains(geom2):
        return "inside"
    if geom1.bounds[3] < geom2.bounds[1]:
        spatial.append("above")
    if geom1.bounds[1] > geom2.bounds[3]:
        spatial.append("below")
    if geom1.bounds[2] < geom2.bounds[0]:
        if "above" in spatial:
            spatial.remove("above")
            spatial.append("top-right")
        elif "below" in spatial:
            spatial.remove("below")
            spatial.append("bottom-right")
        else:
            spatial.append("right")

    if geom1.bounds[0] > geom2.bounds[2]:
        if "above" in spatial:
            spatial.remove("above")
            spatial.append("top-left")
        elif "below" in spatial:
            spatial.remove("below")
            spatial.append("bottom-left")
        else:
            spatial.append("left")
    if geom1.intersects(geom2):
        spatial.append("intersecting")
    if len(spatial) == 0:
        return None
    else:
        return ' '.join(spatial)
