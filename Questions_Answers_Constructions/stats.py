import os
import json
from collections import OrderedDict
import random

DATA_DIR = 'D:\\data\\BDOrtho+Sentinel2\\'
# for dept in ["75","92","93","94"]:
version = "test_version8\\Q&A\\qa_jsons"

target_dir = os.path.join(DATA_DIR, version)
with open(os.path.join(target_dir, "allQuestions_latest.json")) as fp:
    questions = json.load(fp)
with open(os.path.join(target_dir, "allAnswers_latest.json")) as fp:
    answers = json.load(fp)

allAnswers = {"count": {}, "presence": {}, "distance": {},
              "nearest": {"top-left of the image": 0, "top of the image": 0, "top-right of the image": 0,
                          'left of the image': 0, 'center of the image': 0, 'right of the image': 0,
                          "bottom-left of the image": 0, "bottom of the image": 0, "bottom-right of the image": 0}
    , "spatial_relation": {}, "less_more_equal": {},
              "smallest_largest": {}, "area": {}, "density": {}, "pres_flooding": {}, "pres_mountains": {},
              "lvl_flooding": {}, "urban": {}, "type_flooding": {},
              "clc": {}, "vegetation": {}, "water": {}, "department": {},
              "region": {}, 'percent_clc': {}}
custom_order = ["top-left", "above", "top-right", "left", "right", "bottom-left", "below", "bottom-right"]
vals = []
labels = []
patch_id = "1"
for q in questions["questions"]:
    ans = answers["answers"][q["answer_id"]]["answer"]

    if isinstance(ans, list):
        ans = ans[0]
    if q["type"] == "distance":
        ans = str(round(float(ans.split("m")[0]))) + "m"
    elif q["type"] == "area":
        ans = str(round(float(ans.split("m2")[0]))) + "m2"
    elif q["type"] in ["percent_clc","density"]:
        ans = str((float(ans.split("by m2")[0]))) + " by m2"
    if ans in allAnswers[q["type"]].keys():
        allAnswers[q["type"]][ans] += 1
    else:
        allAnswers[q["type"]][ans] = 1
for k1, v1 in allAnswers.items():
    if k1 == "count":
        sorted_data = {k: v for k, v in sorted(v1.items(), key=lambda item: int(item[0]))}
        allAnswers[k1] = sorted_data
    elif k1 == "area":
        sorted_data = {k: v for k, v in sorted(v1.items(), key=lambda item: float(item[0][:-2]))}
        allAnswers[k1] = sorted_data
    elif k1 == "distance":
        sorted_data = {k: v for k, v in sorted(v1.items(), key=lambda item: float(item[0][:-1]))}
        allAnswers[k1] = sorted_data
    elif k1 == "density":
        sorted_data = {k: v for k, v in sorted(v1.items(), key=lambda item: float(item[0][:-5]))}
        allAnswers[k1] = sorted_data
    elif k1 == "spatial_relation":
        sorted_dict = OrderedDict(sorted(v1.items(), key=lambda x: custom_order.index(x[0])))
        allAnswers[k1] = sorted_dict

for key, value in allAnswers.items():
    answers = list(value.keys())
    occurrences = list(value.values())

    # sorted_indices = sorted(range(len(occurrences)), key=lambda k: occurrences[k], reverse=True)
    # answers = [answers[i] for i in sorted_indices]
    # occurrences = [occurrences[i] for i in sorted_indices]

    answers = answers[:3707] + [''] * (3707 - len(answers))
    occurrences = occurrences[:3707] + [0] * (3707 - len(occurrences))

    vals.append(occurrences)
    labels.append(answers)

for k, v in allAnswers.items():
    if k == "count":
        v1 = [int(k1) for k1 in v.keys()]
    if k == "area":
        v1 = [int(k1.split("m2")[0]) for k1 in v.keys()]
    if k == "density":
        v1 = [float(k1.split("by m2")[0]) for k1 in v.keys()]
    if k == "distance":
        v1 = [float(k1.split("m")[0]) for k1 in v.keys()]
    if k in ["count", "area", "density", "distance"]:
        print(k)
        print(f"Min {min(v1)}, Max: {max(v1)}")
        print("/****/")
# Convert vals list to a numpy array
import numpy as np

vals_array = np.array(vals)

new_labels = []
for x in labels:
    new_labels += x

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(30, 30))

size = 0.3

group_labels = ['Count', "Presence", "Distance", "Nearest", "Relative Location", "Comparison", "Absolute Location",
                 'Area',"Density", "pres_flooding", "pres_mountains", "lvl_flooding", "urban", "type_flooding",
                "clc", "vegetation", "water", "department", "region", 'percent_clc']
answers_labels = new_labels

cmap = plt.get_cmap("tab20c")
vals = vals_array
print(answers_labels)
outer_colors = cmap(np.arange(8) * 2)
outer_colors = cmap([2, 3, 4, 5, 6, 8, 12, 0, 1])
inner_colors = []
for color in outer_colors:
    gradient_cmap = LinearSegmentedColormap.from_list('gradient', ['white', color])
    inner_colors.extend(gradient_cmap(np.linspace(0.2, 1, vals.shape[1])))

ax.pie(vals.sum(axis=1), radius=0.7, colors=outer_colors,
       wedgeprops=dict(width=0.5, edgecolor='w'), labels=group_labels, labeldistance=0.5, textprops={'fontsize': 21})

ax.pie(vals.flatten(), radius=1, colors=inner_colors,
       wedgeprops=dict(width=0.3, edgecolor='w'), labels=answers_labels, textprops={'fontsize': 20})

ax.set(aspect="equal", title='Total questions and answers distribution')
plt.savefig(f"stats30.svg")

plt.show()
patch_id = "0"
"""with open(os.path.join(target_dir, "allAnswers5.json")) as fp:
    answers = json.load(fp)"""
# print(dept,"******")
allImages = {}
for q in questions["questions"]:
    if q["patch_id"] in allImages.keys():
        allImages[q["patch_id"]] += 1
    else:
        allImages[q["patch_id"]] = 1
    """if q["patch_id"] == patch_id:

        print("Question: ", q["question"])
        print("Answer: ", answers["answers"][q["answer_id"]]["answer"])
        print("----")"""
