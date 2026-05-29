import warnings
warnings.filterwarnings("ignore")

import os
import gc
import pickle
import torch
import torchvision.transforms as T
from tqdm import tqdm
from transformers import AdamW, DistilBertTokenizer
from sar_display import *
from ImageDataset import ImageDataset
from VQAModel import VQAModel

# ============================================================
# CONFIG
# ============================================================
pathS1S2      = os.environ.get("S1S2_ROOT", "path/to/s1s2_patches")  # local S1/S2 patches
load_weights  = False

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
IMAGENET_MEANS2 = [0.485, 0.456, 0.406, 0.485, 0.485, 0.485, 0.485, 0.485, 0.485, 0.485]
IMAGENET_STDS2  = [0.229, 0.224, 0.225, 0.229, 0.229, 0.229, 0.229, 0.229, 0.229, 0.229]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transformS2 = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEANS2, std=IMAGENET_STDS2),
])

ratio_images_to_use = 1
number_outputs      = 1000

hyper_params = {
    "optimizer":           "AdamW",
    "batch_size":          80,
    "num_epochs":          3,
    "learning_rate":       3e-5,
    "ratio_images_to_use": ratio_images_to_use,
    "number_outputs":      number_outputs,
    "nm_workers":          10,
}

if os.environ.get("DUMMY_RUN"):
    hyper_params.update({"batch_size": 1, "num_epochs": 1, "nm_workers": 0})
    print("Running in dummy mode")

# ============================================================
# Model
# ============================================================
model  = VQAModel(activate_bdo=True, activate_s1=False, activate_s2=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model  = model.to(device)

if hyper_params['optimizer'] == "AdamW":
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_params['learning_rate'])
elif hyper_params['optimizer'] == "Adam":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_params['learning_rate'])
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_params['learning_rate'])

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

losses       = {"train": [], "validation": []}
number_iters = {"train": 0, "validation": 0}

if load_weights:
    checkpoint = torch.load("MM-RSVQA/BigModelV2_5.tar")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses = checkpoint["losses"]

gc.collect()

# ============================================================
# Train
# ============================================================
def trainFunction():
    dataset = {}
    loaders = {}

    dataset["train"] = ImageDataset(
        split="train",
        pathS1S2=pathS1S2,
        train=True,
        ratio_images_to_use=ratio_images_to_use,
        transform=transform,
        transformS2=transformS2,
        number_outputs=number_outputs,
        activate_bdo=True,
        activate_s1=False,
        activate_s2=False,
    )
    dataset["validation"] = ImageDataset(
        split="val",
        pathS1S2=pathS1S2,
        train=False,
        ratio_images_to_use=ratio_images_to_use,
        transform=transform,
        transformS2=transformS2,
        selected_answers=dataset["train"].selected_answers,
        activate_bdo=True,
        activate_s1=False,
        activate_s2=False,
    )

    # Save selected_answers for test
    with open("selected_answers.pkl", "wb") as f:
        pickle.dump(dataset["train"].selected_answers, f)

    for step in ["train", "validation"]:
        loaders[step] = torch.utils.data.DataLoader(
            dataset[step],
            batch_size=hyper_params['batch_size'],
            shuffle=(step == "train"),
            num_workers=hyper_params["nm_workers"],
            drop_last=True,
        )

    len_dataset = {s: len(dataset[s]) for s in ["train", "validation"]}

    for epoch in range(len(losses["validation"]), hyper_params['num_epochs']):
        print(f"Starting epoch {epoch}:")
        accuracies       = {"train": {}, "validation": {}}
        numQuestionByCat = {"train": {}, "validation": {}}

        for step in ["train", "validation"]:
            if len(losses["train"]) > len(losses["validation"]) and step == "train":
                continue
            model.train() if step == "train" else model.eval()
            total_correct = 0

            with torch.set_grad_enabled(step == "train"):
                losses[step].append(0)
                print(step)

                for i, (data, indices) in enumerate(tqdm(loaders[step])):
                    optimizer.zero_grad()

                    question, answer, *images, type_str = data
                    imgBDO = imgS2 = imgS1 = None
                    if len(images) == 1:
                        imgBDO = images[0]
                    elif len(images) == 2:
                        imgBDO, imgS2 = images
                    elif len(images) == 3:
                        imgBDO, imgS2, imgS1 = images

                    question = {k: v.to(device) for k, v in question.items()}
                    answer   = answer.to(device).long()
                    if imgBDO is not None: imgBDO = imgBDO.float().to(device)
                    if imgS2  is not None: imgS2  = imgS2.float().to(device)
                    if imgS1  is not None: imgS1  = imgS1.float().to(device)

                    pred = model(imageOrtho=imgBDO, imageS2=imgS2, imageS1=imgS1, question=question)
                    loss = criterion(pred, answer)

                    if step == "train":
                        loss.backward()
                        optimizer.step()

                    losses[step][epoch] += loss.cpu().item()
                    _, predicted = torch.max(pred.data, 1)
                    total_correct += (predicted == answer.data).sum()

                    for z in range(hyper_params["batch_size"]):
                        t = type_str[z]
                        numQuestionByCat[step][t] = numQuestionByCat[step].get(t, 0) + 1
                        accuracies[step].setdefault(t, 0)
                        if predicted[z] == answer[z]:
                            accuracies[step][t] += 1

                    del question, answer, imgBDO, pred
                    torch.cuda.empty_cache()

                losses[step][epoch] /= len_dataset[step]
                print(f"{step} correct answers = {accuracies[step]}")
                print(f"{step} numQuestionByCat = {numQuestionByCat[step]}")
                print(f"{step} loss = {losses[step][epoch]}")

        os.makedirs('MM-RSVQA', exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "losses":           losses,
            "number_iters":     number_iters,
            "epoch":            epoch,
            "accuracies":       accuracies,
            "numQuestByCat":    numQuestionByCat,
        }, f"MM-RSVQA/BigModelV2_{epoch}.tar")

    print('Training done.')

# ============================================================
# Test
# ============================================================
def testFunction():
    test_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    try:
        with open("selected_answers.pkl", "rb") as f:
            selected_answers = pickle.load(f)
    except FileNotFoundError:
        print("Warning: selected_answers.pkl not found.")
        selected_answers = []

    dataset_test = ImageDataset(
        split="test",
        pathS1S2=pathS1S2,
        train=False,
        ratio_images_to_use=ratio_images_to_use,
        transform=transform,
        transformS2=transformS2,
        selected_answers=selected_answers,
        activate_bdo=True,
        activate_s1=False,
        activate_s2=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=hyper_params["batch_size"],
        shuffle=False,
        num_workers=hyper_params["nm_workers"],
        drop_last=True,
    )

    model.eval()
    accuracies       = {}
    numQuestionByCat = {}
    predictions      = []
    answers_list     = []

    with torch.no_grad():
        for i, (data, indices) in enumerate(tqdm(test_loader)):
            question, answer, *images, type_str = data
            imgBDO = imgS2 = imgS1 = None
            if len(images) == 1:
                imgBDO = images[0]
            elif len(images) == 2:
                imgBDO, imgS2 = images
            elif len(images) == 3:
                imgBDO, imgS2, imgS1 = images

            question = {k: v.to(device) for k, v in question.items()}
            answer   = answer.to(device).long()
            if imgBDO is not None: imgBDO = imgBDO.float().to(device)
            if imgS2  is not None: imgS2  = imgS2.float().to(device)
            if imgS1  is not None: imgS1  = imgS1.float().to(device)

            pred = model(imageOrtho=imgBDO, imageS2=imgS2, imageS1=imgS1, question=question)
            _, predicted = torch.max(pred.data, 1)

            for z in range(hyper_params["batch_size"]):
                t = type_str[z]
                numQuestionByCat[t] = numQuestionByCat.get(t, 0) + 1
                accuracies.setdefault(t, 0)
                if predicted[z] == answer[z]:
                    accuracies[t] += 1

            predictions  += predicted.tolist()
            answers_list += answer.tolist()

            os.makedirs("MM-RSVQA", exist_ok=True)
            with open("MM-RSVQA/predictionsTest.txt", "a") as f:
                for x in range(hyper_params["batch_size"]):
                    f.write(f"-----Batch {i} item {x}------\n")
                    f.write(f"Pred   : {selected_answers[predicted[x]] if selected_answers else predicted[x]}\n")
                    f.write(f"Answer : {selected_answers[answer[x]] if selected_answers else answer[x]}\n")
                    f.write(f"Question : {test_tokenizer.decode(question['input_ids'][x], skip_special_tokens=True)}\n")
                    f.write(f"Type : {type_str[x]}\n")

    total_correct  = sum(accuracies.values())
    total_total    = sum(numQuestionByCat.values())
    overall_acc    = total_correct / total_total
    cat_accs       = {c: accuracies[c] / numQuestionByCat[c] for c in accuracies}
    average_acc    = sum(cat_accs.values()) / len(cat_accs)

    print(f"Overall Accuracy:  {overall_acc:.2%}")
    print(f"Average Accuracy:  {average_acc:.2%}")

    with open("MM-RSVQA/predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
    with open("MM-RSVQA/answers.pkl", "wb") as f:
        pickle.dump(answers_list, f)

    print('Test done.')


if __name__ == '__main__':
    trainFunction()
    testFunction()
