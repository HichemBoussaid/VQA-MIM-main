import warnings

warnings.filterwarnings("ignore")
import os.path
from torch.utils.data import Dataset
import torchvision.transforms as T
import pickle
from sar_display import *
from transformers import AdamW
from ImageDataset import *
from VQAModel import *

data_files = {
    "train": {
        "images": "split_train_images.json",
        "questions": "split_train_questions.json",
        "answers": "split_train_answers.json"
    },
    "validation": {
        "images": "split_val_images.json",
        "questions": "split_val_questions.json",
        "answers": "split_val_answers.json"
    },
    "test": {
        "images": "split_test_images.json",
        "questions": "split_test_questions.json",
        "answers": "split_test_answers.json"
    },
    "all": {
        "images": "allImagesFull.json",
        "questions": "allQuestions.json",
        "answers": "allAnswers.json"
    }
}

pathBdO = os.environ.get("BDO_ROOT", "path/to/bdortho")
pathS2 = os.environ.get("S2_ROOT", "path/to/sentinel2")
pathS1 = os.environ.get("S1_ROOT", "path/to/sentinel1")
load_weights = False
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]  # what do i do?

IMAGENET_MEANS2 = [0.485, 0.456, 0.406, 0.485, 0.485, 0.485, 0.485, 0.485, 0.485, 0.485]
IMAGENET_STDS2 = [0.229, 0.224, 0.225, 0.229, 0.229, 0.229, 0.229, 0.229, 0.229, 0.229]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transformS2 = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEANS2, std=IMAGENET_STDS2),
])

dataset = {}
ratio_images_to_use = 1
encoder_answers = None

# Try to load selected_answers; support dummy data or missing file
selected_answers_path = os.path.join(os.path.dirname(__file__), "..", "selected_answers.pkl")
if os.path.exists(selected_answers_path):
    with open(selected_answers_path, 'rb') as file:
        selected_answers = pickle.load(file)
else:
    # Fall back to common/expected location or empty list for smoke tests
    try:
        with open("selected_answers.pkl", 'rb') as file:
            selected_answers = pickle.load(file)
    except FileNotFoundError:
        print("Warning: selected_answers.pkl not found. Using empty list (dummy run only).")
        selected_answers = []
gc.collect()

model = VQAModel(activate_bdo=True, activate_s1=False, activate_s2=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = model.to(device)
print('ratio_images_to_use =', ratio_images_to_use)
number_outputs = 1000
hyper_params = {
    "optimizer": "AdamW",
    "batch_size": 80,
    "num_epochs": 3,
    "learning_rate": 3 * 10 ** -5,
    "ratio_images_to_use": ratio_images_to_use,
    "number_outputs": number_outputs,
    "nm_workers": 10

}

# Compact settings for dummy smoke tests (set DUMMY_RUN=1)
if os.environ.get("DUMMY_RUN"):
    hyper_params.update({"batch_size": 1, "num_epochs": 1, "nm_workers": 0})
    print("Running in dummy mode: batch_size=1, num_epochs=1, nm_workers=0")


def ensure_paths_configured():
    missing = []
    if not pathBdO or "path/to/bdortho" in pathBdO:
        missing.append("pathBdO (set BDO_ROOT or edit MM-RSVQA.py)")
    if getattr(model, "activate_s2", False):
        if not pathS2 or "path/to/sentinel2" in pathS2:
            missing.append("pathS2 (set S2_ROOT or edit MM-RSVQA.py)")
    if getattr(model, "activate_s1", False):
        if not pathS1 or "path/to/sentinel1" in pathS1:
            missing.append("pathS1 (set S1_ROOT or edit MM-RSVQA.py)")
    if missing:
        raise ValueError("Missing data roots: " + ", ".join(missing))

print(hyper_params)
if hyper_params['optimizer'] == "Adam":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyper_params['learning_rate'])
elif hyper_params["optimizer"] == "AdamW":
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=hyper_params['learning_rate'])
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=hyper_params['learning_rate'])
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

loaders = {}

losses = {"train": [], "validation": []}
number_iters = {"train": 0, "validation": 0}
if load_weights:
    model_name = "BigModelV2_5.tar"
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    losses = checkpoint["losses"]
part = 1


def testFunction():
    from transformers import DistilBertTokenizer
    model_path = "distilbert/distilbert-base-uncased"
    test_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    dataset["test"] = ImageDataset(
        data_files["test"]["images"],
        encoder_answers,
        data_files["test"]["questions"],
        data_files["test"]["answers"],
        pathBdO,
        pathS1,
        pathS2,
        patch_sizeBDO=1000,
        patch_sizeS2=100,
        train=False,
        ratio_images_to_use=ratio_images_to_use,
        transformS2=transformS2,
        transform=transform,
        selected_answers=selected_answers,
        activate_bdo=True,
        activate_s2=False,
        activate_s1=False,
        part=part,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=hyper_params["batch_size"],
        shuffle=False,
        num_workers=hyper_params["nm_workers"],
        drop_last=True,
    )
    model.eval()
    accuracies = {}
    numQuestionByCat = {}
    predictions = []
    answers = []
    myindex = 0
    with torch.no_grad():
        for i, (data, indices) in enumerate(tqdm(test_loader)):
            optimizer.zero_grad()  # Reset gradients

            # Support variable modality combinations
            question, answer, *images, type_str = data
            imgBDO = imgS2 = imgS1 = None
            if len(images) == 1:
                imgBDO = images[0]
            elif len(images) == 2:
                imgBDO, imgS2 = images
            elif len(images) == 3:
                imgBDO, imgS2, imgS1 = images

            question = {k: v.to(device) for k, v in question.items()}
            answer = answer.to(device).long()
            if imgBDO is not None:
                imgBDO = imgBDO.float().to(device)
            if imgS2 is not None:
                imgS2 = imgS2.float().to(device)
            if imgS1 is not None:
                imgS1 = imgS1.float().to(device)

            pred = model(imageOrtho=imgBDO, imageS2=imgS2, imageS1=imgS1, question=question)

            _, predicted = torch.max(pred.data, 1)
            for z in range(hyper_params["batch_size"]):
                if type_str[z] not in numQuestionByCat:
                    numQuestionByCat[type_str[z]] = 1
                else:
                    numQuestionByCat[type_str[z]] += 1
                if type_str[z] not in accuracies:
                    accuracies[type_str[z]] = 0
                if predicted[z] == answer[z]:
                    accuracies[type_str[z]] += 1
            predictions += predicted
            answers += answer
            for x in range(hyper_params["batch_size"]):
                os.makedirs(f"MM-RSVQA/{indices[x]}", exist_ok=True)
                with open("MM-RSVQA/predictionsTest.txt", "a") as f:
                    f.write(f"-----Batch {i} item {x}------\n")
                    f.write(f"Pred : {selected_answers[predicted[x]]}\n")
                    answer_text = selected_answers[answer[x]] if answer[x] != -1 else "None"
                    f.write(f"Answer : {answer_text}\n")
                    f.write(
                        f"Question : {test_tokenizer.decode(question['input_ids'][x], skip_special_tokens=True)}\n"
                    )
                    f.write(f"Question Type : {type_str[x]}\n")

                save_images(
                    imgBDO if imgBDO is not None else None,
                    imgS1[x] if imgS1 is not None else None,
                    imgS2[x] if imgS2 is not None else None,
                    save_path="MM-RSVQA",
                    batch_index=f"{indices[x]}",
                    save_as_tif=False,
                )
                myindex += 1
                if myindex >= 100:
                    break
        print("accuracies", accuracies)
        print("numQuestionByCat", numQuestionByCat)
        correct_answers = accuracies

        total_answers = numQuestionByCat
        total_correct = sum(correct_answers.values())
        total_total = sum(total_answers.values())
        overall_accuracy = total_correct / total_total
        category_accuracies = {category: correct_answers[category] / total_answers[category] for category in
                               correct_answers}
        average_accuracy = sum(category_accuracies.values()) / len(category_accuracies)
        print("Average Accuracy: {:.2%}".format(overall_accuracy))
        print("Overall Accuracy: {:.2%}".format(average_accuracy))
        # File path
        average_accuracy_str = "Average Accuracy: {:.2%}".format(average_accuracy)
        overall_accuracy_str = "Overall Accuracy: {:.2%}".format(overall_accuracy)
        predictions_file_path = f"MM-RSVQA/predictions_v3_p{part}.pkl"
        answers_file_path = f"MM-RSVQA/answers_v3_p{part}.pkl"

        with open(predictions_file_path, "wb") as f:
            pickle.dump(predictions, f)
        with open(answers_file_path, "wb") as f:
            pickle.dump(answers, f)


gc.collect()


def trainFunction():
    dataset["train"] = ImageDataset(data_files["train"]["images"], encoder_answers, data_files["train"]["questions"],
                                    data_files["train"]["answers"], pathBdO, pathS1, pathS2, patch_sizeBDO=1000,
                                    patch_sizeS2=100,
                                    patch_sizeS1=200,
                                    train=True, ratio_images_to_use=ratio_images_to_use, transformS2=transformS2,
                                    transform=transform, activate_bdo=True, activate_s1=False,
                                    activate_s2=False)  # 0.045
    dataset["validation"] = ImageDataset(data_files["validation"]["images"], encoder_answers,
                                         data_files["validation"]["questions"], data_files["validation"]["answers"],
                                         pathBdO, pathS1, pathS2, patch_sizeBDO=1000, patch_sizeS2=100,
                                         patch_sizeS1=200,
                                         train=False,
                                         ratio_images_to_use=ratio_images_to_use, transformS2=transformS2,
                                         transform=transform,
                                         selected_answers=dataset["train"].selected_answers, activate_bdo=True,
                                         activate_s1=False, activate_s2=False)
    for step in ["train", "validation"]:
        train = step == "train"
        loaders[step] = torch.utils.data.DataLoader(dataset[step], batch_size=hyper_params['batch_size'], shuffle=train,
                                                    num_workers=hyper_params["nm_workers"], drop_last=True)
    len_dataset = {"validation": len(dataset["validation"]), "train": len(dataset["train"])}

    for epoch in range(len(losses["validation"]), hyper_params['num_epochs']):
        print(f"Starting epoch {epoch}:")
        accuracies = {"train": {}, "validation": {}}
        numQuestionByCat = {"train": {}, "validation": {}}
        for step in ["train", "validation", ]:
            if len(losses["train"]) > len(losses["validation"]) and step == "train":
                continue
            if step == "train":
                model.train()
            else:
                model.eval()
            total_correct = 0
            with torch.set_grad_enabled(step == "train"):
                losses[step].append(0)
                print(step)

                for i, (data, indices) in enumerate(tqdm(loaders[step])):
                    optimizer.zero_grad()  # Reset gradients

                    # Support variable modality combinations
                    question, answer, *images, type_str = data
                    imgBDO = imgS2 = imgS1 = None
                    if len(images) == 1:
                        imgBDO = images[0]
                    elif len(images) == 2:
                        imgBDO, imgS2 = images
                    elif len(images) == 3:
                        imgBDO, imgS2, imgS1 = images

                    question = {k: v.to(device) for k, v in question.items()}
                    answer = answer.to(device).long()
                    if imgBDO is not None:
                        imgBDO = imgBDO.float().to(device)
                    if imgS2 is not None:
                        imgS2 = imgS2.float().to(device)
                    if imgS1 is not None:
                        imgS1 = imgS1.float().to(device)

                    pred = model(imageOrtho=imgBDO, imageS2=imgS2, imageS1=imgS1, question=question)

                    loss = criterion(pred, answer)
                    if step == "train":
                        loss.backward()
                        optimizer.step()
                    losses[step][epoch] += loss.cpu().item()  # Use non in-place operation

                    _, predicted = torch.max(pred.data, 1)
                    batch_correct = (predicted == answer.data).sum()
                    total_correct += batch_correct

                    for z in range(hyper_params["batch_size"]):
                        if type_str[z] not in numQuestionByCat[step].keys():
                            numQuestionByCat[step][type_str[z]] = 1
                        else:
                            numQuestionByCat[step][type_str[z]] += 1
                        if type_str[z] not in accuracies[step].keys():
                            accuracies[step][type_str[z]] = 0
                        if predicted[z] == answer[z]:
                            accuracies[step][type_str[z]] = accuracies[step][
                                                                type_str[z]] + 1  # Use non in-place operation

                    # Delete tensors and free up GPU memory
                    del question, answer, imgBDO, pred
                    torch.cuda.empty_cache()

                losses[step][epoch] /= len_dataset[step]
                print(f"{step} correct answers = {accuracies[step]}")
                print(f"{step} number of questions by category = {numQuestionByCat[step]}")
                print(f"{step} loss = {losses[step][epoch]}")
            if epoch > 0:
                plt.figure()
                handles = []
                handles.append(plt.plot(losses["train"], label="train")[0])
                handles.append(plt.plot(losses["validation"], label="validation")[0])
                plt.legend(handles=handles)
                os.makedirs('MM-RSVQA', exist_ok=True)
                save_path = os.path.join('MM-RSVQA',
                                         f'plot_epoch_{epoch}.svg')
                save_path2 = os.path.join('MM-RSVQA',
                                          f'plot_epoch_{epoch}.png')
                plt.savefig(save_path)
                plt.savefig(save_path2)
                plt.close()
            os.makedirs('MM-RSVQA', exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "losses": losses,
                        "number_iters": number_iters, "epoch": epoch, "accuracies": accuracies,
                        "numQuestByCat": numQuestionByCat},
                       "MM-RSVQA/BigModelV2_" + str(epoch) + ".tar")
    print('the end')


if __name__ == '__main__':
    ensure_paths_configured()
    trainFunction()
    testFunction()
