import tqdm
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, SwinModel
from PIL import Image
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch
import os
from seg_loader2 import SegLoader
import torchvision.transforms as T
from torch.optim import AdamW, Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import itertools
import wandb


def plot_curves(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(25, 20))

    # plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    # plt.savefig("files_for_metrics5/losses.svg")
    #plt.show()


channels = ['BATIMENT',
            'CIMETIERE',
            'TERRAIN_DE_SPORT',
            'RESERVOIR',
            'PYLONE',
            'CONSTRUCTION_SURFACIQUE',
            'ZONE_D_ESTRAN',
            'ZONE_DE_VEGETATION',
            'SURFACE_HYDROGRAPHIQUE',
            'AERODROME',
            'EQUIPEMENT_DE_TRANSPORT',
            'TRONCON_DE_ROUTE',
            'TRONCON_DE_VOIE_FERREE',
            'FORET_PUBLIQUE',
            'PARC_OU_RESERVE',
            'TOPONYMIE_SERVICES_ET_ACTIVITES',
            'Overall']


class CustomUnetForSemanticSegmentation(UperNetForSemanticSegmentation):
    def __init__(self, config, num_classes=2):
        super(CustomUnetForSemanticSegmentation, self).__init__(config)
        self.num_classes = num_classes
        self.decode_head1 = torch.nn.Conv2d(150, num_classes, kernel_size=1)

    def forward(self, inputs):
        outputs = super().forward(inputs)
        logits = self.decode_head1(outputs.logits)
        logits = nn.functional.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        return logits, outputs.logits


def calculate_iou(output, target):
    output_softmax = torch.nn.functional.softmax(output, dim=1)
    output_softmax = (output_softmax >= 0.5).float()

    target_binary = target.float()
    intersection = torch.sum(output_softmax * target_binary, dim=(2, 3))
    union = torch.sum(output_softmax + target_binary - output_softmax * target_binary, dim=(2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    mean_iou = torch.mean(iou, dim=(0, 1))

    return mean_iou


def plot_precision_recall_curves(data_dict, epoch=0, colors=None):
    plt.figure(figsize=(25, 20))

    markers = itertools.cycle(('o', 's', '^', 'D', 'v', 'p', '*', 'H', '+', 'x', '<', '>', '8', '.', ',', '1', '_'))

    class_labels = list(range(17))

    for class_idx in range(16):
        precisions = []
        recalls = []
        for threshold, precision_recall_list in data_dict.items():
            precision, recall = precision_recall_list[class_idx]
            precisions.append(precision)
            recalls.append(recall)
        marker = next(markers)
        color = colors[class_idx] if colors and class_idx in colors else None
        plt.plot(recalls, precisions, marker=marker, color=color, label=f'{channels[class_idx]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different Classes and Thresholds')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.grid(True)
    # plt.savefig(f"files_for_metrics5/{epoch+5}_prec_recall.svg")
    #plt.show()


def calculate_metrics(true_masks, predicted_masks, th):
    # output_softmax = torch.nn.functional.softmax(predicted_masks, dim=1)

    output_softmax = (predicted_masks >= th).float()
    predicted_masks_flat = output_softmax.view(-1)
    true_masks_flat = true_masks.contiguous().view(-1)

    tp = torch.sum((true_masks_flat == 1) & (predicted_masks_flat == 1)).item()
    fp = torch.sum((true_masks_flat == 0) & (predicted_masks_flat == 1)).item()
    fn = torch.sum((true_masks_flat == 1) & (predicted_masks_flat == 0)).item()
    tn = torch.sum((true_masks_flat == 0) & (predicted_masks_flat == 0)).item()

    return tp, fp, fn, tn


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def plot_outputs(inputs, outputs, labels,outputs_table):
    denormalize = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1 / i for i in IMAGENET_STD]),
        T.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1, 1, 1]),
    ])
    denormalized_img = denormalize(inputs[0])
    denormalized_numpy_img = denormalized_img.cpu().numpy().transpose(1, 2, 0)

    plt.imshow(denormalized_numpy_img)
    plt.axis('off')  # Hide axes for better visualization
    plt.savefig("input.png")
    plt.clf()
    inp = Image.open("input.png")
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    # Iterate through the channels and visualize each segmentation mask
    for i, ax in enumerate(axes.flat):
        ax.imshow(labels[0][i], cmap='jet')  # Using 'jet' colormap for visualization
        ax.axis('off')
        ax.set_title(f'{channels[i]}')

    # Adjust layout to prevent overlapping titles
    plt.savefig(f"target.png")
    targ = Image.open("target.png")
    plt.tight_layout()
    #plt.show()
    plt.clf()

    output = (outputs.cpu())

    output = (output[0] >= 0.5).float()

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(output[i], cmap='jet')
        ax.axis('off')
        ax.set_title(f'{channels[i]}')

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()
    plt.savefig(f"output.png")
    outp = Image.open("output.png")
    #plt.show()
    plt.clf()
    plt.close()
    outputs_table.add_data(wandb.Image(inp),wandb.Image(outp),wandb.Image(targ))
    #plt.show()


def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, resume=False):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                               collate_fn=my_collate,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                             collate_fn=my_collate,
                                             drop_last=True)

    wandb.init(
        project="segmentation-test",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss"
        })
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    resume = False
    if resume:
        checkpoint = torch.load("files_for_metrics5/SwinUpernetLogits_proj_BCELoss_to_5epochs4.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.cuda()
    valLoss = []
    trainLoss = []
    # trainLoss = [0.1268246598019292, 0.07955695619703614, 0.07413623979613537, 0.0707764935852286, 0.06845813805213281]
    # valLoss = [0.6910444775667791, 0.07953088119648935, 0.07093534901989954, 0.06996316353457131, 0.06729416028351828]
    val_ious = []
    train_ious = []
    # thresholds = np.logspace(-3, 0, 10)
    thresholds = [0.5]


    for epoch in tqdm.tqdm(range(num_epochs)):
        outputs_table = wandb.Table(["input", "output", "target"])

        metrics_table = wandb.Table(columns=["class", "precision", "recall", "f1", "iou"])
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        channel_metrics = {}  # { 0 : [], 0.1:[], 0.2 : [] }
        prec_rec_classes_to_plot = []
        values_for_th = {}
        other_values_for_th = {}
        with torch.no_grad():
            model.eval()
            runningLoss = 0
            for data in tqdm.tqdm(val_loader):
                inputs, labels = data
                """if epoch == 0:
                    Weights = torch.ones_like(labels, dtype=float)
                    for i in range(16):
                        Weights[:, i, :, :] *= weights[i]"""
                inputs = Variable(inputs).cuda()
                outputs, fp = model(inputs)
                segmap = Variable(labels.float()).cuda()
                loss = criterion(outputs, segmap)
                # loss = torch.mean(Weights.cuda() * loss)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                runningLoss += loss.cpu().item() * inputs.shape[0]
                if epoch % 5 == 0:
                    plot_outputs(inputs, outputs, labels, outputs_table)
                for j in range(16):
                    channel_output = outputs[:, j:j + 1, :, :]

                    channel_label = labels[:, j:j + 1, :, :]

                    for th in thresholds:
                        tp, fp, fn, tn = calculate_metrics(channel_label.float(), channel_output.cpu(), th)
                        if th not in channel_metrics.keys():
                            channel_metrics[th] = []
                        if len(channel_metrics[th]) < 16:
                            channel_metrics[th].append({
                                'channel': channels[j],
                                'tp': tp,
                                'fp': fp,
                                'fn': fn,
                                'tn': tn
                            })
                        else:
                            channel_metrics[th][j]['tp'] += tp
                            channel_metrics[th][j]['fp'] += fp
                            channel_metrics[th][j]['fn'] += fn
                            channel_metrics[th][j]['tn'] += tn

            for th, channel_metric_list in channel_metrics.items():
                prec_rec_classes_to_plot = []
                f1_iou_accuracy_to_keep = []
                for metric in channel_metric_list:
                    tp = metric["tp"]
                    fp = metric["fp"]
                    fn = metric["fn"]
                    tn = metric["tn"]
                    IoU = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    prec_rec_classes_to_plot.append((precision, recall))
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    f1_iou_accuracy_to_keep.append((f1, IoU, accuracy))
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn
                    metrics_table.add_data(metric["channel"],precision,recall,f1,IoU)

                IoU = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
                prec_rec_classes_to_plot.append((precision, recall))
                values_for_th[th] = prec_rec_classes_to_plot
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_iou_accuracy_to_keep.append((f1, IoU, accuracy))
                other_values_for_th[th] = f1_iou_accuracy_to_keep
                wandb.log({"Avg_Precision":precision,"Avg_Recall":recall})
            #plot_precision_recall_curves(values_for_th, epoch)

        valLoss.append(runningLoss / len(validate_dataset))
        if epoch%5==0:
            wandb.log({f"outputs_{epoch}": outputs_table})

        print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
        model.train()
        runningLoss = 0
        iou = 0
        for data in tqdm.tqdm((train_loader)):
            inputs, labels = data

            inputs = Variable(inputs).cuda()
            outputs, fp = model(inputs)
            segmap = Variable(labels.float()).cuda()

            # Reshape the predicted logits and target masks to [batch_size * height * width, num_classes]

            loss = criterion(outputs, segmap)
            # loss = torch.mean(Weights.cuda() * loss)
            runningLoss += loss.cpu().item() * inputs.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_ious.append(iou / len(train_dataset))
        trainLoss.append(runningLoss / len(train_dataset))
        if len(trainLoss) > 1:
            plot_curves(trainLoss, valLoss)
        print('epoch #%d train loss: %.3f' % (epoch, trainLoss[epoch]))
        wandb.log({"Train": trainLoss[epoch]})
        wandb.log({"Validation": valLoss[epoch]})
        wandb.log( {f"{epoch}_Metrics":metrics_table})
        """torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"files_for_metrics5/SwinUpernetLogits_proj_BCELoss_to_5epochs{epoch}.pt")"""


def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)

if __name__ == '__main__':
    data_path = 'D:\\data\\BDOrtho+Sentinel2\\test_version8'
    images_path = os.path.join(data_path)
    q_and_a_path = os.path.join(data_path, 'Q&A')

    imagesTrainJSON = os.path.join(q_and_a_path, 'train.json')
    imagesValJSON = os.path.join(q_and_a_path, 'val.json')
    ratio_images_to_use = 1
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    patch_size = 1000
    train_dataset = SegLoader(images_path, imagesTrainJSON, train=True,
                              ratio_images_to_use=ratio_images_to_use, transform=transform,
                              patch_size=patch_size)
    val_dataset = SegLoader(images_path, imagesValJSON, train=False,
                            ratio_images_to_use=ratio_images_to_use, transform=transform,
                            patch_size=patch_size)

    model = CustomUnetForSemanticSegmentation(
        UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-base").config, 16)
    # model.auxiliary_head.classifier = nn.Conv2d(256, 16, kernel_size=(31, 31), stride=(16, 16), padding=(7, 7))
    # model.decode_head.classifier =nn.Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1))

    # model.load_state_dict(torch.load("files_for_metrics5/SwinUpernetLogits_proj_BCELoss_to_5epochs4.pth"))
    model.cuda()
    resume = False

    train(model, train_dataset, val_dataset, 2, 100, 1e-5, resume)
