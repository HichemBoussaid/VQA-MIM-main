import warnings
import torch
import torch.nn as nn
from torchvision import models as torchmodels
from sar_display import *
from transformers import VisualBertConfig, VisualBertForQuestionAnswering
warnings.filterwarnings("ignore")


class LCEmodel(nn.Module):
    def __init__(self, input_size=120, number_outputs=61, requires_grad=False):
        super(LCEmodel, self).__init__()
        self.visual = torchmodels.resnet50()
        if requires_grad == False:
            for param in self.visual.parameters():
                param.requires_grad = False
        # to train theratio_ hidden layers
        elif requires_grad == True:
            for param in self.visual.parameters():
                param.requires_grad = True

        self.visual.fc = nn.Linear(self.visual.fc.in_features, number_outputs)
        self.sigm = nn.Sigmoid()
        self.visual_body = self.visual

    def forward(self, input_v):
        x1 = self.visual_body(input_v)
        x = self.sigm(x1)
        return x


class VQAModel(nn.Module):
    def __init__(self, input_size=512, glimpses=1, pretrained=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),activate_bdo=True, activate_s1=True, activate_s2=True):  # vocab_answers, glimpses=2
        super(VQAModel, self).__init__()
        self.activate_bdo = activate_bdo
        self.activate_s1 = activate_s1
        self.activate_s2 = activate_s2

        model_pathVBERT = "uclanlp/visualbert-vqa"
        config = VisualBertConfig.from_pretrained(model_pathVBERT)
        config.num_labels = 1000

        self.VBERT = VisualBertForQuestionAnswering.from_pretrained(model_pathVBERT, config=config,
                                                                    ignore_mismatched_sizes=True)

        for param in self.VBERT.parameters():  #
            param.requires_grad = True

        # BDORTHO
        if self.activate_bdo:
            self.visualORTHO = torchmodels.resnet152(weights=None)
            try:
                state_dict = torch.load("checkpoints/resnet152-f82ba261.pth")
                self.visualORTHO.load_state_dict(state_dict)
            except FileNotFoundError:
                warnings.warn("ResNet152 checkpoint not found; using randomly initialized weights for dummy/testing.")
            for param in self.visualORTHO.parameters():
                param.requires_grad = False
            self.visualORTHO = torch.nn.Sequential(*list(self.visualORTHO.children())[:-1])
        # Sentinel 1
        if self.activate_s1:
            self.visualS1 = LCEmodel(requires_grad=True)
            checkpoint = torch.load('weights/sar.tar')
            self.visualS1.load_state_dict(checkpoint['model_state_dict'])
            for param in self.visualS1.parameters():
                param.requires_grad = True

            extracted_layersS1 = list(self.visualS1.children())
            extracted_layersS1 = extracted_layersS1[0]
            filtered_layersS1 = []
            for layer in extracted_layersS1.children():
                if not isinstance(layer, (nn.Linear)):
                    filtered_layersS1.append(layer)
            self.visualS1 = nn.Sequential(*filtered_layersS1)

        # SENTINEL 2
        if self.activate_s2:
            self.visualS2 = LCEmodel(requires_grad=True)
            checkpoint = torch.load('weights/opt.tar')
            self.visualS2.load_state_dict(checkpoint['model_state_dict'])
            for param in self.visualS2.parameters():
                param.requires_grad = True
            extracted_layersS2 = list(self.visualS2.children())
            extracted_layersS2 = extracted_layersS2[0]
            filtered_layersS2 = []
            for layer in extracted_layersS2.children():
                if not isinstance(layer, nn.Linear):
                    filtered_layersS2.append(layer)

            self.visualS2 = nn.Sequential(*filtered_layersS2)
            pretrained_weights = self.visualS2[0].weight.clone()
            self.visualS2[0] = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize the new conv layer weights with the pre-trained weights for the first 3 channels
            with torch.no_grad():
                self.visualS2[0].weight[:, :3, :, :] = pretrained_weights
            # Copy the weights from the red channel to the remaining 7 channels
            with torch.no_grad():
                self.visualS2[0].weight[:, 3:, :, :] = pretrained_weights[:, 0:1, :, :]

    def _questionPart(self, input_q):
        q = self.seq2vec(**input_q)
        attention_mask = input_q['attention_mask']
        expanded_attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, q[0].size(dim=2))  # 768 why 768?
        q = torch.sum(q[0] * expanded_attention_mask.float(), dim=1) / torch.sum(expanded_attention_mask,
                                                                                 dim=1)  # weighted
        x_q = nn.Tanh()(q)
        return x_q

    def forward(self, imageOrtho=None, imageS1=None, imageS2=None, question=None):
        tensors_to_concat = []

        if self.activate_bdo:
            ortho_feat = self.visualORTHO(imageOrtho)
            v_ortho = ortho_feat / (ortho_feat.norm(p=2, dim=1, keepdim=True).expand_as(ortho_feat) + 1e-8)
            visual_embeds = v_ortho.view(-1, 1, 2048)
            tensors_to_concat.append(visual_embeds)

        if self.activate_s1:
            S1_feat = self.visualS1(imageS1)
            v_S1 = S1_feat / (S1_feat.norm(p=2, dim=1, keepdim=True).expand_as(S1_feat) + 1e-8)
            v_S1 = v_S1.reshape(v_S1.shape[0], -1, v_S1.shape[1])
            tensors_to_concat.append(v_S1)

        if self.activate_s2:
            S2_feat = self.visualS2(imageS2)
            v_S2 = S2_feat / (S2_feat.norm(p=2, dim=1, keepdim=True).expand_as(S2_feat) + 1e-8)
            v_S2 = v_S2.reshape(v_S2.shape[0], -1, v_S2.shape[1])
            tensors_to_concat.append(v_S2)

        # Check the number of tensors to concatenate
        if len(tensors_to_concat) == 1:
            concatenated_vis_embeds = tensors_to_concat[0]
        elif len(tensors_to_concat) > 1:
            concatenated_vis_embeds = torch.cat(tensors_to_concat, dim=1)
        else:
            raise ValueError("You need at least one image modality")

        device = concatenated_vis_embeds.device
        visual_token_type_ids = torch.ones(concatenated_vis_embeds.shape[:-1], device=device).long()
        visual_attention_mask = torch.ones_like(concatenated_vis_embeds[..., 0])

        question.update(
            {
                "visual_embeds": concatenated_vis_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        outputf = self.VBERT(**question)
        return outputf.logits
