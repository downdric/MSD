import torch
import tqdm
import os
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch import optim, nn
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F
from queue import Queue
import scipy.stats as st
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import cv2
import json
def read_json(path):
    with open(path,"r",encoding = 'utf-8') as f:
        data = json.load(f)
    return data
def write_json(path,data):
    with open(path,"w",encoding = 'utf-8') as f:
        json.dump(data,f)
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class MultiModalDataset(Dataset):
    def __init__(self, text_tools, vision_transforms, args, mode):
        self.args = args
        self.vision_transform = vision_transforms[mode]
        self.mode = mode
        self.text_arr, self.img_path, self.label, self.idx2file = self.init_data()

    def init_data(self):
        if self.mode == 'train':
            text_path = self.args.train_text_path
            vision_path = self.args.train_image_path
        else:
            text_path = self.args.test_text_path
            vision_path = self.args.test_image_path

        text_arr, img_path, labels, idx2file = {}, {}, {}, []
        skip_words = ['exgag', 'sarcasm', 'sarcastic', '<url>', 'reposting', 'joke', 'humor', 'humour', 'jokes', 'irony', 'ironic']
        for line in open(text_path, 'r').readlines():
            content = eval(line)
            file_name, text, label = content[0], content[1], content[2]
            flag = False
            for skip_word in skip_words:
                if skip_word in content[1]: flag = True
            if flag: continue

            cur_img_path = os.path.join(vision_path, file_name+'.jpg')
            if not os.path.exists(cur_img_path):
                print(file_name)
                continue

            text_arr[file_name], labels[file_name] = text, label
            img_path[file_name] = os.path.join(vision_path, file_name+'.jpg')
            idx2file.append(file_name)
        return text_arr, img_path, labels, idx2file

    def __getitem__(self, idx):
        file_name = self.idx2file[idx]
        text = self.text_arr[file_name]
        img_path = self.img_path[file_name]
        label = self.label[file_name]

        img = Image.open(img_path).convert("RGB")
        img = self.vision_transform(img)
        return file_name, img, text, label

    def __len__(self):
        return len(self.label)


class MSD_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sentiment_fc1 = nn.Linear(768, 768, bias=True)
        self.ReLu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.sentiment_fc2 = nn.Linear(768, 1, bias=True)
        self.sentiment_criterion = nn.MSELoss()

        self.correlation_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.multimodal_fusion = args.multimodal_fusion
        self.multilevel_fusion = args.multilevel_fusion
        if self.multilevel_fusion != 'concat' and self.multimodal_fusion != 'concat': self.final_fc = nn.Linear(768, 1, bias=True)
        elif self.multilevel_fusion == 'concat' and self.multimodal_fusion == 'concat': self.final_fc = nn.Linear(4*768, 1, bias=True)
        else: self.final_fc = nn.Linear(2*768, 1, bias=True)

        self.memory_length = args.memory_length
        self.sarcasm_bank = Queue(maxsize=self.memory_length)
        self.non_sarcasm_bank = Queue(maxsize=self.memory_length)

    def fusion(self, embeddings1, embeddings2, strategy):
        assert strategy in ['sum', 'product', 'concat']
        if strategy == 'sum': return (embeddings1+embeddings2) / 2
        elif strategy == 'product': return embeddings1 * embeddings2
        else: return torch.cat([embeddings1, embeddings2], dim=1)

    def forward(self, vision_embeddings, text_embeddings, text_sentiment, label=None):
        vision_embeddings, text_embeddings = vision_embeddings['embeddings'], text_embeddings['embeddings']

        batch_size = vision_embeddings.size()[0]

        text_embedd = text_embeddings.transpose(1, 2)
        vision_embedd = vision_embeddings
        attention_map = torch.bmm(vision_embedd, text_embedd)
        attention_map = self.correlation_conv(attention_map.unsqueeze(1)).squeeze()
        vision_c, text_c = attention_map.size(1), attention_map.size(2)

        vision_attention, text_attention = torch.sum(attention_map, dim=2)/text_c, torch.sum(attention_map, dim=1)/vision_c
        vision_attention, text_attention = torch.sigmoid(vision_attention), torch.sigmoid(text_attention)
        aligned_vision_embeddings = vision_attention.unsqueeze(-1) * vision_embedd
        aligned_text_embeddings = text_attention.unsqueeze(-1) * text_embedd.transpose(1,2)

        vision_embeddings = aligned_vision_embeddings
        text_embeddings = aligned_text_embeddings
        vision_nums, text_nums = vision_embeddings.size(1), text_embeddings.size(1)
        vision_CLS = torch.sum(vision_embeddings, dim=1) / vision_nums
        text_CLS = torch.sum(text_embeddings, dim=1) / text_nums


        # sentiment model
        text_sentiment_loss = 0
        for idx, cur_text_sentiment in enumerate(text_sentiment):
            cur_text_len = len(cur_text_sentiment)
            if self.args.text_backbone == 'bert':
                if cur_text_len > 510: cur_text_len, cur_text_sentiment = 510, cur_text_sentiment[:510]
            else:
                if cur_text_len > 512: cur_text_len, cur_text_sentiment = 512, cur_text_sentiment[:512]
            cur_text_embeddings = text_embeddings[idx, 0:cur_text_len, :]
            predicted_text_sentiment_embedding = self.sentiment_fc1(cur_text_embeddings)
            predicted_text_sentiment_embedding = self.dropout(self.ReLu(predicted_text_sentiment_embedding))
            predicted_text_sentiment = self.sentiment_fc2(predicted_text_sentiment_embedding)

            mask = torch.ones_like(cur_text_sentiment)
            mask[cur_text_sentiment == 0] = 0
            predicted_text_sentiment = predicted_text_sentiment * mask.unsqueeze(1)
            text_sentiment_loss += self.sentiment_criterion(predicted_text_sentiment.squeeze(1), cur_text_sentiment)

        text_sentiment_loss /= len(text_sentiment)


        text_cls_sentiment_embedding = self.sentiment_fc1(text_CLS)
        vision_cls_sentiment_embedding = self.sentiment_fc1(vision_CLS)

        with torch.no_grad():
            vision_cls_sentiment_embedd = self.dropout(self.ReLu(vision_cls_sentiment_embedding))
            vision_cls_sentiment = self.sentiment_fc2(self.ReLu(vision_cls_sentiment_embedd))
            text_cls_sentiment_embedd = self.dropout(text_cls_sentiment_embedding)
            text_cls_sentiment = self.sentiment_fc2(text_cls_sentiment_embedd)

            contrast_label = torch.abs(vision_cls_sentiment - text_cls_sentiment.t())
            contrast_label = torch.exp(-contrast_label)
            contrast_label = contrast_label / contrast_label.sum(1, keepdim=True)

        sim = torch.exp(torch.mm(F.normalize(vision_cls_sentiment_embedding, dim=1), F.normalize(text_cls_sentiment_embedding, dim=1).t()) / 0.2)
        sim = sim / sim.sum(1, keepdim=True)
        sentiment_contrast_loss = F.kl_div(torch.log(sim), contrast_label, reduction='batchmean')

        lamda_sentiment = torch.abs(text_cls_sentiment.squeeze(1) - vision_cls_sentiment.squeeze(1))

        # semantic model
        variance_vision = torch.nn.functional.normalize(torch.var(vision_CLS, dim=0), dim=-1)
        variance_text = torch.nn.functional.normalize(torch.var(text_CLS, dim=0), dim=-1)
        semantic_vision_embeddings = vision_CLS + vision_CLS*variance_vision.unsqueeze(0).repeat(batch_size,1)
        semantic_text_embeddings = text_CLS + text_CLS*variance_text.unsqueeze(0).repeat(batch_size,1)

        COS = nn.CosineSimilarity(dim=-1, eps=1e-6)

        sims = COS(semantic_vision_embeddings, semantic_text_embeddings)

        if label is not None:
            with torch.no_grad():
                for id in range (batch_size):
                    if label[id]==0:
                        if self.non_sarcasm_bank.full() == True: self.non_sarcasm_bank.get()
                        self.non_sarcasm_bank.put(sims[id])
                    elif label[id]==1:
                        if self.sarcasm_bank.full() == True: self.sarcasm_bank.get()
                        self.sarcasm_bank.put(sims[id])

        if self.non_sarcasm_bank.full() == True and self.sarcasm_bank.full() == True:
            with torch.no_grad():
                sarcasm_list = list(self.sarcasm_bank.queue)
                mu_sarcasm = sum(sarcasm_list) / self.args.memory_length
                sigma_sarcasm = torch.sqrt(sum([(tmp-mu_sarcasm)**2 for tmp in sarcasm_list]))

                non_sarcasm_list = list(self.non_sarcasm_bank.queue)
                mu_non_sarcasm = sum(non_sarcasm_list) / self.args.memory_length
                sigma_non_sarcasm = torch.sqrt(sum([(tmp-mu_non_sarcasm)**2 for tmp in non_sarcasm_list]))

            prob_sarcasm = (1/(sigma_sarcasm*np.sqrt(2*math.pi))) * torch.exp(-50*((sims - mu_sarcasm)/sigma_sarcasm)**2)
            prob_non_sarcasm = (1/(sigma_non_sarcasm*np.sqrt(2*math.pi))) * torch.exp(-50*((sims - mu_non_sarcasm)/sigma_non_sarcasm)**2)
            lamda_semantic = prob_sarcasm - prob_non_sarcasm
        else:
            lamda_semantic = torch.zeros_like(lamda_sentiment)
            prob_sarcasm=torch.zeros_like(lamda_sentiment)
            prob_non_sarcasm=torch.zeros_like(lamda_sentiment)


        # fusion
        semantic_cls = self.fusion(semantic_vision_embeddings, semantic_text_embeddings, self.multimodal_fusion)
        sentiment_cls = self.fusion(vision_cls_sentiment_embedding, text_cls_sentiment_embedding, self.multimodal_fusion)
        final_cls = self.fusion(semantic_cls, sentiment_cls, self.multilevel_fusion)
        final_cls = self.final_fc(final_cls).squeeze()
        fuse_final_cls = final_cls + self.args.lambda_sentiment*lamda_sentiment + self.args.lambda_semantic*lamda_semantic - self.args.constant

        return fuse_final_cls, sentiment_contrast_loss, text_sentiment_loss
        

def get_multimodal_model(args):
    return MSD_Net(args)


def get_multimodal_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.multimodal_lr, weight_decay=args.multimodal_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=num_training_steps)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, scheduler, criterion
