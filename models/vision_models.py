import torch
# import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils import data
from torch import optim, nn
import torch.nn.functional as F
from transformers import ViTModel, get_linear_schedule_with_warmup


def initialize_transforms(args):
    input_size = 224
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = {}
    image_transforms['train'] = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    image_transforms['test'] = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    return image_transforms


def generate_vision_loader(args, image_transforms):
    train_set = ImageFolder(args.train_data_dir, image_transforms['train'])
    test_set = ImageFolder(args.test_data_dir, image_transforms['test'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader


class VIT_MODEL(nn.Module):
    def __init__(self, vit, output_dim, alg):
        super().__init__()
        self.vit = vit
        if alg == 'base': self.vit.classifier = nn.Linear(768, output_dim)
        else: self.vit.classifier = nn.Linear(1024, output_dim)
        
    def forward(self, image):
        output = self.vit(image, return_dict=True)
        res = {'embeddings': output['last_hidden_state'][:, 1:, :], 'cls': output['pooler_output']}
        return res


class ResNet_MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.fc = nn.Linear(2048, 768, bias=True)
        
    def forward(self, image):
        output = self.model(image)
        batch_size, channles, w, h = output.size()
        output = output.view(batch_size, channles, -1)
        output = output.transpose(1, 2)
        output = self.fc(output)

        cls = torch.sum(output, dim=1) / (w*h)
        res = {'embeddings': output, 'cls': cls}
        return res


def get_vision_model(args):
    if args.vision_backbone == 'vit':
        if args.vision_model == 'base': vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        elif args.vision_model == 'large' : vit = ViTModel.from_pretrained('google/vit-large-patch16-224')
        else:
            print('Only support base and large models')
            exit(0)
        base_model = VIT_MODEL(vit, args.output_dim, args.vision_model)
    else:
        base_model = ResNet_MODEL()
    return base_model


def get_vision_configuration(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.vision_lr, weight_decay=args.vision_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, scheduler, criterion
