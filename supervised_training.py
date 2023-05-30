#%% TO-Do
# implement WeightedRandomSampler - done
# experiment with colorjitter
# calculate mean and SD, include normalization in transforms
# (remember to normalize both train and inference transform)

# data_snu_snub_train_1900normals
# MEAN: 0.5263558626174927
# SD  : 0.25668418407440186

#%%
import os
import sys
import math
import argparse
from PIL import Image
from pathlib import Path

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.autograd.set_detect_anomaly(True)

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import utils
import vision_transformer as vit_o

#%%
def get_arguments():
    parser = argparse.ArgumentParser('multilabel_classification',
                                     add_help=False)
    parser.add_argument('--data_path', default=r'/mnt/d/data_snu_snub_train_sample',
                        help='Path to training data folder')
    parser.add_argument('--checkpoint', default = None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--n_classes', default=4, type=int,
                        help='Number of classes for multilabel classification.')
    parser.add_argument('--save_dir', default=r'.',
                        help='Path to save trained models')
    parser.add_argument('--save_name', default='best.pt')
    
    # Basic hyperparameters
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--use_fp16', default=True, type=bool)
    parser.add_argument('--log_interval', default=1, type=int)
    
    # hyperparameters for schedulers
    parser.add_argument('--lr', default=0.00002, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    
    # hyperparameters for gradient descent
    parser.add_argument('--clip_grad', default=3.0, type=float,
                        help="""Maximal parameter gradient norm if using gradient clipping.
                        Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures.
                        0 for disabling.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output layer fixed.
                        Typically doing so during the first epoch helps training.
                        Try increasing this value if the loss does not decrease.""")
    
    # args = parser.parse_args("")
    return parser


#%%
# data_snu_snub_train_1900normals
# MEAN: 0.5263558626174927
# SD  : 0.25668418407440186
def make_dataloader(args, mean=0.5263558626174927, SD=0.25668418407440186):
    if not mean or not SD:
        mean, SD = utils.calculate_mean_SD(args.data_path)
    train_transform = T.Compose([
        T.RandomResizedCrop(256, scale=(0.85, 1), interpolation=Image.BICUBIC),
        T.RandomRotation(degrees=(-15,15)),
        T.GaussianBlur(kernel_size=251, sigma=(0.3, 0.5)),
        T.ToTensor(),
        T.Normalize(mean, SD),
    ])
    dataset = ImageFolder(root=args.data_path, transform=train_transform)
    classes = os.listdir(args.data_path)
    class_counts = {cl:0 for cl in classes}
    for cl in class_counts:
        class_counts[cl] = len(os.listdir(os.path.join(args.data_path, cl)))
    print("\n Number of files in the dataset for each class: \n", class_counts)
    class_counts = {dataset.class_to_idx[cl]:count for cl, count in class_counts.items()}
    print("class_to_idx: \n", class_counts)
    weights = [1/class_counts[cl] for cl in dataset.targets]
    
    datasampler = WeightedRandomSampler(weights=weights,
                                        num_samples=len(dataset),
                                        replacement=True)
    dataloader = DataLoader(
        dataset,
        sampler=datasampler,
        batch_size = args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

#%%
def load_model(args, patch_size=8, out_dim=65536):

    model = vit_o.__dict__['vit_small'](patch_size=patch_size)
    embed_dim = model.embed_dim
    model = utils.MultiCropWrapper(
        model,
        vit_o.DINOHead(in_dim=embed_dim, out_dim=out_dim),
        # vit_o.CLS_head(in_dim=384, hidden_dim=256, num_class=1)
        vit_o.MLP_Head(in_dim=384, hidden_dim=256, n_classes=args.n_classes)
    )
    
    
    param_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups)
    fp16_scaler = torch.cuda.amp.GradScaler()
    
    if args.checkpoint:
        utils.restart_from_checkpoint(
            ckp_path=args.checkpoint,
            model=model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler
        )
    model.eval()
    
    return model.to(args.device), optimizer, fp16_scaler


#%%
def train(args):
    wandb.init(project='multilabel-classification-supervized-training')
    dataloader = make_dataloader(args=args)
    model, optimizer, fp16_scaler = load_model(args)
    bce_loss = nn.BCEWithLogitsLoss()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
        
    # initialize schedulers
    lr_schedule = utils.cosine_scheduler(
        base_value= args.lr * (args.batch_size * utils.get_world_size()) / 16.,
        final_value= args.min_lr,
        epochs= args.epochs,
        niter_per_ep= len(dataloader),
        warmup_epochs= args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(dataloader)
    )
    
    best_loss = 9999
    cum_avg_loss = 0
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        
        for it, (images, labels) in enumerate(dataloader):
            it = len(dataloader) * epoch + it # global training iteration count
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[it]
                if i == 0: # only the first group is regularized
                    param_group['weight_decay'] = wd_schedule[it]
                    
            images = images.to(args.device)
            labels = labels.float().to(args.device)
            labels_one_hot = F.one_hot(labels.long(), num_classes=args.n_classes)
            
            loss_dict = {}
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                preds = model(images)
                for i in range(args.n_classes):
                    loss_dict[f"loss{i}"] = bce_loss(torch.flatten(preds[i].to(args.device)),
                                                     labels_one_hot[:, i].float().to(args.device))

                
            optimizer.zero_grad()
            
            loss_all = 0
            for i, (_, loss) in enumerate(loss_dict.items()):
                if not math.isfinite(loss.item()):
                    print(f"Loss is invalid: \n {loss_dict}")
                    print("Stopping training.")
                    sys.exit(1)
                    
                # if i == args.n_classes-1:
                #     fp16_scaler.scale(loss).backward()
                # else:
                #     fp16_scaler.scale(loss).backward(retain_graph=True)
                
                loss_all += loss
            
            fp16_scaler.scale(loss_all).backward()
        
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                                args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            
            avg_loss = sum(loss_dict.values()).item() / len(loss_dict)
            cum_avg_loss += avg_loss
            if (it+1) % args.log_interval == 0:
                wandb.log({k:v.item() for k,v in loss_dict.items()},
                          step=it)
                cum_avg_loss /= args.log_interval
                print(cum_avg_loss)
                if cum_avg_loss <= best_loss:
                    best_loss = cum_avg_loss
                    save_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'fp16_scaler': fp16_scaler.state_dict()
                    }
                    utils.save_on_master(save_dict, os.path.join(args.save_dir, args.save_name))
                    print(f'Saved {args.save_name} trained on {it+1} batches')
                    print(f'LOSS: {best_loss}')
                cum_avg_loss = 0

#%%
if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args("")
    wandb.login()
    train(args)
# %%
