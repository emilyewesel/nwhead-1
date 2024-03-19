import os
import random
import numpy as np
import torch
from torchvision import transforms, datasets
import torchvision
from tqdm import tqdm
import argparse
from pprint import pprint
import json
import wandb
import matplotlib.pyplot as plt

# from data.bird import Cub200Dataset
# from data.dog import StanfordDogDataset
from util.metric import Metric, ECELoss
from util.utils import parse_bool, ParseKwargs, summary, save_checkpoint, initialize_wandb
from util import metric
from model import load_model
from nwhead.nw import NWNet
from fchead.fc import FCNet
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import math 
from collections import Counter






class ChexpertDataset(Dataset):
    def __init__(self, csv_file, train_base_path, test_base_path, transform=None, train=True, inject_underdiagnosis_bias=False, mode = "Cardiomegaly"):
        self.df = pd.read_csv(csv_file)
        if mode == "Cardiomegaly":
            self.df = self.df[self.df["Cardiomegaly"].isin([0, 1])]
        elif mode == "Pneumothorax":
            self.df = self.df[self.df["Pneumothorax"].isin([0, 1])]
        if mode == "No Finding":
            self.df["No Finding"].fillna(0, inplace=True)
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        self.df.dropna(subset=["Sex"], inplace=True)
        self.df = self.df[self.df.iloc[:, 1].isin(["Female", "Male"])]
        if train and inject_underdiagnosis_bias:
            female_indices = self.df[(self.df["Sex"] == "Female") & (self.df[mode] == 1)].index
            num_female_samples = len(female_indices)
            num_samples_to_convert = int(0.25 * num_female_samples)
            indices_to_convert = np.random.choice(female_indices, num_samples_to_convert, replace=False)
            self.df.loc[indices_to_convert, mode] = 0
        self.base_path = train_base_path if train else test_base_path
        self.transform = transform
        self.targets = torch.tensor(self.df[mode].values, dtype=torch.long)  
        self.genders = self.df.iloc[:, 1].dropna().map({'Female': 1, 'Male': 0}).astype(int).tolist()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0].split('/', 1)[-1]
        img_name = os.path.join(self.base_path, img_name)
        image = Image.open(img_name).convert('RGB')  

        label = self.targets[idx]
        gender = self.genders[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, gender
    def compute_class_weights(self):
        class_counts = Counter(self.targets.numpy())
        total_samples = sum(class_counts.values())
        class_weights = [total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))]
        sum_weights = sum(class_weights)
        class_weights = [weight / sum_weights for weight in class_weights]
        return torch.tensor(class_weights)
    def compute_class_weights2(self):
        class_counts_male = Counter()
        class_counts_female = Counter()

        for label, gender in zip(self.targets.numpy(), self.genders):
            if gender == 0:  # Male
                class_counts_male[label] += 1
            else:  # Female
                class_counts_female[label] += 1

        return {
            'male': {
                'positive': class_counts_male[1],
                'negative': class_counts_male[0]
            },
            'female': {
                'positive': class_counts_female[1],
                'negative': class_counts_female[0]
            }
        }
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, mode='min', delta=0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf

    def __call__(self, val_loss):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.mode == 'min':
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        else:
            print(f'Validation loss increased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')




class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='NW Head Training')
        # I/O parameters
        self.add_argument('--models_dir', default='./',
                  type=str, help='directory to save models')
        self.add_argument('--data_dir', default='./',
                  type=str, help='directory where data lives')
        self.add_argument('--log_interval', type=int,
                  default=25, help='Frequency of logs')
        self.add_argument('--workers', type=int, default=2,
                  help='Num workers')
        self.add_argument('--gpu_id', type=int, default=0,
                  help='gpu id to train on')
        self.add_bool_arg('debug_mode', False)

        # Machine learning parameters
        self.add_argument('--dataset', type=str, required=True)
        self.add_argument('--lr', type=float, default=5e-3,
                  help='Learning rate')
        self.add_argument('--batch_size', type=int,
                  default=64, help='Batch size')
        self.add_argument('--num_steps_per_epoch', type=int,
                  default=100000, help='Num steps per epoch')
        self.add_argument('--num_val_steps_per_epoch', type=int,
                  default=100000, help='Num validation steps per epoch')
        self.add_argument('--num_epochs', type=int, default=200,
                  help='Total training epochs')
        self.add_argument('--scheduler_milestones', nargs='+', type=int,
                  default=(100, 150), help='Step size for scheduler')
        self.add_argument('--scheduler_gamma', type=float,
                  default=0.1, help='Multiplicative factor for scheduler')
        self.add_argument('--seed', type=int,
                  default=0, help='Seed')
        self.add_argument('--weight_decay', type=float,
                  default=1e-4, help='Weight decay')
        self.add_argument('--arch', type=str, default='resnet18')
        self.add_argument(
          '--train_method', default='nwhead')
        self.add_bool_arg('freeze_featurizer', False)
        self.add_argument('--mode', type=str,
                  default="No Finding")

        # NW head parameters
        self.add_argument('--kernel_type', type=str, default='euclidean',
                  help='Kernel type')
        self.add_argument('--proj_dim', type=int,
                  default=0)
        self.add_argument('--n_shot', type=int,
                  default=2, help='Number of examples per class in support')
        self.add_argument('--n_way', type=int,
                  default=None, help='Number of training classes per query in support')

        # Weights & Biases
        self.add_bool_arg('use_wandb', True)
        self.add_argument('--wandb_api_key_path', type=str,
                            help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
        self.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                            help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    def add_bool_arg(self, name, default=True):
        """Add boolean argument to argparse parser"""
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name: default})

    def parse(self):
        args = self.parse_args()
        args.run_dir = os.path.join(args.models_dir,
                      'method{method}_dataset{dataset}_arch{arch}_lr{lr}_bs{batch_size}_projdim{proj_dim}_nshot{nshot}_nway{nway}_wd{wd}_seed{seed}'.format(
                        method=args.train_method,
                        dataset=args.dataset,
                        arch=args.arch,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        proj_dim=args.proj_dim,
                        nshot=args.n_shot,
                        nway=args.n_way,
                        wd=args.weight_decay,
                        seed=args.seed
                      ))
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        if not os.path.exists(args.run_dir):
            os.makedirs(args.run_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        # Print args and save to file
        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
        return args

    
def main():
    
    # Parse arguments
    args = Parser().parse()

    # Set random seed
    seed = args.seed
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
        print('No GPU detected... Training will be slow!')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Get transforms
    if args.dataset in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
                  transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        transform_test = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    else:
        transform_train = transforms.Compose([
                  transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip(), 
                  transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
              ])
        transform_test = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
              ])
    print('Transforms:\n', transform_train, transform_test)

    # Get dataloaders
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, True, transform_train, download=True)
        val_dataset = datasets.CIFAR10(args.data_dir, False, transform_test, download=True)
        train_dataset.num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir, True, transform_train, download=True)
        val_dataset = datasets.CIFAR100(args.data_dir, False, transform_test, download=True)
        train_dataset.num_classes = 100
    elif args.dataset == "chexpert":
        train_csv = '/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/train.csv'
        test_csv = '/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/valid.csv'
        baase = "/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/"
        baase2 = "/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/"
        train_dataset = ChexpertDataset(csv_file=train_csv, train_base_path=baase, test_base_path=baase2, transform=transform_train, train=True)
        val_dataset = ChexpertDataset(csv_file=test_csv, train_base_path=baase, test_base_path=baase2, transform=transform_test, train=False)
        print("initialized datasets")
        train_dataset.num_classes = 2
        genders = train_dataset.genders
        # train_dataset.targets = train_dataset._labels  # Add this line
        class_weights = train_dataset.compute_class_weights()
        print("Class Weights:", class_weights)
        
        class_counts = train_dataset.compute_class_weights2()

        print("Male class counts:")
        print("Positive:", class_counts['male']['positive'])
        print("Negative:", class_counts['male']['negative'])

        print("\nFemale class counts:")
        print("Positive:", class_counts['female']['positive'])
        print("Negative:", class_counts['female']['negative'])

    elif args.dataset == 'flower':
        train_dataset = datasets.Flowers102(args.data_dir, 'train', transform_train, download=True)
        val_dataset = datasets.Flowers102(args.data_dir, 'test', transform_test, download=True)
        train_dataset.num_classes = 102
        train_dataset.targets = train_dataset._labels
    elif args.dataset == 'aircraft':
        train_dataset = datasets.FGVCAircraft(args.data_dir, 'trainval', transform=transform_train, download=True)
        val_dataset = datasets.FGVCAircraft(args.data_dir, 'test', transform=transform_test, download=True)
        train_dataset.num_classes = 100
    else:
      raise NotImplementedError()

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    # Get network
    if args.arch == 'resnet18':
        feat_dim = 512
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_ResNet18')
        else:
            featurizer = load_model('resnet18')
    elif args.arch == 'densenet121':
        feat_dim = 1024
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_DenseNet121')
        else:
            featurizer = load_model('densenet121')
    elif args.arch == 'dinov2_vits14':
        feat_dim = 384
        featurizer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else:
        raise NotImplementedError
    
    if args.freeze_featurizer:
        for param in featurizer.parameters():
            param.requires_grad = False
    # args.train_method = 'fchead'
    if args.train_method == 'fchead':
        network = FCNet(featurizer, 
                        feat_dim, 
                        num_classes)
    elif args.train_method == 'nwhead':
        network = NWNet(featurizer, 
                        num_classes,
                        support_dataset=train_dataset,
                        feat_dim=feat_dim,
                        proj_dim=args.proj_dim,
                        kernel_type=args.kernel_type,
                        n_shot=args.n_shot,
                        n_way=args.n_way,
                        env_array = genders,
                        debug_mode=args.debug_mode)
    else:
        raise NotImplementedError()
    summary(network)
    network.to(args.device)

    # Set loss, optimizer, and scheduler
    # weight = torch.tensor()
    criterion = torch.nn.NLLLoss(weight = class_weights.to(args.device))
    optimizer = torch.optim.SGD(network.parameters(), 
                                lr=args.lr, 
                                momentum=0.9, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                          milestones=args.scheduler_milestones,
                          gamma=args.scheduler_gamma)

    
    # Tracking metrics
    list_of_metrics = [
        'loss:train',
        # 'balanced_acc:train',
        # 'macro_acc:train',
        'acc:train',
    ]
    if args.train_method == 'nwhead':
        list_of_val_metrics = [
            'loss:val:random',
            'loss:val:full',
            'loss:val:cluster',
            'acc:val:random',
            'acc:val:full',
            'acc:val:cluster',
            'acc:val:random:male',
            'acc:val:full:male',
            'acc:val:cluster:male',
            'acc:val:random:female',
            'acc:val:full:female',
            'acc:val:cluster:female',
            'balanced_acc:val:random',   # New metric for balanced accuracy
            'balanced_acc:val:full',     # New metric for balanced accuracy
            'balanced_acc:val:cluster',  # New metric for balanced accuracy
            # 'macro_acc:val:random',      # New metric for macro accuracy
            # 'macro_acc:val:full',        # New metric for macro accuracy
            # 'macro_acc:val:cluster',     # New metric for macro accuracy
            'ece:val:random',
            'ece:val:full',
            'ece:val:cluster',
            'loss:val:ensemble',
            'loss:val:knn',
            'loss:val:hnsw',
            'acc:val:ensemble',
            'acc:val:knn',
            'acc:val:hnsw',
            'balanced_acc:val:random:male',
            'balanced_acc:val:full:male',
            'balanced_acc:val:cluster:male',
            # 'macro_acc:val:random:male',   # New metric for male macro accuracy
            # 'macro_acc:val:full:male',     # New metric for male macro accuracy
            # 'macro_acc:val:cluster:male',  # New metric for male macro accuracy
            'ece:val:random:male',
            'ece:val:full:male',
            'ece:val:cluster:male',
            'balanced_acc:val:random:female',
            'balanced_acc:val:full:female',
            'balanced_acc:val:cluster:female',
            # 'macro_acc:val:random:female',   # New metric for female macro accuracy
            # 'macro_acc:val:full:female',     # New metric for female macro accuracy
            # 'macro_acc:val:cluster:female',  # New metric for female macro accuracy
            'ece:val:random:female',
            'ece:val:full:female',
            'ece:val:cluster:female',
            'acc:val:ensemble:male',
            'acc:val:knn:male',
            'acc:val:hnsw:male',
            'balanced_acc:val:ensemble:male',
            'balanced_acc:val:knn:male',
            'balanced_acc:val:hnsw:male',
            # 'macro_acc:val:ensemble:male',
            # 'macro_acc:val:knn:male',
            # 'macro_acc:val:hnsw:male',
            'acc:val:ensemble:female',
            'acc:val:knn:female',
            'acc:val:hnsw:female',
            'balanced_acc:val:ensemble:female',
            'balanced_acc:val:knn:female',
            'balanced_acc:val:hnsw:female',
            # 'macro_acc:val:ensemble:female',
            # 'macro_acc:val:knn:female',
            # 'macro_acc:val:hnsw:female',
            'f1:val:random',
            'f1:val:full',
            'f1:val:cluster',
            'tpr:val:random',
            'tpr:val:full',
            'tpr:val:cluster',
            'auc:val:random',
            'auc:val:full',
            'auc:val:cluster',
            'acc:val:random:male', 'f1:val:random:male', 'tpr:val:random:male', 'auc:val:random:male',
            'f1:val:full:male', 'tpr:val:full:male', 'auc:val:full:male',
            'f1:val:cluster:male', 'tpr:val:cluster:male', 'auc:val:cluster:male',
            'f1:val:random:female', 'tpr:val:random:female', 'auc:val:random:female',
            'f1:val:full:female', 'tpr:val:full:female', 'auc:val:full:female',
            'f1:val:cluster:female', 'tpr:val:cluster:female', 'auc:val:cluster:female',
            'f1:val:knn:male',
            'f1:val:hnsw:male',
            'f1:val:ensemble:male',
            'f1:val:ensemble',
            'f1:val:knn',
            'f1:val:hnsw',
            'f1:val:ensemble:female',
            'f1:val:knn:female',
            'f1:val:hnsw:female',
            'tpr:val:knn:male',
            'tpr:val:hnsw:male',
            'tpr:val:ensemble:male',
            'tpr:val:ensemble',
            'tpr:val:knn',
            'tpr:val:hnsw',
            'tpr:val:ensemble:female',
            'tpr:val:knn:female',
            'tpr:val:hnsw:female',
            'auc:val:knn:male',
            'auc:val:hnsw:male',
            'auc:val:ensemble:male',
            'auc:val:ensemble',
            'auc:val:knn',
            'auc:val:hnsw',
            'auc:val:ensemble:female',
            'auc:val:knn:female',
            'auc:val:hnsw:female',

        ]


    else:
        list_of_val_metrics = [
            'loss:val',
            'acc:val',
            'ece:val',
            # 'loss:val:female',
            'acc:val:female',
            'balanced_acc:val:female',
            'balanced_acc:val:male',
            'ece:val:female',
            # 'loss:val:male',
            'acc:val:male',
            'ece:val:male',
            'f1:val',
            'tpr:val',
            'auc:val',
            'f1:val:male',
            'tpr:val:male',
            'auc:val:male',
            'f1:val:female',
            'tpr:val:female',
            'auc:val:female'
        ] 
    args.metrics = {}
    args.metrics.update({key: Metric() for key in list_of_metrics})
    args.val_metrics = {}
    args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

    if args.use_wandb:
        initialize_wandb(args)
        print("initilaized wandb")

    # Training loop
    start_epoch = 1
    best_acc1 = 0
    early_stopping = EarlyStopping(patience=5, mode='min')
    for epoch in range(start_epoch, args.num_epochs+1):
        print('Epoch:', epoch)
        if args.train_method == 'nwhead':
            # print("WRONG!! WRONG!! epoch")
            network.eval()
            network.precompute()
            print('Evaluating on random mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='random')
            print('Evaluating on full mode...')
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args, mode='full')
            print('Evaluating on cluster mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='cluster')
            print('Evaluating on ensemble mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='ensemble')
            print('Evaluating on knn mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='knn')
            print('Evaluating on hnsw mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='hnsw')

        else:
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args)

        print('Training...')
        train_epoch(train_loader, network, criterion, optimizer, args)
        scheduler.step()

        # Remember best acc and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if epoch % args.log_interval == 0:
            save_checkpoint(epoch, network, optimizer,
                      args.ckpt_dir, scheduler, is_best=is_best)
        print("Train loss={:.6f}, train acc={:.6f}, lr={:.6f}".format(
            args.metrics['loss:train'].result(), args.metrics['acc:train'].result(), scheduler.get_last_lr()[0]))
        if args.train_method == 'fchead':
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val'].result(), args.val_metrics['acc:val'].result()))
            print()
        else:
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:random'].result(), args.val_metrics['acc:val:random'].result()))
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:full'].result(), args.val_metrics['acc:val:full'].result()))
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:cluster'].result(), args.val_metrics['acc:val:cluster'].result()))
            print()
            early_stopping(args.val_metrics['loss:val:full'].result())

        # If early stopping criterion met, break the loop
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        if args.use_wandb:
            wandb.log({k: v.result() for k, v in args.metrics.items()})
            wandb.log({k: v.result() for k, v in args.val_metrics.items()})

        # Reset metrics
        for _, metric in args.metrics.items():
            metric.reset_state()
        for _, metric in args.val_metrics.items():
            metric.reset_state()
def balanced_acc_fcn(preds, gts, class_labels):
    balanced_acc_per_class = []
    for label in class_labels:
        class_indices = (gts == label).nonzero()
        class_preds = preds[class_indices]
        class_gts = gts[class_indices]
        class_acc = metric.acc(class_preds, class_gts)
        balanced_acc_per_class.append(class_acc)
    balanced_acc = torch.tensor(balanced_acc_per_class).mean()
    return balanced_acc.item()

def macro_acc_fcn(preds, gts, class_labels):
    return balanced_acc_fcn(preds, gts, class_labels) * 100
def tpr_score(y_true, y_pred):
    # Calculate True Positive Rate (TPR)
    print("y_true", y_true)
    print("y_pred", y_pred)
    y_pred = y_pred > 0.5
    tpr = sum((y_true == 1) & (y_pred == 1)) / sum(y_true == 1)
    return tpr if not math.isnan(tpr) else 0.0

def auc_score(y_true, y_pred_proba):
    # Calculate Area Under the Curve (AUC)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_value = auc(fpr, tpr)
    return auc_value if not math.isnan(auc_value) else 0.0


def train_epoch(train_loader, network, criterion, optimizer, args):
    """Train for one epoch."""
    network.train()

    for i, batch in tqdm(enumerate(train_loader), 
        total=min(len(train_loader), args.num_steps_per_epoch)):
        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=True)
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=True)
        args.metrics['loss:train'].update_state(step_res['loss'], step_res['batch_size'])
        args.metrics['acc:train'].update_state(step_res['acc'], step_res['batch_size'])
        # args.metrics['balanced_acc:train'].update_state(step_res['balanced_acc'], step_res['batch_size'])
        # args.metrics['macro_acc:train'].update_state(step_res['macro_acc'], step_res['batch_size'])
        if i == args.num_steps_per_epoch:
            break

def eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    probs = {'male': [], 'female': []}
    gts = {'male': [], 'female': []}

    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        img, label, gender = batch
        img = img.float().to(args.device)
        label = label.to(args.device)
        gender = gender.to(args.device)

        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=False)
            args.val_metrics['loss:val'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics['acc:val'].update_state(step_res['acc'], step_res['batch_size'])
            predictions = np.argmax(step_res['prob'].cpu().numpy(), axis=1)
            args.val_metrics['f1:val'].update_state(f1_score(step_res['gt'].cpu().numpy(), predictions, average='weighted'), step_res['batch_size'])
            args.val_metrics['tpr:val'].update_state(tpr_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])
            args.val_metrics['auc:val'].update_state(auc_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])

            for j in range(len(gender)):
                gender_str = 'male' if gender[j] == 0 else 'female'
                probs[gender_str].append(step_res['prob'][j].unsqueeze(0))
                gts[gender_str].append(step_res['gt'][j].unsqueeze(0))
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=False, mode=mode)
            args.val_metrics[f'loss:val:{mode}'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics[f'acc:val:{mode}'].update_state(step_res['acc'], step_res['batch_size'])
            print("f1 gt", step_res['gt'].cpu().numpy())
            predictions = np.argmax(step_res['prob'].cpu().numpy(), axis=1)

            args.val_metrics[f'f1:val:{mode}'].update_state(f1_score(step_res['gt'].cpu().numpy(), predictions, average='weighted'), step_res['batch_size'])
            args.val_metrics[f'tpr:val:{mode}'].update_state(tpr_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])
            args.val_metrics[f'auc:val:{mode}'].update_state(auc_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])

            # Separate metrics for males and females
            for j in range(len(gender)):
                gender_str = 'male' if gender[j] == 0 else 'female'
                probs[gender_str].append(step_res['prob'][j].unsqueeze(0))
                gts[gender_str].append(step_res['gt'][j].unsqueeze(0))

        if i == args.num_val_steps_per_epoch:
            break

    # Log metrics for males
    male_probs = torch.cat(probs['male'], dim=0)
    male_gts = torch.cat(gts['male'], dim=0)
    male_acc = metric.acc(male_probs.argmax(-1), male_gts)
    male_ece = (ECELoss()(male_probs, male_gts) * 100).item()
    

    # Log metrics for females
    female_probs = torch.cat(probs['female'], dim=0)
    female_gts = torch.cat(gts['female'], dim=0)
    female_acc = metric.acc(female_probs.argmax(-1), female_gts)

    male_probs_np = male_probs.cpu().numpy()
    female_probs_np = female_probs.cpu().numpy()
    male_gts_np = male_gts.cpu().numpy()
    female_gts_np = female_gts.cpu().numpy()

    male_balanced_acc = balanced_acc_fcn(male_probs.argmax(-1), male_gts, class_labels=[0, 1])
    female_balanced_acc = balanced_acc_fcn(female_probs.argmax(-1), female_gts, class_labels=[0, 1])
    male_macro_acc = macro_acc_fcn(male_probs.argmax(-1), male_gts, class_labels=[0, 1])
    female_macro_acc = macro_acc_fcn(female_probs.argmax(-1), female_gts, class_labels=[0, 1])

    female_ece = (ECELoss()(female_probs, female_gts) * 100).item()
    
    if args.train_method == 'fchead':
        args.val_metrics[f'acc:val:male'].update_state(male_acc * 100, 1)
        # args.val_metrics[f'ece:val:male'].update_state(male_ece, 1)
        args.val_metrics[f'acc:val:female'].update_state(female_acc * 100, 1)
        args.val_metrics[f'balanced_acc:val:male'].update_state(male_balanced_acc*100, 1)
        args.val_metrics[f'balanced_acc:val:female'].update_state(female_balanced_acc * 100, 1)
        # args.val_metrics[f'ece:val:female'].update_state(female_ece, 1)
        male_predictions = np.argmax(male_probs_np, axis=1)
        female_predictions = np.argmax(female_probs_np, axis=1)
        args.val_metrics[f'f1:val:male'].update_state(f1_score(male_gts_np, male_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:male'].update_state(tpr_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:male'].update_state(auc_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'f1:val:female'].update_state(f1_score(female_gts_np, female_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:female'].update_state(tpr_score(female_gts_np, female_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:female'].update_state(auc_score(female_gts_np, female_predictions), step_res['batch_size'])
        return args.val_metrics['acc:val'].result()
    else:
        male_predictions = np.argmax(male_probs_np, axis=1)
        female_predictions = np.argmax(female_probs_np, axis=1)
        args.val_metrics[f'acc:val:{mode}:male'].update_state(male_acc * 100, 1)
        # args.val_metrics[f'ece:val:{mode}:male'].update_state(male_ece, 1)
        args.val_metrics[f'acc:val:{mode}:female'].update_state(female_acc * 100, 1)
        # args.val_metrics[f'ece:val:{mode}:female'].update_state(female_ece, 1)
        args.val_metrics[f'balanced_acc:val:{mode}:male'].update_state(male_balanced_acc*100, 1)
        args.val_metrics[f'balanced_acc:val:{mode}:female'].update_state(female_balanced_acc * 100, 1)
        # args.val_metrics[f'macro_acc:val:{mode}:male'].update_state(male_macro_acc*100, 1)
        # args.val_metrics[f'macro_acc:val:{mode}:female'].update_state(female_macro_acc * 100, 1)
        args.val_metrics[f'f1:val:{mode}:male'].update_state(f1_score(male_gts_np, male_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:{mode}:male'].update_state(tpr_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:{mode}:male'].update_state(auc_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'f1:val:{mode}:female'].update_state(f1_score(female_gts_np, female_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:{mode}:female'].update_state(tpr_score(female_gts_np, female_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:{mode}:female'].update_state(auc_score(female_gts_np, female_predictions), step_res['batch_size'])



        return args.val_metrics[f'acc:val:{mode}'].result()

def fc_step(batch, network, criterion, optimizer, args, is_train=True):
    '''Train/val for one step.'''
    img, label, gender = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        output = network(img)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)
        # balanced_acc = balanced_acc_fcn(output.argmax(-1), label, class_labels=[0, 1])
        # macro_acc = macro_acc_fcn(output.argmax(-1), label, class_labels=[0, 1])

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc * 100, \
            # 'balanced_acc': balanced_acc * 100, \
            # 'macro_acc': macro_acc * 100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}


def nw_step(batch, network, criterion, optimizer, args, is_train=True, mode='random'):
    '''Train/val for one step.'''
    img, label, gender = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    gender = gender.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        if is_train:
            output = network(img, gender)
        else:
            output = network.predict(img, mode)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)
        # balanced_acc = balanced_acc_fcn(output.argmax(-1), label, class_labels=[0, 1])
        # macro_acc = macro_acc_fcn(output.argmax(-1), label, class_labels=[0, 1])

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc * 100, \
            # 'balanced_acc': balanced_acc * 100, \
            # 'macro_acc': macro_acc * 100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}

if __name__ == '__main__':
    main()