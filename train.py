import os
import random
import numpy as np
import torch
from torchvision import transforms, datasets
import sys
# sys.path.insert(0,'/dataNAS/people/ewesel1/nwhead-1/data/')
# import data.bird
# import data.chexpert
# from data.chexpert import ChexpertDataset
from tqdm import tqdm
import argparse
from pprint import pprint
import json
import wandb
import matplotlib.pyplot as plt
from util.metric import Metric, ECELoss
from sklearn.metrics import f1_score
from util.utils import parse_bool, ParseKwargs, summary, save_checkpoint, initialize_wandb, EarlyStopping
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
from torch.optim import AdamW
from collections import Counter
from torch.utils.data import Dataset

def crop_path_results(path):
    return path.split('/train/')[-1]  # Crop path after valid
def crop_path_train(path):
    return path.split('/train/')[-1]  # Crop path after valid
# Crop paths in the second dataset


class ChexpertDataset(Dataset):
    def __init__(self, csv_file, train_base_path, test_base_path, transform=None, train=True, inject_underdiagnosis_bias=False, train_class = "Cardiomegaly", fc_results= None, correct_support_only=False):
        self.df = pd.read_csv(csv_file)
        
        if train_class == "Cardiomegaly" and correct_support_only and train:
            
            self.df_fc_results = pd.read_csv(fc_results)
            
            self.df_fc_results['merge_Path'] = self.df_fc_results['Path'].apply(crop_path_results)
            self.df['merge_Path'] = self.df['Path'].apply(crop_path_train)
            print("emily original", self.df["merge_Path"])
            print("emily fc", self.df_fc_results["merge_Path"])
            merged = pd.merge(self.df, self.df_fc_results, on='merge_Path', how='inner')
            print("emily merged", merged.head())
            filtered_df = merged[merged['Ground Truth'] == merged['Prediction']]
            print("emily filtered", filtered_df.head())
            self.df = filtered_df

        if train_class == "No Finding":
            self.df["No Finding"].fillna(0, inplace=True)
        else:
            self.df = self.df[self.df[train_class].isin([0, 1])]
        
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        self.df.dropna(subset=["Sex"], inplace=True)
        self.df = self.df[self.df.iloc[:, 1].isin(["Female", "Male"])]
        if train and inject_underdiagnosis_bias:
            female_indices = self.df[(self.df["Sex"] == "Female") & (self.df[train_class] == 1)].index
            num_female_samples = len(female_indices)
            num_samples_to_convert = int(0.25 * num_female_samples)
            indices_to_convert = np.random.choice(female_indices, num_samples_to_convert, replace=False)
            self.df.loc[indices_to_convert, train_class] = 0
        self.base_path = train_base_path if train else test_base_path
        self.transform = transform
        self.targets = torch.tensor(self.df[train_class].values, dtype=torch.long)  
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

        return image, label, gender, img_name
    
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


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='NW Head Training')
        # I/O parameters
        self.add_argument('--models_dir', default='./saved_models/',
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
        self.add_argument('--lr', type=float, default=5e-4,
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
                  default=(50, 75), help='Step size for scheduler')
        self.add_argument('--scheduler_gamma', type=float,
                  default=0.1, help='Multiplicative factor for scheduler')
        self.add_argument('--seed', type=int,
                  default=1964, help='Seed')
        self.add_argument('--weight_decay', type=float,
                  default=1e-4, help='Weight decay')
        self.add_argument('--optimizer', type=str,
                  default='sgd', help='Weight decay') # options: sgd, adam, adamw
        self.add_argument('--arch', type=str, default='resnet18')
        self.add_bool_arg('pretrained', True)
        self.add_argument(
          '--train_method', default='nwhead')
        self.add_bool_arg('freeze_featurizer', False)
        self.add_argument('--train_class', type=str, default="Cardiomegaly")

        # NW head parameters
        self.add_argument('--kernel_type', type=str, default='euclidean',
                  help='Kernel type')
        self.add_argument('--proj_dim', type=int,
                  default=0)
        self.add_argument('--n_shot', type=int,
                  default=2, help='Number of examples per class in support')
        self.add_argument('--n_way', type=int,
                  default=None, help='Number of training classes per query in support')
        self.add_argument('--correct_support_only', type = bool, default = False, help = "Should only correctly classified samples be included in the dataset? True if yes, False if no.")

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
                      'method{method}_dataset{dataset}_arch{arch}_pretrained{pretrained}_lr{lr}_bs{batch_size}_projdim{proj_dim}_nshot{nshot}_nway{nway}_wd{wd}_seed{seed}_class{train_class}'.format(
                        method=args.train_method,
                        dataset=args.dataset,
                        arch=args.arch,
                        pretrained=args.pretrained,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        proj_dim=args.proj_dim,
                        nshot=args.n_shot,
                        nway=args.n_way,
                        wd=args.weight_decay,
                        seed=args.seed,
                        train_class=args.train_class,
                        correct_support_only = args.correct_support_only
                      ))
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        args.output_csv_dir = os.path.join(args.run_dir, 'output_csv') 
        if not os.path.exists(args.run_dir):
            os.makedirs(args.run_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.output_csv_dir):
            os.makedirs(args.output_csv_dir)

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
    # If seed is 0, the experiments are random, otherwise the seed is set
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
    elif args.dataset in ['chexpert']:
        transform_train = transforms.Compose([
                  transforms.Resize((224, 224)),  # Resize to 224x224
                  transforms.RandomRotation(20),  # rotation within the range [-20, 20] degrees
                  transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random crop with scaling between 80% and 100%
                  transforms.ColorJitter(brightness=0.1, contrast=0.1),transforms.ToTensor(),  # Convert the image to a tensor
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        transform_test = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        fc_head_results = "/dataNAS/people/ewesel1/nwhead-1/saved_models/methodfchead_datasetchexpert_archresnet18_pretrainedFalse_lr0.0001_bs64_projdim0_nshot8_nwayNone_wd0.0001_seed1964_classCardiomegaly/output_csv/train_model_output_epoch_26.csv"
        train_dataset = ChexpertDataset(csv_file=train_csv, train_base_path=baase, test_base_path=baase2, transform=transform_train, train_class=args.train_class, train=True, fc_results=fc_head_results, correct_support_only=False)
        train_dataset_correct_only = ChexpertDataset(csv_file=train_csv, train_base_path=baase, test_base_path=baase2, transform=transform_train, train_class=args.train_class, train=True, fc_results=fc_head_results, correct_support_only=True)
        val_dataset = ChexpertDataset(csv_file=test_csv, train_base_path=baase, test_base_path=baase2, transform=transform_test, train_class=args.train_class, train=False)
        print("initialized datasets")
        train_dataset.num_classes = 2
        
        genders = train_dataset.genders
        args.correct_support_only = False
        if args.correct_support_only:
            train_dataset_correct_only.num_classes = 2
            genders = train_dataset_correct_only.genders
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
    if 'resnet' in args.arch:
        if '18' in args.arch:
            feat_dim = 512
        elif '50' in args.arch:
            feat_dim = 2048
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_ResNet18')
        elif args.dataset in ['chexpert']:
            featurizer = load_model(args.arch, pretrained=args.pretrained)
        else:
            featurizer = load_model(args.arch)
    elif 'densenet' in args.arch:
        feat_dim = 1024
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_DenseNet121')
        elif args.dataset in ['chexpert']:
            featurizer = load_model(args.arch, pretrained=args.pretrained)
        else:
            featurizer = load_model(args.arch)
    elif args.arch == 'dinov2_vits14':
        feat_dim = 384
        featurizer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else:
        raise NotImplementedError
    
    if args.freeze_featurizer:
        for param in featurizer.parameters():
            param.requires_grad = False
    if args.train_method == 'fchead':
        network = FCNet(featurizer, 
                        feat_dim, 
                        num_classes)
    elif args.train_method == 'nwhead':
        if args.correct_support_only:
            network = NWNet(featurizer, 
                            num_classes,
                            support_dataset=train_dataset_correct_only,
                            feat_dim=feat_dim,
                            proj_dim=args.proj_dim,
                            kernel_type=args.kernel_type,
                            n_shot=args.n_shot,
                            n_way=args.n_way,
                            env_array = genders,
                            debug_mode=args.debug_mode)
        else:
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
    criterion = torch.nn.NLLLoss(weight = class_weights.to(args.device))
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), 
                                lr=args.lr, 
                                momentum=0.9, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                             lr=args.lr, 
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(network.parameters(),
                  lr=args.lr,
                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                          milestones=args.scheduler_milestones,
                          gamma=args.scheduler_gamma)

    # Define tracked metrics during train and validation
    list_of_metrics, list_of_val_metrics = metric.define_train_eval_metrics(args.train_method)
    
    args.metrics = {}
    args.metrics.update({key: Metric() for key in list_of_metrics})
    args.val_metrics = {}
    args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

    if args.use_wandb:
        initialize_wandb(args)
        print("initilaized wandb")

    # Training loop
    start_epoch = 1
    best_bacc1 = 0
    lowest_val_loss = np.Inf
    early_stopping = EarlyStopping(patience=5, mode='min')
    
    # Run evaluation one time to make sure everything runs
    run_evaluation(val_loader, network, criterion, optimizer, args)
    
    for epoch in range(start_epoch, args.num_epochs+1):
        print('Epoch:', epoch)

        print('Training...')
        train_csv_output_dict = train_epoch(train_loader, network, criterion, optimizer, args)
        scheduler.step()
        
        bacc1, val_loss, csv_output_dict = run_evaluation(val_loader, network, criterion, optimizer, args)
        
        # DataFrame construction
        csv_output_df = pd.DataFrame(csv_output_dict)
        if args.train_method == "fchead":
            train_csv_output_df = pd.DataFrame(train_csv_output_dict)

        # Step 4: Write to CSV
        output_csv_path = args.output_csv_dir + f'/model_output_epoch_{epoch}.csv'
        csv_output_df.to_csv(output_csv_path, index=False)
        if args.train_method == "fchead":
            output_csv_path = args.output_csv_dir + f'/train_model_output_epoch_{epoch}.csv'
            train_csv_output_df.to_csv(output_csv_path, index=False)
        
        # Save checkpoint based on best acc
        # This criterion saves the models with the highest val accuracy
        bacc_is_best = bacc1 > best_bacc1
        best_bacc1 = max(bacc1, best_bacc1)
        if bacc_is_best:
            save_checkpoint(epoch, network, optimizer,
                      args.ckpt_dir, scheduler, is_best=False) # Save global "best" model based on loss and not on BACC
            
        # This criterion saves the models with the lowest validation loss
        is_best = val_loss < lowest_val_loss
        lowest_val_loss = min(val_loss, lowest_val_loss)
        
        if is_best:
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
        for _, metric_m in args.metrics.items():
            metric_m.reset_state()
        for _, metric_m in args.val_metrics.items():
            metric_m.reset_state()


def run_evaluation(val_loader, network, criterion, optimizer, args):
        if args.train_method == 'nwhead':
            network.eval()
            network.precompute()
            print('Evaluating on random mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='random')
            print('Evaluating on full mode...')
            bacc1, val_loss, csv_output_dict = eval_epoch(val_loader, network, criterion, optimizer, args, mode='full')
            print('Evaluating on cluster mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='cluster')
            print('Evaluating on ensemble mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='ensemble')
            print('Evaluating on knn mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='knn')
            print('Evaluating on hnsw mode...')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='hnsw')

        else:
            bacc1, val_loss, csv_output_dict = eval_epoch(val_loader, network, criterion, optimizer, args)
            
        return bacc1, val_loss, csv_output_dict


def train_epoch(train_loader, network, criterion, optimizer, args):
    """Train for one epoch."""
    network.train()
    train_csv_output_dict = []
    for i, batch in tqdm(enumerate(train_loader), 
        total=min(len(train_loader), args.num_steps_per_epoch)):
        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=True)
            #step_res has predictions 
            #step_res["gt"]
            img, label, gender, id = batch
            img = img.float().to(args.device)
            label = label.to(args.device)
            gender = gender.to(args.device)
            #predictions will be argmax of softmax
            
            predictions = np.argmax(step_res['prob'].detach().cpu().numpy(), axis=1)
            # predictions = np.argmax(step_res['prob'].cpu().numpy(), axis=1)
            for label, pred, prob, gend, img_id in zip(label, predictions, step_res['prob'].detach().cpu().numpy(), gender.detach().cpu().numpy(), id):
                
                train_csv_output_dict.append({
                    'Ground Truth': label.item(),
                    'Prediction': pred,
                    'Probability Class 0': prob[0],
                    'Probability Class 1': prob[1],
                    'Gender': gend,
                    'Path': img_id
                })
            

        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=True)
        args.metrics['loss:train'].update_state(step_res['loss'], step_res['batch_size'])
        args.metrics['acc:train'].update_state(step_res['acc'], step_res['batch_size'])
        args.metrics['balanced_acc:train'].update_state(step_res['balanced_acc'], step_res['batch_size'])
        if i == args.num_steps_per_epoch:
            break
    if args.train_method == 'fchead':
        return train_csv_output_dict


def eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    probs = {'male': [], 'female': []}
    gts = {'male': [], 'female': []}
    # I want to create a dataframe to store the predictions, gt, scores for posthoc analysis
    csv_output_dict = [] 

    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        img, label, gender, id = batch
        img = img.float().to(args.device)
        label = label.to(args.device)
        gender = gender.to(args.device)

        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=False)
            args.val_metrics['loss:val'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics['acc:val'].update_state(step_res['acc'], step_res['batch_size'])
            args.val_metrics['balanced_acc:val'].update_state(step_res['balanced_acc'], step_res['batch_size'])
            
            overall_ece = (ECELoss()(step_res['prob'], label) * 100).item()
            predictions = np.argmax(step_res['prob'].cpu().numpy(), axis=1)
            
            # Collect data; ensure they are detached and moved to CPU if necessary
            for label, pred, prob, gend, img_id in zip(label, predictions, step_res['prob'].cpu().numpy(), gender.cpu().numpy(), id):
                
                csv_output_dict.append({
                    'Ground Truth': label.item(),
                    'Prediction': pred,
                    'Probability Class 0': prob[0],
                    'Probability Class 1': prob[1],
                    'Gender': gend,
                    'Path': img_id
                })
            
            args.val_metrics['ece:val'].update_state(overall_ece, 1)
            args.val_metrics['f1:val'].update_state(f1_score(step_res['gt'].cpu().numpy(), predictions, average='weighted'), step_res['batch_size'])
            args.val_metrics['tpr:val'].update_state(metric.tpr_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])
            args.val_metrics['auc:val'].update_state(metric.auc_score(step_res['gt'].cpu().numpy(), step_res['prob'].cpu().numpy()[:,1]), step_res['batch_size'])

            for j in range(len(gender)):
                gender_str = 'male' if gender[j] == 0 else 'female'
                probs[gender_str].append(step_res['prob'][j].unsqueeze(0))
                gts[gender_str].append(step_res['gt'][j].unsqueeze(0))
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=False, mode=mode)
            args.val_metrics[f'loss:val:{mode}'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics[f'acc:val:{mode}'].update_state(step_res['acc'], step_res['batch_size'])
            args.val_metrics[f'balanced_acc:val:{mode}'].update_state(step_res['balanced_acc'], step_res['batch_size'])
            
            overall_ece = (ECELoss()(step_res['prob'], label) * 100).item()
            predictions = np.argmax(step_res['prob'].cpu().numpy(), axis=1)
            
            if mode == 'full':
                # Collect data; ensure they are detached and moved to CPU if necessary
                for label, pred, prob, gend, img_id in zip(label, predictions, step_res['prob'].cpu().numpy(), gender.cpu().numpy(), id):
                    
                    csv_output_dict.append({
                        'Ground Truth': label.item(),
                        'Prediction': pred,
                        'Probability Class 0': prob[0],
                        'Probability Class 1': prob[1],
                        'Gender': gend,
                        'Path': img_id
                    })
                
            args.val_metrics[f'ece:val:{mode}'].update_state(overall_ece, 1)
            args.val_metrics[f'f1:val:{mode}'].update_state(f1_score(step_res['gt'].cpu().numpy(), predictions, average='weighted'), step_res['batch_size'])
            args.val_metrics[f'tpr:val:{mode}'].update_state(metric.tpr_score(step_res['gt'].cpu().numpy(), predictions), step_res['batch_size'])
            args.val_metrics[f'auc:val:{mode}'].update_state(metric.auc_score(step_res['gt'].cpu().numpy(), step_res['prob'].cpu().numpy()[:,1]), step_res['batch_size'])

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

    male_balanced_acc = metric.balanced_acc_fcn(male_probs.argmax(-1), male_gts)
    female_balanced_acc = metric.balanced_acc_fcn(female_probs.argmax(-1), female_gts)

    female_ece = (ECELoss()(female_probs, female_gts) * 100).item()
    
    if args.train_method == 'fchead':
        args.val_metrics[f'acc:val:male'].update_state(male_acc * 100, 1)
        args.val_metrics[f'ece:val:male'].update_state(male_ece, 1)
        args.val_metrics[f'acc:val:female'].update_state(female_acc * 100, 1)
        args.val_metrics[f'balanced_acc:val:male'].update_state(male_balanced_acc*100, 1)
        args.val_metrics[f'balanced_acc:val:female'].update_state(female_balanced_acc * 100, 1)
        args.val_metrics[f'ece:val:female'].update_state(female_ece, 1)
        male_predictions = np.argmax(male_probs_np, axis=1)
        female_predictions = np.argmax(female_probs_np, axis=1)
        
        args.val_metrics[f'f1:val:male'].update_state(f1_score(male_gts_np, male_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:male'].update_state(metric.tpr_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:male'].update_state(metric.auc_score(male_gts_np, male_probs_np[:,1]), step_res['batch_size'])
        args.val_metrics[f'f1:val:female'].update_state(f1_score(female_gts_np, female_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:female'].update_state(metric.tpr_score(female_gts_np, female_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:female'].update_state(metric.auc_score(female_gts_np,  female_probs_np[:,1]), step_res['batch_size'])
        return args.val_metrics['balanced_acc:val'].result(), args.val_metrics['loss:val'].result(), csv_output_dict
    else:
        male_predictions = np.argmax(male_probs_np, axis=1)
        female_predictions = np.argmax(female_probs_np, axis=1)
        args.val_metrics[f'acc:val:{mode}:male'].update_state(male_acc * 100, 1)
        args.val_metrics[f'ece:val:{mode}:male'].update_state(male_ece, 1)
        args.val_metrics[f'acc:val:{mode}:female'].update_state(female_acc * 100, 1)
        args.val_metrics[f'ece:val:{mode}:female'].update_state(female_ece, 1)
        args.val_metrics[f'balanced_acc:val:{mode}:male'].update_state(male_balanced_acc*100, 1)
        args.val_metrics[f'balanced_acc:val:{mode}:female'].update_state(female_balanced_acc * 100, 1)
        args.val_metrics[f'f1:val:{mode}:male'].update_state(f1_score(male_gts_np, male_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:{mode}:male'].update_state(metric.tpr_score(male_gts_np, male_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:{mode}:male'].update_state(metric.auc_score(male_gts_np,  male_probs_np[:,1]), step_res['batch_size'])
        args.val_metrics[f'f1:val:{mode}:female'].update_state(f1_score(female_gts_np, female_predictions, average='weighted'), step_res['batch_size'])
        args.val_metrics[f'tpr:val:{mode}:female'].update_state(metric.tpr_score(female_gts_np, female_predictions), step_res['batch_size'])
        args.val_metrics[f'auc:val:{mode}:female'].update_state(metric.auc_score(female_gts_np,  female_probs_np[:,1]), step_res['batch_size'])

        return args.val_metrics[f'balanced_acc:val:{mode}'].result(), args.val_metrics[f'loss:val:{mode}'].result(), csv_output_dict

def fc_step(batch, network, criterion, optimizer, args, is_train=True):
    '''Train/val for one step.'''
    img, label, gender, id = batch
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
        balanced_acc = metric.balanced_acc_fcn(output.argmax(-1), label)
        #emily use prob and gt to create the dictionary

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc * 100, \
            'balanced_acc': balanced_acc * 100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}


def nw_step(batch, network, criterion, optimizer, args, is_train=True, mode='random'):
    '''Train/val for one step.'''
    img, label, gender, id = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    gender = gender.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        if is_train:
            # Apply dropout during training
            output = network(img, gender)
            # output = F.dropout(output, p=0.5, training=True)  # Applying dropout
        else:
            output = network.predict(img, mode)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)
        balanced_acc = metric.balanced_acc_fcn(output.argmax(-1), label)

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc * 100, \
            'balanced_acc': balanced_acc * 100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}

if __name__ == '__main__':
    main()