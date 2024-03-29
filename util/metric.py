import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import math 

def check_type(x):
    return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

def acc(pred, targets):
    '''Returns accuracy given batch of categorical predictions and targets.'''
    pred = check_type(pred)
    targets = check_type(targets)
    return accuracy_score(targets, pred)

def roc(pr, gt):
    """pr  : prediction, B 
        gt  : ground-truth binary mask, B."""
    pr = check_type(pr)
    gt = check_type(gt)
    return 100 * float(roc_auc_score(gt, pr))

def balanced_acc_fcn(preds, gts):
    balanced_acc = balanced_accuracy_score(gts.cpu().numpy(), preds.cpu().numpy())
    return balanced_acc

def tpr_score(y_true, y_pred):
    # Calculate True Positive Rate (TPR)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    return recall

def auc_score(y_true, y_pred_proba):
    # Calculate Area Under the Curve (AUC)
    try:
        auc_value = roc_auc_score(y_true, y_pred_proba)
        return auc_value
    # If y_true contains only one class AUC is not defined (relevant for unbalanced classes like Fracture)
    except ValueError:
        return 0.0

def support_influence(softmaxes, qlabels, sweights, slabels):
    '''
    Influence is defined as L(rescaled_softmax, qlabel) - L(softmax, qlabel).
    Positive influence => removing support image increases loss => support image was helpful
    Negative influence => removing support image decreases loss => support image was harmful
    bs should be 1.
    
    softmaxes: (bs, num_classes)
    qlabel: One-hot encoded query label (bs, num_classes)
    sweights: Weights between query and each support (bs, num_support)
    slabels: One-hot encoded support label (bs, num_support, num_classes)
    '''
    batch_influences = []
    bs = len(softmaxes)
    for bid in range(bs):
        softmax = softmaxes[bid]
        qlabel = qlabels[bid]
        sweight = sweights[bid]

        qlabel_cat = qlabel.argmax(-1).item()
        slabels_cat = slabels.argmax(-1)
        
        p = softmax[qlabel_cat]
        indicator = (slabels_cat==qlabel_cat).long()
        influences = torch.log((p - p*sweight)/(p - sweight*indicator))

        batch_influences.append(influences[None])
    return torch.cat(batch_influences, dim=0)

class Metric:
    def __init__(self) -> None:
        self.tot_val = 0
        self.num_samples = 0

    def update_state(self, val, samples):
        if isinstance(val, torch.Tensor):
            val = val.cpu().detach().item()
        if isinstance(val, np.ndarray):
            val = val.item()
        self.num_samples += samples
        self.tot_val += (val * samples)
  
    def result(self):
        if self.num_samples == 0:
            return 0
        return self.tot_val / self.num_samples
    
    def reset_state(self):
        self.tot_val = 0
        self.num_samples = 0

# https://github.com/gpleiss/temperature_scaling
class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class SmoothNLLLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, log_preds, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, log_preds.size(-1), self.smoothing)
        # log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))
    
    
def define_train_eval_metrics(train_method):
    # Tracking metrics
    list_of_metrics = [
        'loss:train',
        'balanced_acc:train',
        'acc:train',
    ]
    if train_method == 'nwhead':
        list_of_val_metrics = [
            'loss:val:random',
            'loss:val:full',
            'loss:val:cluster',
            'loss:val:ensemble',
            'loss:val:knn',
            'loss:val:hnsw',
            'acc:val:random',
            'acc:val:full',
            'acc:val:cluster',
            'acc:val:ensemble',
            'acc:val:knn',
            'acc:val:hnsw',
            'acc:val:random:male',
            'acc:val:full:male',
            'acc:val:cluster:male',
            'acc:val:random:female',
            'acc:val:full:female',
            'acc:val:cluster:female',
            'balanced_acc:val:random',
            'balanced_acc:val:full',
            'balanced_acc:val:cluster',
            'balanced_acc:val:ensemble',
            'balanced_acc:val:knn',
            'balanced_acc:val:hnsw',
            'ece:val:random',
            'ece:val:full',
            'ece:val:cluster',
            'ece:val:ensemble',
            'ece:val:knn',
            'ece:val:hnsw',
            'balanced_acc:val:random:male',
            'balanced_acc:val:full:male',
            'balanced_acc:val:cluster:male',
            'balanced_acc:val:random:female',
            'balanced_acc:val:full:female',
            'balanced_acc:val:cluster:female',
            'ece:val:random:male',
            'ece:val:full:male',
            'ece:val:cluster:male',
            'ece:val:ensemble:male',
            'ece:val:knn:male',
            'ece:val:hnsw:male',
            'ece:val:random:female',
            'ece:val:full:female',
            'ece:val:cluster:female',
            'ece:val:ensemble:female',
            'ece:val:knn:female',
            'ece:val:hnsw:female',
            'acc:val:ensemble:male',
            'acc:val:knn:male',
            'acc:val:hnsw:male',
            'balanced_acc:val:ensemble:male',
            'balanced_acc:val:knn:male',
            'balanced_acc:val:hnsw:male',
            'acc:val:ensemble:female',
            'acc:val:knn:female',
            'acc:val:hnsw:female',
            'balanced_acc:val:ensemble:female',
            'balanced_acc:val:knn:female',
            'balanced_acc:val:hnsw:female',
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
            'balanced_acc:val',
            'acc:val:female',
            'balanced_acc:val:female',
            'balanced_acc:val:male',
            'ece:val:female',
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
    
    return list_of_metrics, list_of_val_metrics