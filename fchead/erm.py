import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from transformers import get_scheduler
from fchead.optimizers import get_optimizers
import fchead.networks as networks

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)
    def forward(self, x):
        return self.network(x)