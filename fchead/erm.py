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
        all_a = np.array(all_a) if isinstance(all_a, list) else all_a

    # Check if all_a is a numpy array or tensor
        for a in np.unique(all_a):  # Use np.unique for numpy array or torch.unique for tensor
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
class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(IRM, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def _compute_loss(self, i, x, y, a, step):
        penalty_weight = self.hparams['irm_lambda'] \
            if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0
        nll = 0.
        penalty = 0.

        logits = self.network(x)
        for idx_a, idx_samples in self.return_attributes(a):
            nll += F.cross_entropy(logits[idx_samples], y[idx_samples])
            penalty += self._irm_penalty(logits[idx_samples], y[idx_samples])
        a = np.array(a) if isinstance(a, list) else a
        a_unique = np.unique(a)
        nll /= len(a_unique)
        penalty /= len(a_unique)
        loss_value = nll + (penalty_weight * penalty)

        self.update_count += 1
        return loss_value
    
class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(GroupDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        print("classes", self.num_classes)
        print("attributes", self.num_attributes)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).cuda())

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)
        # print("y", y.type, y) #tensor 
        
        # print("a", a) #list 
        a= torch.tensor(a, device=y.device)
        # a = a.astype(y.type)
        # print("return groups", self.return_groups(y, a))
        for idx_g, idx_samples in self.return_groups(y, a):
            # print("losses", losses[idx_samples])
            # print("self.q", self.q[idx_g])
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value

class LISA(ERM):
    """
    Improving Out-of-Distribution Robustness via Selective Augmentation [https://arxiv.org/pdf/2201.00299.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _to_ohe(self, y):
        return F.one_hot(y, num_classes=self.num_classes)

    def _lisa_mixup_data(self, s, a, x, y, alpha):
        if (not self.data_type == "images") or self.hparams['LISA_mixup_method'] == 'mixup':
            fn = self._mix_up
        elif self.hparams['LISA_mixup_method'] == 'cutmix':
            fn = self._cut_mix_up

        all_mix_x, all_mix_y = [], []
        bs = len(x)
        # repeat until enough samples
        while sum(list(map(len, all_mix_x))) < bs:
            start_len = sum(list(map(len, all_mix_x)))
            # same label, mixup between attributes
            if s:
                # can't do intra-label mixup with only one attribute
                if len(torch.unique(a)) < 2:
                    return x, y

                for y_i in range(self.num_classes):
                    mask = y[:, y_i].squeeze().bool()
                    a = a.to(mask.device)
                    x_i, y_i, a_i = x[mask], y[mask], a[mask]
                    unique_a_is = torch.unique(a_i)
                    if len(unique_a_is) < 2:
                        continue

                    # if there are multiple attributes, choose a random pair
                    a_i1, a_i2 = unique_a_is[torch.randperm(len(unique_a_is))][:2]
                    mask2_1 = a_i == a_i1
                    mask2_2 = a_i == a_i2
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            # same attribute, mixup between labels
            else:
                # can't do intra-attribute mixup with only one label
                if len(y.sum(axis=0).nonzero()) < 2:
                    return x, y

                for a_i in torch.unique(a):
                    mask = a == a_i
                    x_i, y_i = x[mask], y[mask]
                    unique_y_is = y_i.sum(axis=0).nonzero()
                    if len(unique_y_is) < 2:
                        continue

                    # if there are multiple labels, choose a random pair
                    y_i1, y_i2 = unique_y_is[torch.randperm(len(unique_y_is))][:2] 
                    mask2_1 = y_i[:, y_i1].squeeze().bool()
                    mask2_2 = y_i[:, y_i2].squeeze().bool()
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            end_len = sum(list(map(len, all_mix_x)))
            # each attribute only has one unique label
            if end_len == start_len:
                return x, y

        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

        shuffle_idx = torch.randperm(len(all_mix_x))
        return all_mix_x[shuffle_idx][:bs], all_mix_y[shuffle_idx][:bs]

    @staticmethod
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    @staticmethod
    def _mix_up(alpha, x1, x2, y1, y2):
        # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        n_classes = y1.shape[1]
        bsz = len(x1)
        l = np.random.beta(alpha, alpha, [bsz, 1])
        if len(x1.shape) == 4:
            l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        else:
            l_x = np.tile(l, (1, *x1.shape[1:]))
        l_y = np.tile(l, [1, n_classes])

        # mixed_input = l * x + (1 - l) * x2
        mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
        mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

        return mixed_x, mixed_y

    def _cut_mix_up(self, alpha, x1, x2, y1, y2):
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        input = torch.cat([x1, x2])
        target = torch.cat([y1, y2])

        rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

        lam = np.random.beta(alpha, alpha)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        return input, lam * target_a + (1-lam) * target_b

    def _compute_loss(self, i, x, y, a, step):
        s = np.random.random() <= self.hparams['LISA_p_sel']
        y_ohe = self._to_ohe(y)
        if self.data_type == "text":
            feats = self.featurizer(x)
            mixed_feats, mixed_y = self._lisa_mixup_data(s, a, feats, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.classifier(mixed_feats)
        else:
            mixed_x, mixed_y = self._lisa_mixup_data(s, a, x, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.predict(mixed_x)

        mixed_y_float = mixed_y.type(torch.FloatTensor)
        loss_value = F.cross_entropy(predictions, mixed_y_float.to(predictions.device))
        return loss_value
class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions using MMD
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                 grp_sizes=None, gaussian=False):
        super(AbstractMMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    @staticmethod
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def _compute_loss(self, i, x, y, a, step):
        all_feats = self.featurizer(x)
        outputs = self.classifier(all_feats)
        objective = F.cross_entropy(outputs, y)

        features = []
        for _, idx_samples in self.return_attributes(a):
            features.append(all_feats[idx_samples])

        penalty = 0.
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                penalty += self.mmd(features[i], features[j])

        if len(features) > 1:
            penalty /= (len(features) * (len(features) - 1) / 2)

        loss_value = objective + (self.hparams['mmd_gamma'] * penalty)
        return loss_value


class MMD(AbstractMMD):
    """MMD using Gaussian kernel"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(MMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=True)


class CORAL(AbstractMMD):
    """MMD using mean and covariance difference"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CORAL, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=False)


