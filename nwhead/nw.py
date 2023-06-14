import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from .support import SupportSet
from .kernel import get_kernel
from .utils import linear_normalization
from torch.nn.init import xavier_uniform_

class NWNet(nn.Module):
    def __init__(self, 
                 featurizer, 
                 support_dataset, 
                 num_classes,
                 feat_dim,
                 kernel_type='euclidean', 
                 train_type='random', 
                 env_array=None, 
                 num_per_class=1, 
                 total_per_class=100,
                 subsample_classes=None,
                 embed_dim=0, 
                 finetune_softfeatmask=False,
                 finetune_proj=False,
                 debug_mode=False,
                 device='cuda:0', 
                 use_nll_loss=False
                 ):
        '''Top level NW net class. Creates kernel, NWHead, and support set as modules.

        Args:
            featurizer: Feature extractor
            support_dataset: Pytorch Dataset object. Assumes has attributes
                .targets containing categorical classes of dataset.
            d_out: Output dimension of featurizer'''
        super(NWNet, self).__init__()
        self.featurizer = featurizer
        self.train_type = train_type
        self.device = device
        self.debug_mode = debug_mode
        self.use_nll_loss = use_nll_loss
        self.num_classes = num_classes
        assert hasattr(support_dataset, 'targets'), 'Support set must have .targets attribute'

        # Kernel
        kernel = get_kernel(kernel_type)


        if finetune_softfeatmask:
            assert embed_dim == 0
            # Freeze the weights of featurizer
            for param in self.featurizer.parameters():
                param.requires_grad = False

        if finetune_proj:
            assert embed_dim > 0
            # Freeze the weights of featurizer
            for param in self.featurizer.parameters():
                param.requires_grad = False

        # NW Head
        self.nwhead = NWHead(kernel=kernel,
                             feat_dim=feat_dim,
                             embed_dim=embed_dim,
                             apply_softfeatmask=finetune_softfeatmask)

        # Support dataset
        self.sset = SupportSet(support_dataset,
                               train_type,
                               num_per_class,
                               total_per_class,
                               self.num_classes,
                               subsample_classes=subsample_classes,
                               env_array=env_array)

    def precompute(self):
        '''Precomputes all support features, cluster centroids, and 
           random iterator. Call before running inference.'''
        assert not self.featurizer.training
        sinfo = self._compute_all_support_feats()
        self.sset.update_feats(*sinfo)

    def predict(self, x, mode='random'):
        qfeat = self.featurizer(x)
        sfeat, sy = self.sset.get_infer_support(mode)
        if mode == 'ensemble':
            outputs = 0
            num_envs = len(sfeat)
            for env_feat, env_y in zip(sfeat, sy):
                env_feat, env_y = env_feat.to(
                    x.device), env_y.to(x.device)
                output = self._forward(qfeat, env_feat, env_y)
                if self.use_nll_loss:
                    outputs += output.exp()
                else:
                    outputs += output.log()
            if self.use_nll_loss:
                return torch.log(outputs / num_envs), torch.full((len(x),), True)
            else:
                return (outputs / num_envs).exp(), torch.full((len(x),), True)
        else:
            sfeat, sy = sfeat.to(x.device), sy.to(x.device)
            return self._forward(qfeat, sfeat, sy), torch.full((len(x),), True)

    def forward(self, x, y, metadata=None):
        '''Forward pass using images for query and support.
        Support set is some random subset of support dataset.'''
        sx, sy, sm = self.sset.get_train_support(y, metadata)
        sx, sy = sx.to(x.device), sy.to(x.device)

        batch_size = len(x)
        inputs = torch.cat((x, sx), dim=0)
        feats = self.featurizer(inputs)
        qfeat, sfeat = feats[:batch_size], feats[batch_size:]
        
        isin = torch.isin(y, sy)
        if self.debug_mode:
            qgrid = torchvision.utils.make_grid(linear_normalization(x), nrow=8)
            sgrid = torchvision.utils.make_grid(linear_normalization(sx), nrow=8)
            print('qbatch shape:', x.shape)
            print('sbatch shape:', sx.shape)
            plt.imshow(qgrid.permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            plt.imshow(sgrid.permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            print('qy:', y)
            print('sy:', sy)
            if metadata is not None:
                print('qmeta:', metadata)
                print('smeta:', sm)
            print('qy in sy:', isin)

        return self._forward(qfeat, sfeat, sy), isin
    
    def _forward(self, qfeat, sfeat, sy):
        '''Forward pass on features'''
        batch_size = len(qfeat)
        sy = F.one_hot(sy, self.num_classes).float()
        sfeat = sfeat[None].expand(batch_size, *sfeat.shape)
        sy = sy[None].expand(batch_size, *sy.shape)
        output = self.nwhead(qfeat, sfeat, sy)
        if self.use_nll_loss:
            return torch.log(output+1e-12)
        else:
            return output.exp()

    def _compute_all_support_feats(self, save=False):
        # print('Precomputing all features in support dataset...')
        feats = []
        labels = []
        meta = []
        separated_feats = []
        separated_labels = []
        separated_meta = []
        for loader in self.sset.support_loaders:
            env_feats = []
            env_labels = []
            env_meta = []
            for qimg, qlabel, qmeta in loader:
                qimg = qimg.to(self.device)
                feat = self.featurizer(qimg).cpu().detach()
                feats.append(feat)
                labels.append(qlabel.cpu().detach())
                meta.append(qmeta)
                env_feats.append(feat)
                env_labels.append(qlabel.cpu().detach())
                env_meta.append(qmeta)
            env_feats = torch.cat(env_feats, dim=0)
            env_labels = torch.cat(env_labels, dim=0)
            env_meta = torch.cat(env_meta, dim=0)
            separated_feats.append(env_feats)
            separated_labels.append(env_labels)
            separated_meta.append(env_meta)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        meta = torch.cat(meta, dim=0)

        # if save:
        #     print('Saving embeddings...')
        #     root_dir = os.path.join(self.run_dir, 'embedding')
        #     if not os.path.exists(root_dir):
        #         os.makedirs(root_dir)
        #     embedding_path = os.path.join(root_dir, 'embeddings.npy')
        #     label_path = os.path.join(root_dir, 'labels.npy')
        #     metadata_path = os.path.join(root_dir, 'metadata.npy')
        #     np.save(embedding_path, embeddings.cpu().detach().numpy())
        #     np.save(label_path, labels.cpu().detach().numpy())
        #     np.save(metadata_path, metadata.cpu().detach().numpy())

        return feats, labels, meta, separated_feats, separated_labels, separated_meta

class NWHead(nn.Module):
    def __init__(self, 
                 kernel, 
                 feat_dim,
                 embed_dim=0, 
                 apply_softfeatmask=False,
                 device='cuda:0', 
                 dtype=torch.float32):
        super(NWHead, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.kernel = kernel
        self.embed_dim = embed_dim
        self.apply_softfeatmask = apply_softfeatmask

        if self.embed_dim > 0:
            self.proj_weight = nn.Parameter(torch.empty((1, feat_dim, embed_dim), **factory_kwargs))
            xavier_uniform_(self.proj_weight)
        
        if self.apply_softfeatmask:
            if self.embed_dim > 0:
                self.softfeatmask = nn.Parameter(torch.ones((1, 1, embed_dim), **factory_kwargs))
            else:
                self.softfeatmask = nn.Parameter(torch.ones((1, 1, feat_dim), **factory_kwargs))
            self.softfeatmask.requires_grad = True
        
    def embed(self, x):
        bs = len(x)
        return torch.bmm(x, self.proj_weight.repeat(bs, 1, 1)) 

    def forward(self, x, support_x, support_y):
        """
        Computes Nadaraya-Watson head on query, key and value tensors.
        Args:
            q: query (b, embed_dim)
            k: key (b, num_support, embed_dim)
            v: labels (b, num_support, num_classes)

        Output:
            output: softmaxed probabilities (b, num_classes)
            attn_probs: Weights per support element, (b, num_support)
            attn_scores: Scores per support element, (b, num_support)
        """
        x = x.unsqueeze(1)
        if self.embed_dim > 0:
            x = self.embed(x)
            support_x = self.embed(support_x)
        
        if self.apply_softfeatmask:
            squash_mask = torch.sigmoid(self.softfeatmask)
            x = squash_mask * x
            support_x = squash_mask * support_x

        scores = self.kernel(x, support_x)

        probs = F.softmax(scores, dim=-1)
        # (B, num_queries, num_keys) x (B, num_keys=num_vals, num_classes) -> (B, num_queries, num_classes)
        output = torch.bmm(probs, support_y)
        return output.squeeze(1)
