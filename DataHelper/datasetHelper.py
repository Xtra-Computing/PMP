from .dataset import dataset
from dgl.data.fraud import FraudDataset, FraudAmazonDataset
from dgl.data.citation_graph import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
from scipy import sparse as sp
import torch
import dgl
import os.path as osp
from torch.utils.data import Dataset, Subset, DataLoader
from dgl.dataloading import DataLoader as DGLDataLoader
from dgl.dataloading import (
	MultiLayerFullNeighborSampler,
	NeighborSampler,
)
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split
from dgl import backend as F
from dgl import AddSelfLoop, RowFeatNormalizer
from dgl import transforms as T

class DatasetHelper(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    mask = None
    feat_transform = None
    recache = False

    def __init__(self, config, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.config = config


            
    def load(self):
        homo = self.config.get('homo', False)
        self.train_set, self.val_set, self.test_set = (None, None, None)
        
        if self.dataset_name in ['amazon', 'yelp']:
            dataset = FraudDataset(name=self.dataset_name,
                                   raw_dir = self.dataset_source_folder_path,
                                   train_size=self.config['train_size'],
                                   val_size=self.config['val_size'],
                                   random_seed=self.config['dataset_seed'], 
                                   force_reload=self.config['force_reload'])
            if self.dataset_name == 'amazon' and self.config.get('BWGNN_split', False) and self.config['train_size'] in [0.01, 0.001]:
                dataset = FraudAmazonDataset()

        elif self.dataset_name in ['tfinance', 'tsocial']:
            dataset, label_dict = load_graphs('datasets/{}'.format(self.dataset_name))

        data = dataset[0]
        norm = self.config['norm_feat']
        # Feature tensor dtpye is float64, change it to float32
        data.ndata['feature'] = torch.from_numpy(self.row_normalize(data.ndata['feature'], dtype=np.float32)) if norm else data.ndata['feature'].float()  
        if self.dataset_name == 'grab':
            labels = data.ndata['label']
            labels[labels==2] = 0
            data.ndata['label'] = labels
        if self.dataset_name in ['tfinance', 'tsocial', 'grab']:
            if self.dataset_name == 'tfinance':
                data.ndata['label'] = data.ndata['label'].argmax(1)
            labels = data.ndata['label']
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index,  # train_data 
                                                                    labels[index],   # train_target
                                                                    stratify     = labels[index],
                                                                    train_size   = self.config['train_size'],
                                                                    random_state = 2, 
                                                                    shuffle      = True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, 
                                                                    y_rest, 
                                                                    stratify     = y_rest,  # val 和test中的不同类别比例 保持y_rest中相同的比例分配
                                                                    test_size    = 0.67,
                                                                    random_state = 2, 
                                                                    shuffle      = True)
            train_mask = torch.zeros([len(labels)]).bool()
            val_mask   = torch.zeros([len(labels)]).bool()
            test_mask  = torch.zeros([len(labels)]).bool()

            train_mask[idx_train]    = 1
            val_mask[idx_valid]      = 1
            test_mask[idx_test]      = 1
            data.ndata["train_mask"] = F.tensor(train_mask)
            data.ndata["val_mask"]   = F.tensor(val_mask)
            data.ndata["test_mask"]  = F.tensor(test_mask)


        if self.dataset_name == 'amazon' and self.config.get('BWGNN_split', False) and self.config['train_size']==0.01:
            labels = data.ndata['label']
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index,  # train_data 
                                                                    labels[index],   # train_target
                                                                    stratify     = labels[index],
                                                                    train_size   = self.config['train_size'],
                                                                    random_state = 2, 
                                                                    shuffle      = True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, 
                                                                    y_rest, 
                                                                    stratify     = y_rest,  # val 和test中的不同类别比例 保持y_rest中相同的比例分配
                                                                    test_size    = 0.67,
                                                                    random_state = 2, 
                                                                    shuffle      = True)
            train_mask = torch.zeros([len(labels)]).bool()
            val_mask   = torch.zeros([len(labels)]).bool()
            test_mask  = torch.zeros([len(labels)]).bool()

            train_mask[idx_train]    = 1
            val_mask[idx_valid]      = 1
            test_mask[idx_test]      = 1
            data.ndata["train_mask"] = F.tensor(train_mask)
            data.ndata["val_mask"]   = F.tensor(val_mask)
            data.ndata["test_mask"]  = F.tensor(test_mask)            

        relations =list(data.etypes)  # relation name
        if self.config['add_self_loop']:
            for etype in relations:
                data = dgl.remove_self_loop(data, etype=etype)
                data = dgl.add_self_loop(data, etype=etype)
            print('add self loops')
        self.config_data(data, dataset)
        if homo:
            data = dgl.to_homogeneous(data, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            if self.config['add_self_loop']:
                data = dgl.remove_self_loop(data)
                data = dgl.add_self_loop(data) 

        if self.config['model'] in ['LA-SAGE2', 'LA-SAGE-LI', 'LA-SAGE-S']:
            full_relations = data.canonical_etypes  # [('review', 'net_rsr', 'review'), ('review', 'net_rtr', 'review'), ('review', 'net_rur', 'review')]
            sampled_neighbors = self.config['sampled_neighbors']
            train_fanouts = [] 
            for i in range(self.config['n_layer']):
                train_fanouts.append({etype: sampled_neighbors[i] for etype in full_relations})
            label_unk = (torch.ones(self.num_nodes)*2).long()
            label_unk[self.train_nid] = self.labels[self.train_nid]  # train benign nodes label = 0 train fraud nodes label = 1 others=2
            self.data.ndata['label_unk'] = label_unk

            if self.config.get('full_neighbors', False):
                sampler = MultiLayerFullNeighborSampler(num_layers=self.config['n_layer'])
            else:
                sampler = NeighborSampler(train_fanouts if not self.config['homo'] else sampled_neighbors,
                                          prefetch_node_feats=["feature"],
                                          prefetch_labels=["label"]) 
            self.train_loader, self.val_loader, self.test_loader = self.get_DGLloader(self.data, sampler)
     
    def get_DGLloader(self, data, sampler):
        train_loader = DGLDataLoader(data,
                                     self.train_nid,
                                     sampler,
                                     batch_size = self.config['batch_size'],
                                     shuffle = True,
                                     drop_last = False,
                                    #  device=torch.cuda.current_device(),
                                     num_workers=8)
        val_loader   = DGLDataLoader(data,
                                     self.val_nid,
                                     sampler,
                                     batch_size = self.config.get('val_batch_size', self.config['batch_size']),
                                     shuffle=False,
                                     drop_last=False,
                                    #  device=torch.cuda.current_device(),
                                     num_workers=8)
        test_loader  = DGLDataLoader(data,
                                     self.test_nid,
                                     sampler,
                                     batch_size = self.config.get('test_batch_size', self.config['batch_size']),
                                     shuffle=False,
                                     drop_last=False,
                                    #  device=torch.cuda.current_device(),
                                     worker_init_fn = lambda id: np.random.seed(1234),
                                     num_workers=0)
        return train_loader, val_loader, test_loader

    def get_data_loader(self, data_sets):
        train_set, val_set, test_set = data_sets
        train_loader = DataLoader(train_set, batch_size = self.config['batch_size'], 
                                                shuffle = True,
                                              drop_last = False,            
                                            num_workers = 0)
        val_loader   = DataLoader(val_set,   batch_size = self.config['batch_size'], 
                                                shuffle = False,
                                              drop_last = False, 
                                            num_workers = 0)
        test_loader  = DataLoader(test_set,  batch_size = self.config['batch_size'], 
                                                shuffle = False,
                                              drop_last = False, 
                                            num_workers = 0)
        return train_loader, val_loader, test_loader
        
        
    def row_normalize(self, mx, dtype=np.float32):
        r"""Row-normalize sparse matrix.
        Reference: <https://github.com/williamleif/graphsage-simple>
        
        Parameters
        ----------
        mx : np.ndarray
            Feature matrix of all nodes.
        dtype : np.dtype
            Data type for normalized features. Default=np.float32

        Return : np.ndarray
            Normalized features.
        """
        rowsum    = np.array(mx.sum(1)) + 0.01
        r_inv     = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx        = r_mat_inv.dot(mx)
        
        return mx.astype(dtype)


    def split_dataset(self, nodes_dataset):
        nodes_dataset = FDDataset(nodes_dataset, self.labels)
        train_set     = Subset(nodes_dataset, self.train_nid)  # train_dataset
        val_set       = Subset(nodes_dataset, self.val_nid)      # val_dataset
        test_set      = Subset(nodes_dataset, self.test_nid)    # test_dataset
        return train_set, val_set, test_set

    def config_data_C(self, data, dataset):
        self.data = data
        self.dataset = dataset
        self.train_mask      = data.ndata['train_mask']
        self.val_mask        = data.ndata['val_mask']
        self.test_mask       = data.ndata['test_mask']
        self.train_nid       = torch.LongTensor(torch.nonzero(self.train_mask, as_tuple=True)[0])
        self.val_nid         = torch.LongTensor(torch.nonzero(self.val_mask, as_tuple=True)[0])
        self.test_nid        = torch.LongTensor(torch.nonzero(self.test_mask, as_tuple=True)[0])
        self.num_classes     = dataset.num_classes
        self.feat            = data.ndata['feat']
        self.labels          = data.ndata['label']
        self.feat_dim        = self.feat.shape[1]
        self.num_nodes       = data.num_nodes()


    def config_data(self, data, dataset):
        self.data            = data
        self.dataset         = dataset
        self.train_mask      = data.ndata['train_mask']
        self.val_mask        = data.ndata['val_mask']
        self.test_mask       = data.ndata['test_mask']
        self.train_nid       = torch.LongTensor(torch.nonzero(self.train_mask, as_tuple=True)[0])
        self.val_nid         = torch.LongTensor(torch.nonzero(self.val_mask, as_tuple=True)[0])
        self.test_nid        = torch.LongTensor(torch.nonzero(self.test_mask, as_tuple=True)[0])
        self.num_classes     = dataset.num_classes if not self.dataset_name in ['tfinance', 'tsocial', 'grab'] else len(torch.unique(data.ndata['label']))
        self.relations       = list(data.etypes)
        self.num_relations   = len(data.etypes)
        self.feat            = data.ndata['feature']  
        self.feat_dim        = self.feat.shape[1]      
        self.labels          = data.ndata['label'].squeeze().long()
        self.num_nodes       = self.labels.shape[0]
        print(f"[Global] Dataset <{self.dataset_name}> Overview\n"
        #   f"\tAverage in-degree {sum(self.data.in_degrees):>6}, Average out-degree {sum(self.data.out_degrees):>6} \n"
          f"\t Num Edges {data.number_of_edges():>6}\n"
          f"\t Num Features {self.feat_dim:>6}\n"
          f"\tEntire (fraud/total) {torch.sum(self.labels):>6} / {self.labels.shape[0]:<6}\n"
          f"\tTrain  (fraud/total) {torch.sum(self.labels[self.train_nid]):>6} / {self.labels[self.train_nid].shape[0]:<6}\n"
          f"\tValid  (fraud/total) {torch.sum(self.labels[self.val_nid]):>6} / {self.labels[self.val_nid].shape[0]:<6}\n"
          f"\tTest   (fraud/total) {torch.sum(self.labels[self.test_nid]):>6} / {self.labels[self.test_nid].shape[0]:<6}\n")



class FDDataset(Dataset):
    def __init__(self, data, labels):
        super(FDDataset, self).__init__()
        # (N,S,E)
        self.data = data
        self.labels = labels
        self.n_samples = labels.shape[0]

    def __getitem__(self, idx: torch.LongTensor):
        batch_seq = self.data[idx]
        batch_labels = self.labels[idx]
        return batch_seq, batch_labels

    def __len__(self):
        return self.n_samples