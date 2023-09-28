import numpy as np
import torch
import torch.nn as nn
from dgl import backend as F
import torch.nn.functional as torchF
import argparse
from dgl.data.fraud import FraudDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
import dgl
from dgl.nn.pytorch import GraphConv
from sklearn import metrics
from collections import namedtuple
from tqdm import tqdm
import os.path as osp

def row_normalize(mx, dtype=np.float32):
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

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def prob2pred(logits_fraud, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(logits_fraud, dtype=np.int32)  
    y_pred[logits_fraud >= thres] = 1
    y_pred[logits_fraud < thres] = 0
    return y_pred

def convert_probs(labels, logits, threshold_moving=True, thres=0.5):
    logits = torch.nn.Sigmoid()(logits)   
    logits = logits.detach().cpu().numpy() # logits
    
    logits_fraud = logits[:, 1]          
    if threshold_moving:
        preds = prob2pred(logits_fraud, thres=thres)
    else:
        preds = logits.argmax(axis=1)  

    return labels, logits_fraud, preds

def calc_roc_and_thres(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    auc_gnn = metrics.auc(fpr, tpr)

    J = tpr - fpr
    ks_val = max(abs(J))
    
    idx = J.argmax(axis=0)
    best_thres = thresholds[idx]
    return auc_gnn, best_thres

def calc_ap_and_thres(y_true, y_prob):
    ap_gnn = metrics.average_precision_score(y_true, y_prob)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    F1 = 2 * precision * recall / (precision + recall)
    idx = F1.argmax(axis=0)
    best_thres = thresholds[idx]

    return ap_gnn, best_thres

def calc_acc(y_true, y_pred):
    """
    Compute the accuracy of prediction given the labels.
    """
    return metrics.accuracy_score(y_true, y_pred)


def calc_f1(y_true, y_pred):
    f1_binary_1_gnn = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    f1_binary_0_gnn = metrics.f1_score(y_true, y_pred, pos_label=0, average='binary')
    f1_micro_gnn = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro_gnn = metrics.f1_score(y_true, y_pred, average='macro')

    return f1_binary_1_gnn, f1_binary_0_gnn, f1_micro_gnn, f1_macro_gnn

def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

def eval_model(y_true, y_prob, y_pred):
    """
    :param y_true: torch.Tensor
    :param y_prob: torch.Tensor
    :param y_pred: torch.Tensor
    :return: namedtuple
    """
    acc = calc_acc(y_true, y_pred)
    f1_binary_1, f1_binary_0, f1_micro, f1_macro = calc_f1(y_true, y_pred)

    auc_gnn, best_roc_thres = calc_roc_and_thres(y_true, y_prob)
    # auc_gnn = metrics.roc_auc_score(labels, probs_1)

    ap_gnn, best_pr_thres = calc_ap_and_thres(y_true, y_prob)

    precision_1 = metrics.precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall_1 = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')

    conf_gnn = metrics.confusion_matrix(y_true, y_pred)
    gmean_gnn = calc_gmean(conf_gnn)
    tn, fp, fn, tp = conf_gnn.ravel()


    DataType = namedtuple('Metrics', ['f1_binary_1', 'f1_binary_0', 'f1_macro', 'auc_gnn',
                                      'gmean_gnn', 'recall_1', 'precision_1', 'ap_gnn',
                                      'best_roc_thres', 'best_pr_thres', 'recall_macro'])
    # 1:fraud->positive, 0:benign->negtive
    results = DataType(f1_binary_1=f1_binary_1, f1_binary_0=f1_binary_0, f1_macro=f1_macro,
                       auc_gnn=auc_gnn, gmean_gnn=gmean_gnn, ap_gnn=ap_gnn,
                       recall_1=recall_1, precision_1=precision_1, recall_macro=recall_macro,
                       best_pr_thres=best_pr_thres, best_roc_thres=best_roc_thres)

    return results

def save_checkpoint(model, save_path):
    """Saves model when validation loss decreases."""
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, save_path):
    """Load the latest checkpoint."""
    model.load_state_dict(torch.load(save_path))
    return model