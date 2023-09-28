import os.path as osp
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import torch
import random
from utils.logger import Logger, SummaryBox, Timer
from utils.utils import *
from utils.random_seeder import set_random_seed
from training_procedure import Trainer
from DataHelper.datasetHelper import DatasetHelper
from torch.utils.data import DataLoader
import pathlib
import utils.plot_tools as plot_tools
import warnings
import datetime
from sklearn.metrics import recall_score, roc_auc_score, roc_auc_score, precision_score, confusion_matrix
from torch.nn.functional import softmax
from collections import namedtuple
from imblearn.metrics import geometric_mean_score

warnings.filterwarnings('ignore') 

METRIC_NAME = ['auc_gnn', 
               'ap_gnn',
               'gmean_gnn',
               'recall_macro', 
               'f1_macro',
               'best_roc_thres', 
               'best_pr_thres',
               'f1_binary_1', 
               'f1_binary_0',
               'recall_1', 
               'precision_1']
def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5

def calc_acc(y_true, y_pred):
    """
    Compute the accuracy of prediction given the labels.
    """
    # return (y_pred == y_true).sum() * 1.0 / len(y_pred)
    return metrics.accuracy_score(y_true, y_pred)

def calc_f1(y_true, y_pred):
    f1_binary_1_gnn = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    f1_binary_0_gnn = metrics.f1_score(y_true, y_pred, pos_label=0, average='binary')
    f1_micro_gnn = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro_gnn = metrics.f1_score(y_true, y_pred, average='macro')

    return f1_binary_1_gnn, f1_binary_0_gnn, f1_micro_gnn, f1_macro_gnn

def calc_roc_and_thres(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    auc_list = []


    auc_gnn = metrics.auc(fpr, tpr)

    J = tpr - fpr
    ks_val = max(abs(J))
    
    idx = J.argmax(axis=0)
    best_thres = thresholds[idx]
    return auc_gnn, best_thres

def calc_ap_and_thres(y_true, y_prob):
    # \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n, 和AUPRC略有不同
    ap_gnn = metrics.average_precision_score(y_true, y_prob)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    F1 = 2 * precision * recall / (precision + recall)
    idx = F1.argmax(axis=0)
    best_thres = thresholds[idx]

    return ap_gnn, best_thres

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

def eval_model(y_true, y_prob, y_pred):
    """
    :param y_true: torch.Tensor tst_labels
    :param y_prob: torch.Tensor tst_fraud_prob
    :param y_pred: torch.Tensor
    :return: namedtuple
    """
    acc = calc_acc(y_true, y_pred)
    f1_binary_1, f1_binary_0, f1_micro, f1_macro = calc_f1(y_true, y_pred)

    auc_gnn, best_roc_thres = calc_roc_and_thres(y_true, y_prob)
    # auc_gnn = metrics.roc_auc_score(y_true, y_prob)

    ap_gnn, best_pr_thres = calc_ap_and_thres(y_true, y_prob)

    precision_1 = metrics.precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall_1 = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')

    # conf_gnn = metrics.confusion_matrix(y_true, y_pred)
    conf_gnn = metrics.confusion_matrix(y_true, y_pred)
    gmean_gnn = calc_gmean(conf_gnn)
    # tn, fp, fn, tp = conf_gnn.ravel()


    DataType = namedtuple('Metrics', ['f1_binary_1', 'f1_binary_0', 'f1_macro', 'auc_gnn',
                                      'gmean_gnn', 'recall_1', 'precision_1', 'ap_gnn',
                                      'best_roc_thres', 'best_pr_thres', 'recall_macro'])
    # 1:fraud->positive, 0:benign->negtive
    results = DataType(f1_binary_1=f1_binary_1, 
                       f1_binary_0=f1_binary_0, 
                       f1_macro=f1_macro,
                       auc_gnn=auc_gnn, 
                       gmean_gnn=gmean_gnn, 
                       ap_gnn=ap_gnn,
                       recall_1=recall_1, 
                       precision_1=precision_1, 
                       recall_macro=recall_macro,
                       best_pr_thres=best_pr_thres, 
                       best_roc_thres=best_roc_thres,
                       )

    return results

def convert_probs(labels, logits, threshold_moving=True, thres=0.5):
    logits = torch.nn.Sigmoid()(logits)   
    logits = logits.detach().cpu().numpy() 
    
    logits_fraud = logits[:, 1]          
    if threshold_moving:
        preds = prob2pred(logits_fraud, thres=thres)
    else:
        preds = logits.argmax(axis=1)  

    return labels, logits_fraud, preds

@torch.no_grad()
def evaluate(datasetHelper, 
            val_test_loader, 
            model, 
            threshold_moving=True,  
            thres = 0.5, 
            dataset = None):
    model.eval()

    logits_list = []
    label_list  = []  
    num_blocks = 0
    gmean = 0
    if config['model_name'] == 'GAGA':
        for (batch_seq, batch_labels) in val_test_loader:
            batch_seq    = batch_seq.cuda()
            batch_logits = model(batch_seq)
            logits_list.append(batch_logits.cpu())
            label_list .append(batch_labels.cpu())
    elif config['model_name'] in ['GraphSAGE', 'LA-SAGE', 'LA-SAGE2', 'LA-SAGE-LI', 'LA-SAGE-S']:
        relations = datasetHelper.relations
        for step, (input_nodes, output_nodes, blocks) in enumerate(val_test_loader):
            blocks = [b.to(torch.cuda.current_device()) for b in blocks]
            val_test_feats = blocks[0].srcdata['feature']
            val_test_label = blocks[-1].dstdata['label']
            batch_logits = model(blocks, relations, val_test_feats)

            logits_list.append(batch_logits.cpu())
            label_list .append(val_test_label.cpu())

    # shape=(len(eval_loader), 2)
    logits = torch.cat(logits_list, dim=0)  # predicted logits torch.Size([4595, 2])
    # for label alighment when using train_loader
    eval_labels   = torch.cat(label_list, dim=0)  # ground truth labels
    eval_labels   = eval_labels.detach().cpu().numpy()
    eval_labels, fraud_probs, preds = convert_probs(eval_labels, logits, threshold_moving=threshold_moving, thres=thres)
    geometric_mean_scores = geometric_mean_score(eval_labels, preds, average='micro')
    return eval_labels, fraud_probs, preds, geometric_mean_scores

def run_best_model(args, config, loaders, logger: Logger):
    T           = Trainer(config=config, args= args, logger= logger)
    model, _, _,_ = T.init(datasetHelper)
    best_model_path = config['best_model_path']
    best_model  = load_checkpoint(model, best_model_path)
    val_loader  = loaders[1]
    test_loader = loaders[2]
    labels, fraud_probs, preds, geometric_mean_scores = evaluate(datasetHelper, 
                                                val_loader, 
                                                best_model, 
                                                threshold_moving=config['threshold_moving'], 
                                                thres = config['thres'])
    best_dev_results = eval_model(labels, fraud_probs, preds)
    tst_labels, tst_fraud_prob, tst_preds,tst_geometric_mean_scores = evaluate(datasetHelper, 
                                                        test_loader, 
                                                        best_model, 
                                                        threshold_moving=config['threshold_moving'], 
                                                        thres = best_dev_results.best_pr_thres)
    final_test_results = eval_model(tst_labels, tst_fraud_prob, tst_preds)
    logger.append = ""
    val_string = "Best Validation Results"
    tst_string = "Final Test Results"
    logger.log("#" * (len(val_string)+2))
    logger.log("#Best Validation Results#")
    logger.log("#" * (len(val_string)+2))
    for metric in METRIC_NAME:
        # metric_list = np.around([getattr(result, metric) for result in final_test_results], decimals=5)
        metric_value = getattr(best_dev_results, metric)
        # logger.log("%s : %s" % (metric , str([round(x,4) for x in metric_list])))
        logger.log("%s : = %.4f" % (metric , metric_value))    
    logger.log("#" * (len(tst_string)+2))
    logger.log("#Final Test Results#")
    logger.log("#" * (len(tst_string)+2))
    for metric in METRIC_NAME:
        # metric_list = np.around([getattr(result, metric) for result in final_test_results], decimals=5)
        metric_value = getattr(final_test_results, metric)
        # logger.log("%s : %s" % (metric , str([round(x,4) for x in metric_list])))
        logger.log("%s : = %.4f" % (metric , metric_value))

    # AUC, GMEAN = evaluate(datasetHelper, 
    #                                                     test_loader, 
    #                                                     best_model, 
    #                                                     threshold_moving=config['threshold_moving'], 
    #                                                     thres = best_dev_results.best_pr_thres)
    # print(AUC, GMEAN)
    logger.log("gmean_micro : = %.4f" % (tst_geometric_mean_scores))



if __name__ == "__main__":
    parser = ArgumentParser()
    # test gaga model: python main.py --model GAGA --dataset yelp --gpu_id 2 --run_best
    # test GNN4FD:     python main.py --model GNN4FD --dataset yelp --gpu_id 2 --run_best
    parser.add_argument('--dataset',        type   = str,             default = 'yelp') 
    parser.add_argument('--num_workers', default   = 8,                  type = int, choices = [0, 8])
    parser.add_argument('--seed',        default   = 1234,               type = int, choices = [0, 1, 1234])
    parser.add_argument('--data_dir',       type   = str,             default = "datasets/") 
    parser.add_argument('--hyper_file',     type   = str,             default = 'config/')
    parser.add_argument('--log_dir',          type = str,             default = 'logs/')
    parser.add_argument('--best_model_path',  type = str,             default = 'checkpoints/')
    parser.add_argument('--train_size',     type   = float,           default = 0.4)
    parser.add_argument('--val_size',       type   = float,           default = 0.2)
    parser.add_argument('--no_dev',       action   = "store_true" ,   default = False)
    parser.add_argument('--gpu_id',         type   = int,             default = 0)
    parser.add_argument('--model',          type   = str,             default ='LA-SAGE-S')  # GCN, GAT or other

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    if args.model == 'LA-SAGE-S' and args.train_size == 0.4 and args.val_size == 0.2 and args.dataset == 'amazon':
        start_wall_time = '2023-07-03_23-52-09'
    if args.model == 'LA-SAGE-S' and args.train_size == 0.4 and args.val_size == 0.2 and args.dataset == 'yelp':
        start_wall_time = '2023-07-03_20-27-53'
    if args.model == 'LA-SAGE-S' and args.train_size == 0.4 and args.val_size == 0.2 and args.dataset == 'tfinance':
        start_wall_time = '2023-07-04_13-07-46'
    logger = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()

    if args.train_size == 0.01 and args.val_size == 0.1:
        config_path = osp.join(args.best_model_path, '_0.01', args.model ,args.dataset, start_wall_time, args.dataset+ '.yml')
    elif args.train_size == 0.4 and args.val_size == 0.1:
        config_path = osp.join(args.best_model_path, args.model ,args.dataset, start_wall_time, args.dataset+ '.yml')
    elif args.train_size == 0.0001 and args.val_size == 0.1: 
        config_path = osp.join(args.best_model_path, '_0.0001', args.model ,args.dataset, start_wall_time, args.dataset+ '.yml')
    elif args.train_size == 0.4 and args.val_size == 0.2:
        config_path = osp.join(args.best_model_path, 'val_0.2', args.model ,args.dataset, start_wall_time, args.dataset+ '.yml')

    
    config = get_config(config_path)
    model_name = args.model
    # config = config[model_name] 
    config['model_name'] = model_name
    config = args2config(args, config)

    dev_ress = []
    tes_ress = []
    tra_ress = []
    if config.get('seed',-1) > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    seeds = [random.randint(0,233333333) for _ in range(config['multirun'])]
    datasetHelper: DatasetHelper = load_data(args, config)
    datasetHelper.load()  # config dataset
    print_config(config)

    if args.train_size != 0.4:
        config['best_model_path'] = args.best_model_path + '_{}'.format(args.train_size)
    if args.val_size != 0.1:
        config['best_model_path'] = args.best_model_path + 'val_{}'.format(args.val_size)
    checkpoint_path_local = osp.join(config['best_model_path'], config['model_name'], config['dataset'], start_wall_time)
    pathlib.Path(checkpoint_path_local).mkdir(parents=True, exist_ok=True)

    if config['model_name'] in ['GraphSAGE', 'LA-SAGE']:
        best_val_model = f"best_val_model_{args.seed}.pth"
    else:
        best_val_model = f"best_val_model_{args.seed}.pth"
    best_model_path       = osp.join(checkpoint_path_local, best_val_model)
    config['best_model_path'] = best_model_path
    data_loaders = (datasetHelper.train_loader, datasetHelper.val_loader, datasetHelper.test_loader)
    
    run_best_model(args, config, data_loaders, logger)
