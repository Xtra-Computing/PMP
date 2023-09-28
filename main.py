import os.path as osp
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import torch
import random
from utils.logger import Logger
from utils.utils import *
from utils.random_seeder import set_random_seed
from training_procedure import Trainer
from DataHelper.datasetHelper import DatasetHelper
from torch.utils.data import DataLoader
import pathlib
import utils.plot_tools as plot_tools
import warnings
import datetime

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

def main(args, config, logger: Logger, run_id: int, datasetHelper: DatasetHelper, loaders, start_wall_time):
    T            = Trainer(config=config, args= args, logger= logger)
    train_loader, val_loader, test_loader  = loaders
    model, optimizer, loss_func, scheduler = T.init(datasetHelper)   # model of current split


    pbar = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    if config['test_each_epoch']:
        tst_track = tqdm(bar_format='{desc}', ncols = 0)
    patience_cnt 		= 0
    best_metric 	  	= -10000
    best_gmean          = -10000
    best_macro          = -10000
    best_metric_epoch 	= -1 # best number on dev set
    report_dev_res 		= None
    best_model_path = config['best_model_path']
    for epoch in pbar:
        model, loss          = T.train(epoch, model, loss_func, optimizer, train_loader, datasetHelper)
        avg_train_loss       = loss / len(train_loader)
        if config.get('lr_scheduler', False):
            scheduler.step(avg_train_loss)
        if epoch % config['eval_interval'] == 0:
            labels, fraud_probs, preds = T.evaluation(datasetHelper, 
                                                      val_loader, 
                                                      model, 
                                                      threshold_moving=config['threshold_moving'], 
                                                      thres = config['thres'])  # return 2 list, 
            dev_results = T.eval_model(labels, fraud_probs, preds)
            if config['test_each_epoch']:
                tst_labels, tst_fraud_prob, tst_preds = T.evaluation(datasetHelper, 
                                                                     test_loader, 
                                                                     model, 
                                                                     threshold_moving=config['threshold_moving'], 
                                                                     thres = dev_results.best_pr_thres)
                tst_results = T.eval_model(tst_labels, tst_fraud_prob, tst_preds)
            if config['patience'] > 0: # use early stop
                now_metric = getattr(dev_results, config['monitor'])
                now_gmean  = getattr(dev_results, 'gmean_gnn')
                now_macro  = getattr(dev_results, 'f1_macro')
                if best_metric       <= now_metric: 
                    best_metric       = now_metric
                    best_metric_epoch = epoch
                    report_dev_res    = dev_results
                    report_test_res   = tst_results if config['test_each_epoch'] else None
                    patience_cnt      = 0
                    save_checkpoint(model, best_model_path)
               

                else:
                    patience_cnt     +=1
            if config['patience'] > 0 and patience_cnt >= config['patience']:
                break
        # postfix_str = "<E %d> [Train Loss] %.4f [Curr Dev AUC] %.4f <Best Dev Res:> [Epoch] %d [AUC] %.4f [GMean] %.4f [F1Ma] %.4f [AP] %.4f [RecMa] %.4f \n"\
        #                "<Tst Res:> [AUC] %.4f [GMean] %.4f [F1Ma] %.4f [AP] %.4f [RecMa] %.4f " % ( 
        #                 epoch ,   loss,     dev_results.auc_gnn, best_metric_epoch ,report_dev_res.auc_gnn, report_dev_res.gmean_gnn, report_dev_res.f1_macro, report_dev_res.ap_gnn, report_dev_res.recall_macro,
        #                 report_test_res.auc_gnn, report_test_res.gmean_gnn, report_test_res.f1_macro, report_test_res.ap_gnn, report_test_res.recall_macro
        #                 )
        postfix_str = "<Epoch %d> [Train Loss] %.4f [Dev AUC] %.4f <Best Dev Res:> [Epoch] %d [AUC] %.4f [GMean] %.4f [F1Ma] %.4f [AP] %.4f [RecMa] %.4f " % ( 
                epoch ,   float(avg_train_loss),     dev_results.auc_gnn, best_metric_epoch ,report_dev_res.auc_gnn, report_dev_res.gmean_gnn, report_dev_res.f1_macro, report_dev_res.ap_gnn, report_dev_res.recall_macro)
                

        pbar.set_postfix_str(postfix_str)
        if config['test_each_epoch']:
            postfix_str_test =  "<Test Results>: [AUC] %.4f [GMean] %.4f [F1Ma] %.4f [AP] %.4f [RecMa] %.4f " % (report_test_res.auc_gnn, report_test_res.gmean_gnn, report_test_res.f1_macro, report_test_res.ap_gnn, report_test_res.recall_macro)
            tst_track.set_description_str(postfix_str_test)
            # tst_track.set_postfix_str()
            # print(postfix_str_test, end='\r')
    logger.log("best epoch is %d" % best_metric_epoch)
    logger.log("Best Epoch Valid AUC is %.4f" % (report_dev_res.auc_gnn))
    if config['test_each_epoch']:
        logger.log("Best Epoch Test AUC is %.4f" % (report_test_res.auc_gnn))

    if config['patience'] > 0:
        best_model = load_checkpoint(model, best_model_path)
        labels, fraud_probs, preds = T.evaluation(datasetHelper, 
                                                  val_loader, 
                                                  best_model, 
                                                  threshold_moving=config['threshold_moving'], 
                                                  thres = config['thres'])
        best_dev_results = T.eval_model(labels, fraud_probs, preds)
        # report results on best threshold on Precision Recall Curve
        print(f"best_roc_thres: {best_dev_results.best_roc_thres} \n"
              f"best_pr_thres: {best_dev_results.best_pr_thres}")
        tst_labels, tst_fraud_prob, tst_preds = T.evaluation(datasetHelper, 
                                                             test_loader, 
                                                             best_model, 
                                                             threshold_moving=config['threshold_moving'], 
                                                             thres = best_dev_results.best_pr_thres)
        final_test_results = T.eval_model(tst_labels, tst_fraud_prob, tst_preds)

    return best_model,  best_dev_results, final_test_results, loss


def run_best_model(args, config, loaders, logger: Logger):
    T           = Trainer(config=config, args= args, logger= logger)
    model, _, _ = T.init(datasetHelper)
    best_model_path = config['best_model_path']
    print(best_model_path)
    print(osp.exists(best_model_path))
    best_model  = load_checkpoint(model, best_model_path)
    val_loader  = loaders[1]
    test_loader = loaders[2]
    labels, fraud_probs, preds = T.evaluation(datasetHelper, 
                                                val_loader, 
                                                best_model, 
                                                threshold_moving=config['threshold_moving'], 
                                                thres = config['thres'])
    best_dev_results = T.eval_model(labels, fraud_probs, preds)
    tst_labels, tst_fraud_prob, tst_preds = T.evaluation(datasetHelper, 
                                                        test_loader, 
                                                        best_model, 
                                                        threshold_moving=config['threshold_moving'], 
                                                        thres = best_dev_results.best_pr_thres)
    final_test_results = T.eval_model(tst_labels, tst_fraud_prob, tst_preds)
    logger.append = ""
    for metric in METRIC_NAME:
        # metric_list = np.around([getattr(result, metric) for result in final_test_results], decimals=5)
        metric_value = getattr(final_test_results, metric)
        # logger.log("%s : %s" % (metric , str([round(x,4) for x in metric_list])))
        logger.log("%s : = %.4f" % (metric , metric_value))

# need label imbalance enhance


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset',          type = str,             default = 'yelp') 
    parser.add_argument('--num_workers',   default = 8,                  type = int, choices = [0,8])
    parser.add_argument('--seed',          default = 1234,               type = int, choices = [0, 1, 1234])
    parser.add_argument('--data_dir',         type = str,             default = "datasets/") 
    parser.add_argument('--hyper_file',       type = str,             default = 'config/')
    parser.add_argument('--log_dir',          type = str,             default = 'logs/')
    parser.add_argument('--best_model_path',  type = str,             default = 'checkpoints/')
    parser.add_argument('--train_size',       type = float,           default = 0.4)
    parser.add_argument('--val_size',         type = float,           default = 0.2)
    parser.add_argument('--no_dev',         action = "store_true" ,   default = False)
    parser.add_argument('--gpu_id',           type = int,             default = 0)
    parser.add_argument('--multirun',         type = int,             default = 1)
    parser.add_argument('--model',            type = str,             default ='LA-SAGE-S')  # GCN, GAT or LA-SAGE2
    parser.add_argument('--run_best',       action ='store_true',     default = False)
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu_id)
    start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    logger          = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()

    config_path = osp.join(args.hyper_file, args.dataset + '.yml')
    config = get_config(config_path)
    model_name = args.model
    config = config[model_name] 
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
    
    if args.run_best:
        run_best_model(args, config, data_loaders, logger)
        sys.exit()

    for run_id in range(config['multirun']):   # one mask
        logger.add_line()
        logger.log ("\t\t%d th Run" % run_id)
        logger.add_line()


        model, best_dev_results, final_test_results, loss = main(args, config, logger, run_id, datasetHelper, data_loaders, start_wall_time = start_wall_time)

        logger.log("%d th Run ended. Final Train Loss is %s" % (run_id , str(loss)))
        logger.log("%d th Run ended. Best Epoch Valid is %s" % (run_id , str(best_dev_results._asdict())))
        logger.log("%d th Run ended. Best Epoch Test is %s" % (run_id , str(final_test_results._asdict())))
        
        dev_ress.append(best_dev_results)   # multi_run results
        tes_ress.append(final_test_results)  

    logger.add_line()
    print(config['best_model_path'])
    config_resave_file = osp.join(checkpoint_path_local, '{}.yml'.format(args.dataset))
    with open(config_resave_file, 'w+') as f:
        yaml.dump(config, f, sort_keys=True, indent = 2)

    test_result_file = osp.join(checkpoint_path_local, 'results.txt')
    results_file = open(test_result_file,'w')

    for results, name in zip( [dev_ress, tes_ress], ['Dev', 'Test'] ):  
        # results [result, result, ...]
        for metric in METRIC_NAME:
           # results = dev_results list
            metric_list = np.around([getattr(result, metric) for result in results], decimals=5)
            if metric == 'auc_gnn':
                print(metric_list)
            avg         = np.mean(metric_list, axis = 0)
            std         = np.std(metric_list, axis=0, ddof=1)
            logger.log("%s of %s : %s" % (metric , name , str([round(x,4) for x in metric_list])))
            logger.log("%s of %s : avg / std = %.4f / %.4f" % (metric , name , avg , std))
            if name == 'Test':
                results_file.write("%s of %s : avg / std = %.4f / %.4f" % (metric , name , avg , std))
                results_file.write("\n")
        logger.log("")
    results_file.close()