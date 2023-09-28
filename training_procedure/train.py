
import torch.nn as nn
from DataHelper.datasetHelper import DatasetHelper
import torch
def train(self, epoch, model, loss_func, optimizer, train_loader = None, datasetHelper: DatasetHelper = None):
    model.train()
    config = self.config 
    total_loss = 0.0

    if config['model_name'] == 'LA-SAGE-S':
        relations = datasetHelper.relations
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            # train on blocks
            blocks = [b.to(torch.cuda.current_device()) for b in blocks]
            train_feats = blocks[0].srcdata['feature']
            train_label = blocks[-1].dstdata['label']

            optimizer.zero_grad()
            batch_logits = model(blocks, relations, train_feats)
            loss = loss_func(batch_logits, train_label)
            total_loss += loss 
            loss.backward()
            optimizer.step()
    
    return model, total_loss