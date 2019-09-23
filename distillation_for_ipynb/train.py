import numpy as np
from sklearn import metrics
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as NN
import torch.optim as optim
import os
import csv
from utils import XKs, Measure, get_result

def train(dataloader, model, optimizer, epoch, Loss, device=None, time = '', result_dir=None, model_dir=None, csvname = None, modelname=None):
    model.train()
    train_mean_error = 0
    train_pred2save_logit = []
    train_pred2save_score = []
    train_real_train = []

    for i, (data, label) in enumerate(dataloader):
        label = label.to(device).long()

        train_one_hot = torch.zeros([200, 2])
        train_one_hot = train_one_hot.to(device)
        train_one_hot = train_one_hot.scatter_(1, label.view(-1, 1), 1)
        data = data.reshape(200, 1, -1).to(device)

        y = model(data)
        train_real_train.append(label.cpu().numpy())
        pred = y.cpu().detach().numpy()
        pred_logit = pred.argmax(axis=1)
        indx = 1 - pred_logit
        pred_score = np.abs(np.max(pred, axis=1) - indx)
        train_pred2save_logit.append(pred_logit)
        train_pred2save_score.append(pred_score)

        loss = Loss(y.squeeze(), train_one_hot)
        train_mean_error += loss.cpu().data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\r train Epoch: {} step: {} Loss: {}'.format(epoch, i, loss.detach().cpu().numpy()), end="")

    train_real_train = np.array(train_real_train).reshape(-1)
    train_pred2save_score = np.array(train_pred2save_score).reshape(-1)
    train_pred2save_logit = np.array(train_pred2save_logit).reshape(-1)
    train_precision, train_f1, train_recall = Measure(train_real_train, train_pred2save_logit)
    train_acc = metrics.accuracy_score(train_real_train, train_pred2save_logit)
    train_auc = metrics.roc_auc_score(train_real_train, train_pred2save_logit)
    train_ks = XKs(train_real_train, train_pred2save_score)
    train_mean_error = train_mean_error / (i + 1)

    result_train = {
        'epoch': epoch,
        'train_mean_error': train_mean_error.item(),
        'train_acc': train_acc,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_ks': train_ks
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict()
                }, model_dir + 'newest_model.pth')
    if epoch%50 == 0 or epoch==1:
        torch.save(model.state_dict(), model_dir + modelname.format(epoch))

    # result_train = get_result(y=train_real_train, pred=train_pred2save_logit, score=train_pred2save_score, label='(train)')
    with open(result_dir + csvname, 'a', newline='') as f:
        fieldnames = result_train.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if epoch == 0:
            writer.writeheader()
        writer.writerow(result_train)
        f.close()
