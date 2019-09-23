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



def distillation(NN_set, opt_set, dataset, world_size, epoch, device=None):
    '''

    :param NN_set: model set from workers
    :param opt_set: optimizer set for models
    :param dataset: dataset
    :param world_size: total number of nodes
    :return:
    '''

    for worker_id in range(world_size - 1):
        NN_set[worker_id].train()  # 记得到时候。eval（）需要的

    for i, (data, label) in enumerate(dataset):
        data = data.reshape(200, 1, -1).to(device)
        for worker_id in range(world_size - 1):
            opt_set[worker_id].zero_grad()

        p = []
        p_detach = []
        for worker_id in range(world_size - 1):
            p.append(NN_set[worker_id](data))
            p_detach.append(p[worker_id].detach())
        loss = []

        for worker_id in range(world_size - 1):
            loss_temp = F.binary_cross_entropy(p[worker_id], p_detach[worker_id])
            for worker_id_ in range(world_size - 1):
                loss_temp += torch.mean(
                    torch.sum(F.binary_cross_entropy(p[worker_id], p_detach[worker_id_], reduce=False), dim=1)
                )
            loss.append(loss_temp)
            loss[worker_id].backward(retain_graph=True)
            opt_set[worker_id].step()
            
        for worker_id in range(world_size - 1):
            print('\r Epoch: {} Step: {} Distillate Loss_{}: {:.6f}'.format(epoch, i, worker_id+1, loss[worker_id].item()), end="")


def choose_best(NN_set, name, dataset, world_size, epoch, Loss, device=None, time=''): #很久没有更新了，如果需要使用需要先更新在使用
    result_dir = 'result_' + time + '/' + name + '/'
    model_dir = 'models_' + time + '/' + name + '/'
    csvname = '{}_log'.format(name) + time + '.csv'
    modelname = 'model_{:d}.pth'


    result_set = []
    for worker_id in range(world_size - 1):

        NN_set[worker_id].eval()
        test_mean_error = 0
        test_pred2save_logit = []
        test_pred2save_score = []
        test_real_test = []

        for i, (data, label) in enumerate(dataset):
            label = label.to(device).long()

            test_one_hot = torch.zeros([200, 2])
            test_one_hot = test_one_hot.to(device)
            test_one_hot = test_one_hot.scatter_(1, label.view(-1, 1), 1)
            data = data.reshape(200, 1, -1).to(device)

            y = NN_set[worker_id](data)
            test_real_test.append(label.cpu().numpy())
            pred = y.cpu().detach().numpy()
            pred_logit = pred.argmax(axis=1)
            indx = 1 - pred_logit
            pred_score = np.abs(np.max(pred, axis=1) - indx)
            test_pred2save_logit.append(pred_logit)
            test_pred2save_score.append(pred_score)

            loss = Loss(y.squeeze(), test_one_hot)
            test_mean_error += loss.cpu().data

            print('\r Test Epoch: {} workerID: {} step: {} Loss: {}'.format(epoch, worker_id, i, loss.detach().cpu().numpy()), end="")

        test_real_test = np.array(test_real_test).reshape(-1)
        test_pred2save_score = np.array(test_pred2save_score).reshape(-1)
        test_pred2save_logit = np.array(test_pred2save_logit).reshape(-1)
        test_precision, test_f1, test_recall = Measure(test_real_test, test_pred2save_logit)
        test_acc = metrics.accuracy_score(test_real_test, test_pred2save_logit)
        test_auc = metrics.roc_auc_score(test_real_test, test_pred2save_logit)
        test_ks = XKs(test_real_test, test_pred2save_score)
        test_mean_error = test_mean_error / (i + 1)

        result_test = {
            'epoch': epoch,
            'test_mean_error': test_mean_error.item(),
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_ks': test_ks
        }

        result_set.append(result_test)

    error_list = []
    for i in range(world_size-1):
        error_list.append(result_set[i]['test_mean_error'])
    best_idx = error_list.index(min(error_list))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torch.save(model_set[0].state_dict(), model_dir + 'newest_model.pth')
    if (epoch-1)%50 == 0:
        torch.save(NN_set[best_idx].state_dict(), model_dir + modelname.format(epoch))

    # result_test = get_result(y=test_real_test, pred=test_pred2save_logit, score=test_pred2save_score, label='(test)')


    with open(result_dir + csvname, 'a', newline='') as f:
        fieldnames = result_set[best_idx].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if epoch == 1:
            writer.writeheader()
        writer.writerow(result_set[best_idx])
        f.close()

    return best_idx



