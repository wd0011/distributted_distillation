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

def test(dataloader, model, epoch, Loss, device=None, time = '', result_dir=None, model_dir=None, csvname = None, modelname=None):
    model.eval()
    test_mean_error = 0
    test_pred2save_logit = []
    test_pred2save_score = []
    test_real_test = []

    for i, (data, label) in enumerate(dataloader):
        label = label.to(device).long()

        test_one_hot = torch.zeros([200, 2])
        test_one_hot = test_one_hot.to(device)
        test_one_hot = test_one_hot.scatter_(1, label.view(-1, 1), 1)
        data = data.reshape(200, 1, -1).to(device)

        y = model(data)
        test_real_test.append(label.cpu().numpy())
        pred = y.cpu().detach().numpy()
        pred_logit = pred.argmax(axis=1)
        indx = 1 - pred_logit
        pred_score = np.abs(np.max(pred, axis=1) - indx)
        test_pred2save_logit.append(pred_logit)
        test_pred2save_score.append(pred_score)

        loss = Loss(y.squeeze(), test_one_hot)
        test_mean_error += loss.cpu().data

        print('\r Test Epoch: {} step: {} Loss: {}'.format(epoch, i, loss.detach().cpu().numpy()), end="")

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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict()
                }, model_dir + 'newest_model_aftertrain.pth')
    if epoch%50 == 0 or epoch==1:
        torch.save(model.state_dict(), model_dir + 'aftertrain_' + modelname.format(epoch))

    # result_test = get_result(y=test_real_test, pred=test_pred2save_logit, score=test_pred2save_score, label='(test)')
    csvname = 'aftertrain_'+csvname
    with open(result_dir + csvname, 'a', newline='') as f:
        fieldnames = result_test.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if epoch == 0:
            writer.writeheader()
        writer.writerow(result_test)
        f.close()
