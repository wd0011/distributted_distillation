import pandas as pd
import numpy as np
from collections import Counter
import time 
import os
import sys
import glob
import csv
from scipy.stats import ks_2samp
from sklearn import metrics


def XKs(test_y_data, yscore):
    fpr, tpr, thresholds = metrics.roc_curve(test_y_data, yscore)
    ks = np.max(np.abs(tpr-fpr))
    return ks


def Measure(test_y_data, ypred):
    TP = 0    # 预测坏     实际坏
    FP = 0    # 预测坏     实际好 
    FN = 0    # 预测好     实际坏
    TN = 0    # 预测好     实际好

    for i in range(len(ypred)):
        if ypred[i] == 1 and test_y_data[i] == 1:
            TP += 1
        elif ypred[i] == 1 and test_y_data[i] == 0:
            FP += 1
        elif ypred[i] == 0 and test_y_data[i] == 1:
            FN += 1
        elif ypred[i] == 0 and test_y_data[i] == 0:
            TN +=1    
            
    accuracy = float(TP+TN)/float(TP+TN+FP+FN)
    precision = float(TP)/float(TP+FP+1)
    recall = TP/float(TP+FN)
    f1 = 2*TP/float(2*TP+FP+FN)
    
    return precision, recall, f1


def get_ks_score(y_true, y_pred):
    '''
    algorithm for getting ks score
    '''
    d = zip(y_pred, y_true)
    bad = [k[0] for k in d if k[1]==1]
    d = zip(y_pred, y_true)
    good = [k[0] for k in d if k[1]==0]
    if len(bad)+len(good)<100:
        return 0
    return ks_2samp(bad, good)[0]

def get_result(y, pred=[], score=[], threshold=0.5, positive=1, negative=0, label = '(test)'):
    '''
    This is the function for getting the binary classification result (including acc, auc, precision, recall, f1, ks, and counts.).
    Input: 
        1. y is the ground truth of prediction.
        2. pred is the classification prediction.
        3. score is the prediction probability.
        3. threshold to devide probability data to binary.
        4. positive and negative: in case the ground truth data is not 1 and 0.
        5. label for id multipy result.
    '''
    
    assert (any(pred)) or (any(score))
    
    if not(any(pred)):
        pred = [positive if score_value >= threshold else negative for score_value in score]
    
    accuracy = metrics.accuracy_score(y, pred)
    precision = metrics.precision_score(y, pred)
    recall = metrics.recall_score(y, pred)
    f1 = metrics.f1_score(y, pred)    
    if any(score):
        auc = metrics.roc_auc_score(y, score)
        ks = get_ks_score(y, score)       
        result = {'Accuracy'+label:                  accuracy,
                  'AUC'+label:                       auc,
                  'Precision'+label:                 precision,
                  'Recall'+label:                    recall,
                  'F1'+label:                        f1,
                  'KS'+label:                        ks,
                  'Ground_Truth_Counts'+label:       Counter(y),
                  'Counts'+label:                    Counter(pred)
        }
    else:
        result = {'Accuracy'+label:                  accuracy,
                  'Precision'+label:                 precision,
                  'Recall'+label:                    recall,
                  'F1'+label:                        f1,
                  'Ground_Truth_Counts'+label:       Counter(y),
                  'Counts'+label:                    Counter(pred)
        }
    return result

def adjust_learning_rate(opt, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch//500))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

