import sys
import os
import os.path

import numpy as np
import pandas as pd
import os
import re
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from sklearn.metrics import f1_score
from scipy import stats

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def main():

    with open(f'{parentpath1(__file__, f=0)}/data_name_s2.txt', "r") as f:
        lines = f.readlines()
    
    mode = "ssa_theta1"

    data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "chfdb_chf13_45590_1", "chfdb_chf13_45590_2", "chfdbchf15_1", "chfdbchf15_2", "mitdb__100_180_1", "mitdb__100_180_2"]
    data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
    
    #thresholds_best = stats.chi2.ppf(0.99, 1)

    lag_list = [0.7]
    M_list = [30]

    auc_max = []
    auc_ave = []
    condition_list = []

    for data_i in data_list:
        data_name = data_i
        roc_name = ["fpr", "tpr", "thresholds"]
        if data_name == 'chfdb_chf01_275_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_1_label.csv', names=["A"]).values
            #print(1)
        elif data_name == 'chfdb_chf01_275_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_2_label.csv', names=["A"]).values
            #print(2)
        elif data_name == 'chfdb_chf13_45590_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf13_45590_1_label.csv', names=["A"]).values
            #print(3)
        elif data_name == 'chfdb_chf13_45590_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf13_45590_2_label.csv', names=["A"]).values
            #print(4)
        elif data_name == 'chfdbchf15_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdbchf15_1_label.csv', names=["A"]).values
            #print(5)
        elif data_name == 'chfdbchf15_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdbchf15_2_label.csv', names=["A"]).values
            #print(6)
        elif data_name == 'mitdb__100_180_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_1_label.csv', names=["A"]).values
            #print(7)
        elif data_name == 'mitdb__100_180_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_2_label.csv', names=["A"]).values
            #print(8)
        elif data_name == 'nprs44':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/nprs44_label.csv', names=["A"]).values
            #print(8)
        elif data_name == 'stdb_308_0_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_1_label.csv', names=["A"]).values
            #print(7)
        elif data_name == 'stdb_308_0_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_2_label.csv', names=["A"]).values
        for M_i in M_list:                    
            for lag_i in lag_list:
                M = M_i
                N = M_i
                count = 0
                sum1 = 0
                sum2 = 0
                sum3 = 0
                sum4 = 0
                sum5 = 0
                sum6 = 0
                sum7 = 0
                sum8 = 0
                max1 = []
                max2 = []
                max3 = []
                max4 = []
                max5 = []
                max6 = []
                max7 = []
                max8 = []
                trajectory_list = []
                auc_all = []
                in_dir = f'{parentpath1(__file__, f=0)}/dissim_{mode}_{data_name}_{M}_{lag_i}'
                condition_list.append(f'{mode}_{data_name}_{M}_{lag_i}')
                #print(in_dir)
                #各条件での評価
                for image_name in os.listdir(in_dir):
                    trajectory_list.append(image_name)
                    pred = pd.read_csv(in_dir+"/"+image_name, names=["A"])
                    pred = pred['A'].fillna(pred['A'].mean())
                    pred = pred.values
                    '''
                    train_dir = in_dir + "_train"
                    image_name_train = image_name.replace('.csv', '')
                    image_name_train = image_name_train + "_train.csv"
                    pred_train = pd.read_csv(train_dir+"/"+image_name_train, names=["A"])
                    pred_train = pred_train['A'].fillna(pred_train['A'].mean())
                    pred_train = pred_train.values

                    xmean = pred_train.mean(axis=0, keepdims=True)
                    xstd  = np.std(pred_train, axis=0, keepdims=True)
                    #print(pred.shape, pred_train.shape, xmean, xstd)
                    pred = (pred-xmean)/xstd
                    #print(pred.shape)
                    for i in range(pred.shape[0]):
                        if pred[i] < 0:
                            pred[i] = 0
                    pred = np.square(pred)
                    
                    pred_pre = pred.reshape(-1, 1)
                    pred_bin1 = np.zeros(pred.shape[0])
                    #for i in range(pred.shape[0]):
                        #if pred[i] > thresholds_best:
                            #pred_bin1[i] = 1
                    biner = Binarizer(threshold = thresholds_best).fit(pred_pre)
                    pred_bin = biner.transform(pred_pre)
                    pred_bin = pred_bin[:, 0]
                    '''
                    st = image_name.split('_')
                    if mode == "direc_standard":
                        w = int(re.sub(r"\D", "", st[st.index("standard")+1]))
                        o = int(re.sub(r"\D", "", st[st.index("standard")+2]))
                        l = int(re.sub(r"\D", "", st[st.index("standard")+3]))
                    elif mode == "direc_allover":
                        w = int(re.sub(r"\D", "", st[st.index("allover")+1]))
                        o = int(re.sub(r"\D", "", st[st.index("allover")+2]))
                        l = int(re.sub(r"\D", "", st[st.index("allover")+3]))
                    elif mode == "ssa_theta1":
                        w = int(re.sub(r"\D", "", st[st.index("theta1")+1]))
                        o = int(re.sub(r"\D", "", st[st.index("theta1")+2]))
                        l = int((w+o-1)*(1-0.1*int(re.sub(r"\D", "", st[st.index("theta1")+3])))) 
                        
                    dif = true.shape[0] - pred.shape[0]
                    dif_half = int(dif/2)
                    true1 = true[0:true.shape[0]-dif]
                    true2 = true[w:w+pred.shape[0]]
                    true3 = true[o:o+pred.shape[0]]
                    true4 = true[l:l+pred.shape[0]]
                    true5 = true[w+o:w+o+pred.shape[0]]
                    true6 = true[w+l:w+l+pred.shape[0]]
                    true7 = true[o+l:o+l+pred.shape[0]]
                    true8 = true[w+o+l:w+o+l+pred.shape[0]]
                    
                    fpr1, tpr1, thresholds1 = metrics.roc_curve(true1, pred)
                    auc1 = metrics.auc(fpr1, tpr1)
                    fpr2, tpr2, thresholds2 = metrics.roc_curve(true2, pred)
                    auc2 = metrics.auc(fpr2, tpr2)
                    fpr3, tpr3, thresholds3 = metrics.roc_curve(true3, pred)
                    auc3 = metrics.auc(fpr3, tpr3)
                    fpr4, tpr4, thresholds4 = metrics.roc_curve(true4, pred)
                    auc4 = metrics.auc(fpr4, tpr4)
                    fpr5, tpr5, thresholds5 = metrics.roc_curve(true5, pred)
                    auc5 = metrics.auc(fpr5, tpr5)
                    fpr6, tpr6, thresholds6 = metrics.roc_curve(true6, pred)
                    auc6 = metrics.auc(fpr6, tpr6)
                    fpr7, tpr7, thresholds7 = metrics.roc_curve(true7, pred)
                    auc7 = metrics.auc(fpr7, tpr7)
                    fpr8, tpr8, thresholds8 = metrics.roc_curve(true8, pred)
                    auc8 = metrics.auc(fpr8, tpr8)
                    
                    '''
                    auc1 = f1_score(true1, pred_bin)
                    auc2 = f1_score(true2, pred_bin)
                    auc3 = f1_score(true3, pred_bin)
                    auc4 = f1_score(true4, pred_bin)
                    auc5 = f1_score(true5, pred_bin)
                    auc6 = f1_score(true6, pred_bin)
                    auc7 = f1_score(true7, pred_bin)
                    auc8 = f1_score(true8, pred_bin)
                    '''
                    '''
                    new_dir_path = mode+"_"+ex1+"_"+setting+"_roc_info_ver3"
                    os.makedirs(new_dir_path, exist_ok=True)
                    roc_info = np.array([fpr2, tpr2, thresholds2])
                    roc_info = roc_info.T
                    #print(roc_info.shape)
                    df_roc = pd.DataFrame(roc_info, columns = ["fpr", "tpr", "thresholds"])
                    df_roc.to_csv(f'{new_dir_path}/{mode}_{ex1}_w{w}_o{o}_l{l}_d{d}_{setting}.csv')
                    '''

                    auc_one = [auc1, auc2, auc3, auc4, auc5, auc6, auc7, auc8]
                    auc_all.append(auc_one)
                    sum1 += auc1
                    sum2 += auc2
                    sum3 += auc3
                    sum4 += auc4
                    sum5 += auc5
                    sum6 += auc6
                    sum7 += auc7
                    sum8 += auc8
                    max1.append(auc1)
                    max2.append(auc2)
                    max3.append(auc3)
                    max4.append(auc4)
                    max5.append(auc5)
                    max6.append(auc6)
                    max7.append(auc7)
                    max8.append(auc8)
                    count += 1
                new_dir_path1 = f'fscore_{mode}_eachdata/fscore_{mode}_{data_name}'
                os.makedirs(new_dir_path1, exist_ok=True)
                df = pd.DataFrame(auc_all, index = trajectory_list, columns = ['0', 'W', 'O', 'L', 'W+O', 'W+L', 'O+L', 'W+O+L'])
                #df.to_csv(f'{new_dir_path1}/fscore_{mode}_{data_name}_{M}_{lag_i}.csv')
                #print(count)
                #ある条件での平均のauc(8通りの位置合わせ)
                ave_list = [round(sum1/count, 3), round(sum2/count, 3), round(sum3/count, 3), round(sum4/count, 3), 
                            round(sum5/count, 3), round(sum6/count, 3), round(sum7/count, 3), round(sum8/count, 3)]
                #ある条件での最高のauc(8通りの位置合わせ)
                max1 = np.array(max1)
                max2 = np.array(max2)
                max3 = np.array(max3)
                max4 = np.array(max4)
                max5 = np.array(max5)
                max6 = np.array(max6)
                max7 = np.array(max7)
                max8 = np.array(max8)
                max_list = [np.max(max1, axis=0), np.max(max2, axis=0), np.max(max3, axis=0), np.max(max4, axis=0), np.max(max5, axis=0), np.max(max6, axis=0), np.max(max7, axis=0), np.max(max8, axis=0)]
                auc_ave.append(ave_list)
                auc_max.append(max_list)

                pred = pd.read_csv(in_dir+"/"+trajectory_list[np.argmax(max1)], names=["A"])
                pred = pred['A'].fillna(pred['A'].mean())
                pred = pred.values
                print(np.max(max1))  

                st = trajectory_list[np.argmax(max1)].split('_')
                if mode == "direc_standard":
                    w = int(re.sub(r"\D", "", st[st.index("standard")+1]))
                    o = int(re.sub(r"\D", "", st[st.index("standard")+2]))
                    l = int(re.sub(r"\D", "", st[st.index("standard")+3]))
                elif mode == "direc_allover":
                    w = int(re.sub(r"\D", "", st[st.index("allover")+1]))
                    o = int(re.sub(r"\D", "", st[st.index("allover")+2]))
                    l = int(re.sub(r"\D", "", st[st.index("allover")+3]))
                elif mode == "ssa_theta1":
                    w = int(re.sub(r"\D", "", st[st.index("theta1")+1]))
                    o = int(re.sub(r"\D", "", st[st.index("theta1")+2]))
                    l = int((w+o-1)*(1-0.1*int(re.sub(r"\D", "", st[st.index("theta1")+3])))) 
                    
                dif = true.shape[0] - pred.shape[0]
                dif_half = int(dif/2)
                true1 = true[0:true.shape[0]-dif]
                true2 = true[w:w+pred.shape[0]]
                true3 = true[o:o+pred.shape[0]]
                true4 = true[l:l+pred.shape[0]]
                true5 = true[w+o:w+o+pred.shape[0]]
                true6 = true[w+l:w+l+pred.shape[0]]
                true7 = true[o+l:o+l+pred.shape[0]]
                true8 = true[w+o+l:w+o+l+pred.shape[0]]
                
                fpr1, tpr1, thresholds1 = metrics.roc_curve(true1, pred)
                auc1 = metrics.auc(fpr1, tpr1)
                youden = tpr1 - fpr1
                #print(pred.shape)
                #print(fpr[np.argmax(youden)], tpr[np.argmax(youden)], thresholds_pre[np.argmax(youden)])
                thresholds_best = thresholds1[np.argmax(youden)]
                pred_bin = np.zeros(pred.shape[0])
                for i in range(pred.shape[0]):
                    if pred[i] > thresholds_best:
                        pred_bin[i] = 1
                    else:
                        pred_bin[i] = 0

                auc1 = f1_score(true1, pred_bin)
                #print(max_list)
                print(auc1)
                #print("")
                #print(np.max(max1, axis=0), np.max(max2, axis=0), np.max(max3, axis=0), np.max(max4, axis=0), np.max(max5, axis=0), np.max(max6, axis=0), np.max(max7, axis=0), np.max(max8, axis=0))
                #print(trajectory_list[np.argmax(max2)])
                #print("")
    df_ave = pd.DataFrame(auc_ave, index = condition_list, columns = ['0', 'W', 'O', 'L', 'W+O', 'W+L', 'O+L', 'W+O+L'])
    df_all = pd.DataFrame(auc_max, index = condition_list, columns = ['0', 'W', 'O', 'L', 'W+O', 'W+L', 'O+L', 'W+O+L'])
    #df_ave.to_csv('fscore_'+mode+'_ave.csv')
    #df_all.to_csv('fscore_'+mode+'_max.csv')
    #print("")

if __name__=='__main__':
    main()