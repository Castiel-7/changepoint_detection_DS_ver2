'''
only direction
Use eigenvalues all over d(ex:1e-05)
detection type series: data renewal(train_length:1000(about 30%), test_length:2500(about 70%))
'''
import sys
import os
import re
import os.path
import numpy as np
import pandas as pd
from utils.detection_ssa_theta1 import SSAtheta1
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def main():
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #1. detect change point
    for i in range(1):
        for i in range(1):
            data_list_dummy = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "chfdb_chf13_45590_1", "chfdb_chf13_45590_2", "chfdbchf15_1", "chfdbchf15_2", "mitdb__100_180_1", "mitdb__100_180_2"]
            data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
            me = 'ssa_theta1'
            window_list = [64, 128, 256]
            order_list = [64, 128, 256]
            lag_list = [0.7]
            M_list = [30]
            for data_i in data_list:
                for window_i in window_list:
                    for order_i in order_list:
                        for lag_i in lag_list:
                            for M_i in M_list:
                                data_name = data_i
                                window_length = window_i
                                order = order_i
                                lag = int((window_length+order-1)*(1-lag_i))
                                M = M_i
                                N = M_i
                                print(f'{data_name}, w{window_length}, o{order}, l{lag}, m{M}, n{N}')
        
                                if data_name == "chfdb_chf01_275_1":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                elif data_name == "chfdb_chf01_275_2":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                elif data_name == "chfdb_chf13_45590_1":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                elif data_name == "chfdb_chf13_45590_2":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                elif data_name == "chfdbchf15_1":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                elif data_name == "chfdbchf15_2":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                elif data_name == "mitdb__100_180_1":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                elif data_name == "mitdb__100_180_2":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                elif data_name == "nprs44":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[12700:15500]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[15500:22000]
                                elif data_name == "stdb_308_0_1":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 1]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 1]
                                elif data_name == "stdb_308_0_2":
                                    train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 2]
                                    test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 2]

                                model = SSAtheta1(window_length=window_length, order=order, lag = lag, M = M, N = N)
                                score = model.predict(test)
                                #print(train.shape, test.shape, score2.shape)
                                #print(int((score2.shape[1]-1)/2))
                                #print(type(score2))

                                new_dir_path1 = f'dissim_{me}_{data_name}_{M}_{lag_i}'
                                os.makedirs(new_dir_path1, exist_ok=True)
                                np.savetxt(f'{new_dir_path1}/{data_name}_{me}_w{window_length}_o{order}_l{lag_i}_d{M}_s2.csv', score, delimiter=',')
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #2. calculate AUC and F1-score
    mode = "ssa_theta1"
    #data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "chfdb_chf13_45590_1", "chfdb_chf13_45590_2", "chfdbchf15_1", "chfdbchf15_2", "mitdb__100_180_1", "mitdb__100_180_2"]
    data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
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
                #?????????????????????
                for image_name in os.listdir(in_dir):
                    trajectory_list.append(image_name)
                    pred = pd.read_csv(in_dir+"/"+image_name, names=["A"])
                    pred = pred['A'].fillna(pred['A'].mean())
                    pred = pred.values

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
                new_dir_path1 = f'auc_{mode}_eachdata/auc_{mode}_{data_name}'
                os.makedirs(new_dir_path1, exist_ok=True)
                df = pd.DataFrame(auc_all, index = trajectory_list, columns = ['0', 'W', 'O', 'L', 'W+O', 'W+L', 'O+L', 'W+O+L'])
                df.to_csv(f'{new_dir_path1}/auc_{mode}_{data_name}_{M}_{lag_i}.csv')
                #print(count)
                #???????????????????????????auc(8????????????????????????)
                ave_list = [round(sum1/count, 3), round(sum2/count, 3), round(sum3/count, 3), round(sum4/count, 3), 
                            round(sum5/count, 3), round(sum6/count, 3), round(sum7/count, 3), round(sum8/count, 3)]
                #???????????????????????????auc(8????????????????????????)
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
                #print(np.max(max1))  

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

                f_score = f1_score(true1, pred_bin)
                #print(max_list)
                print(f'data:{data_i}, AUC:{np.max(max1, axis=0)}, f_score:{f_score}')
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #3. output graph(change degree)
    mode = "ssa_theta1"
    M = 30
    N = 30
    lag_i = 0.7

    data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
    lag_list = [0.9, 0.7, 0.5, 0.3]
    M_list = [30]
    d_list = [0.00001, 0.001, 0.0001, 0.000001]
    p_list = [30, 50, 70, 90]

    for data_i in data_list:
        data_name = data_i
        roc_name = ["fpr", "tpr", "thresholds"]
        if data_name == 'chfdb_chf01_275_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_1_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 1237
            end = 1437
            #print(1)
        elif data_name == 'chfdb_chf01_275_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_2_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 1341
            end = 1536
            #print(2)
        elif data_name == 'chfdb_chf13_45590_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf13_45590_1_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 1852
            end = 2032
            #print(3)
        elif data_name == 'chfdb_chf13_45590_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf13_45590_2_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 1819
            end = 2014
            #print(4)
        elif data_name == 'chfdbchf15_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdbchf15_1_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 1275
            end = 1468
            #print(5)
        elif data_name == 'chfdbchf15_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdbchf15_2_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 1275
            end = 1468
            #print(6)
        elif data_name == 'mitdb__100_180_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_1_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 799
            end = 987
            #print(7)
        elif data_name == 'mitdb__100_180_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_2_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 799
            end = 987
            #print(8)
        elif data_name == 'nprs44':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/nprs44_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[15500:22000]
            start = 4900
            end = 5380
            #print(8)
        elif data_name == 'stdb_308_0_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_1_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 1]
            start = 772
            end = 1065
            #print(7)
        elif data_name == 'stdb_308_0_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_2_label.csv', names=["A"]).values
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 2]
            start = 763
            end = 1053
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
        #print(in_dir)
        #?????????????????????
        for image_name in os.listdir(in_dir):
            trajectory_list.append(image_name)
            pred = pd.read_csv(in_dir+"/"+image_name, names=["A"])
            pred = pred['A'].fillna(pred['A'].mean())
            pred = pred.values

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
                l = int(re.sub(r"\D", "", st[st.index("theta1")+3]))  
            
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
        #new_dir_path1 = f'auc_{mode}_eachdata/auc_{mode}_{data_name}'
        #os.makedirs(new_dir_path1, exist_ok=True)
        #df = pd.DataFrame(auc_all, index = trajectory_list, columns = ['0', 'W', 'O', 'L', 'W+O', 'W+L', 'O+L', 'W+O+L'])
        #df.to_csv(f'{new_dir_path1}/auc_{mode}_{data_name}_{M}_{DS_dim}_{PS_dim}_{cano_type}_{lag_i}.csv')
        #print(count)
        #???????????????????????????auc(8????????????????????????)
        ave_list = [round(sum1/count, 3), round(sum2/count, 3), round(sum3/count, 3), round(sum4/count, 3), 
                    round(sum5/count, 3), round(sum6/count, 3), round(sum7/count, 3), round(sum8/count, 3)]
        #???????????????????????????auc(8????????????????????????)
        max1 = np.array(max1)
        max2 = np.array(max2)
        max3 = np.array(max3)
        max4 = np.array(max4)
        max5 = np.array(max5)
        max6 = np.array(max6)
        max7 = np.array(max7)
        max8 = np.array(max8)
        max_list = [np.max(max1, axis=0), np.max(max2, axis=0), np.max(max3, axis=0), np.max(max4, axis=0), np.max(max5, axis=0), np.max(max6, axis=0), np.max(max7, axis=0), np.max(max8, axis=0)]
        #auc_ave.append(ave_list)
        #auc_max.append(max_list)
        #print(np.max(max1, axis=0), np.max(max2, axis=0), np.max(max3, axis=0), np.max(max4, axis=0), np.max(max5, axis=0), np.max(max6, axis=0), np.max(max7, axis=0), np.max(max8, axis=0))
        #print(trajectory_list[np.argmax(max2)])
        st = trajectory_list[np.argmax(max2)].split('_')
        if mode == "direc_standard":
            w = int(re.sub(r"\D", "", st[st.index("standard")+1]))
            o = int(re.sub(r"\D", "", st[st.index("standard")+2]))
            l = int((w+o-1)*(1-0.1*int(re.sub(r"\D", "", st[st.index("standard")+3])))) 
        elif mode == "direc_allover":
            w = int(re.sub(r"\D", "", st[st.index("allover")+1]))
            o = int(re.sub(r"\D", "", st[st.index("allover")+2]))
            l = int((w+o-1)*(1-0.1*int(re.sub(r"\D", "", st[st.index("allover")+3]))))
        elif mode == "ssa_theta1":
            w = int(re.sub(r"\D", "", st[st.index("theta1")+1]))
            o = int(re.sub(r"\D", "", st[st.index("theta1")+2]))
            l = int((w+o-1)*(1-0.1*int(re.sub(r"\D", "", st[st.index("theta1")+3])))) 
        new_dir_path = f'image_{mode}_s2'
        os.makedirs(new_dir_path, exist_ok=True)
        changedata = pd.read_csv(f'{parentpath1(__file__, f=0)}/dissim_{mode}_{data_name}_{M}_{lag_i}/{trajectory_list[np.argmax(max1)]}', names=["A"]).values
        #original_data
        changedata_length = test.shape[0] - (w + o + l)
        #print(w, o, l, changedata_length)
        original_data = test[0:0+changedata_length]
        plt.axvspan(start - 0, end - 0, color="lightcoral")
        if data_name == 'nprs44':
            plt.axvspan(2087 - 0, 2553 - 0, color="lightcoral")
        plt.plot(original_data)
        #plt.show()
        plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/{trajectory_list[np.argmax(max2)]}_original.png')
        plt.close()
        #change_data
        plt.axvspan(start - 0, end - 0, color="lightcoral")
        if data_name == 'nprs44':
            plt.axvspan(2087 - 0, 2553 - 0, color="lightcoral")
        plt.plot(changedata)
        plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/{trajectory_list[np.argmax(max2)]}.png')
        plt.close()
        #print(original_data.shape, changedata.shape)
        
if __name__=='__main__':
    main()