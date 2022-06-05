import pylab
import os
from statsmodels.tsa.ar_model import AR
import numpy as np
from pandas import Series
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

# Reference
#   data : https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/texmex/nidd.csv  # noqa

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def train_autoregression(train_data, test_data):
    # train autoregression
    #print(train_data)
    model = AR(train_data)
    model_fit = model.fit(ic = 'aic')
    window = model_fit.k_ar
    coef = model_fit.params
    #print('Lag: %s' % model_fit.k_ar)
    #print('Coefficients: %s' % model_fit.params)
    # make predictions
    #print('aic {}'.format(model_fit.aic))
    history = train_data[len(train_data) - window:]
    history = [h for h in history]
    predictions = []

    for index, test in enumerate(test_data):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        #print("lag", len(lag), lag[0])
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window - d - 1]
        obs = test
        predictions.append(yhat)
        history.append(obs)

    return predictions


def main():
    data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
    auc_list = []
    fscore_list = []
    for data_name in data_list:
        if data_name == 'chfdb_chf01_275_1':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 1237
            end = 1437
            #print(1)
        elif data_name == 'chfdb_chf01_275_2':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 1341
            end = 1536
            #print(2)
        elif data_name == 'mitdb__100_180_1':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
            start = 799
            end = 987
            #print(7)
        elif data_name == 'mitdb__100_180_2':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
            start = 799
            end = 987
            #print(8)
        elif data_name == 'nprs44':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[12700:15500]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[15500:22000]
            start = 4900
            end = 5380
            #print(8)
        elif data_name == 'stdb_308_0_1':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 1]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 1]
            start = 772
            end = 1065
            #print(7)
        elif data_name == 'stdb_308_0_2':
            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 2]
            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 2] 
            start = 763
            end = 1053      
        
        pred = train_autoregression(train, test)
        
        plt.plot(pred)
        #plt.show()
        plt.close()

        dist = test - pred
        pred = pow(dist, 2)
        
        plt.plot(pred)
        #plt.show()
        plt.close()

        plt.plot(test)
        #plt.show()
        plt.close()

        '''
        pylab.figure(figsize=(14, 8))
        pylab.subplot(211)
        pylab.xlabel('time')
        pylab.ylabel('temperature')
        #pylab.plot(test, label='real')
        pylab.plot(dist, '--', label='predict')
        pylab.legend(loc='upper right')
        pylab.show()
        '''
        if data_name == 'chfdb_chf01_275_1':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_1_label.csv', names=["A"]).values
            #print(1)
        elif data_name == 'chfdb_chf01_275_2':
            true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_2_label.csv', names=["A"]).values
            #print(2)
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
            #print(8)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(true, pred)
        auc1 = metrics.auc(fpr1, tpr1)
        auc_list.append(auc1)
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

        fscore = f1_score(true, pred_bin)
        fscore_list.append(fscore)
        #print(max_list)
        #print(auc1)
        #print(fscore)
        #print(f'data:{data_name}, AUC:{auc1}, f_score:{fscore}')
        plt.axvspan(start, end, color="lightcoral")
        if data_name == 'nprs44':
            plt.axvspan(2087, 2553, color="lightcoral")
        plt.plot(pred)
        #plt.show()
        new_dir_path = 'image_arma'
        os.makedirs(new_dir_path, exist_ok=True)
        plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/arma_{data_name}.png')
        plt.close()
    print("")
    print("result:")
    for i in range(len(auc_list)):
        print(f'data:{data_list[i]}, AUC:{auc_list[i]}, f_score:{fscore_list[i]}')
    

if __name__ == '__main__':
    main()