import keras
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import os
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import f1_score

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def Mahalanobis_dist(x, mean, cov):
    d = np.dot(x-mean, np.linalg.inv(cov))
    d = np.dot(d, (x-mean).T)
    return d

def generator(data, lookback, delay, pred_length, min_index, max_index, shuffle=False,
              batch_size=100, step=1):
    if max_index is None:
        max_index = len(data) - delay - pred_length - 1 
    i = min_index + lookback 

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, 
                                    size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                               lookback//step,
                               data.shape[-1]))

        targets = np.zeros((len(rows), pred_length))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay : rows[j] + delay + pred_length].flatten()

        yield samples, targets

data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
auc_list1 = []
fscore_list1 = []
for data_name in data_list:
    auc_list0 = []
    fscore_list0 = []
    if data_name == "chfdb_chf01_275_1":
        tr_start = 0
        tr_end = 1000
        te_start = 1000
        te_end = 3500
        start = 1237
        end = 1437
        lookback = 64
        unit_i = 32
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_1_label.csv', names=["A"]).values
    elif data_name == "chfdb_chf01_275_2":
        tr_start = 0
        tr_end = 1000
        te_start = 1000
        te_end = 3500
        start = 1341
        end = 1536
        lookback = 128
        unit_i = 32
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_2_label.csv', names=["A"]).values
    elif data_name == "mitdb__100_180_1":
        tr_start = 0
        tr_end = 1000
        te_start = 1000
        te_end = 3500
        start = 799
        end = 987
        lookback =64
        unit_i = 32
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_1_label.csv', names=["A"]).values
    elif data_name == "mitdb__100_180_2":
        tr_start = 0
        tr_end = 1000
        te_start = 1000
        te_end = 3500
        start = 799
        end = 987
        lookback = 128
        unit_i = 16
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/mitdb__100_180_2_label.csv', names=["A"]).values
    elif data_name == "nprs44":
        tr_start = 12700
        tr_end = 15500
        te_start = 15500
        te_end = 22000
        start = 4900
        end = 5380
        lookback = 128
        unit_i = 8
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/nprs44_label.csv', names=["A"]).values
    elif data_name == "stdb_308_0_1":
        tr_start = 0
        tr_end = 1500
        te_start = 1500
        te_end = 5000
        start = 772
        end = 1065
        lookback = 256
        unit_i = 32
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_1_label.csv', names=["A"]).values
    elif data_name == "stdb_308_0_2":
        tr_start = 0
        tr_end = 1500
        te_start = 1500
        te_end = 5000
        start = 763
        end = 1053
        lookback = 256
        unit_i = 32
        true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/stdb_308_0_2_label.csv', names=["A"]).values


    df = pd.read_csv(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt', header=None, delimiter='\t')
    print(df.shape)
    if data_name == "nprs44":
        ecg = df.values
    if data_name == "chfdb_chf01_275_1" or data_name == "mitdb__100_180_1" or data_name == "stdb_308_0_1":
        ecg = df.iloc[:,1].values
    if data_name == "chfdb_chf01_275_2" or data_name == "mitdb__100_180_2" or data_name == "stdb_308_0_2":
        ecg = df.iloc[:,2].values
    
    ecg = ecg.reshape(len(ecg), -1)
    print("length of ECG data:", len(ecg))
    
    scaler = StandardScaler()
    std_ecg = scaler.fit_transform(ecg)
    
    normal_cycle = std_ecg[tr_start:tr_end+1]
    test_cycle = std_ecg[te_start:te_end+1]

    #lookback = 128
    pred_length = 3
    step = 1
    delay = 1
    batch_size = 20
    ep = 15
    unit = 32

    for unit_i in [8, 16, 32]:
        for lookback in [128, 256, 64]:
            print("length", tr_start, tr_start+(tr_end-tr_start)*0.8-1, tr_start+(tr_end-tr_start)*0.8, tr_end)

            train_gen = generator(normal_cycle, 
                            lookback=lookback,
                            pred_length=pred_length,
                            delay=delay,
                            min_index=tr_start-tr_start,
                            max_index=int(tr_start+(tr_end-tr_start)*0.7-1)-tr_start,
                            shuffle=True,
                            step=step,
                            batch_size=batch_size)

            val_gen = generator(normal_cycle, 
                            lookback=lookback,
                            pred_length=pred_length,
                            delay=delay,
                            min_index=int(tr_start+(tr_end-tr_start)*0.7)-tr_start,
                            max_index=tr_end-tr_start,
                            step=step,
                            batch_size=batch_size)
                            
            val_steps = (tr_end - int(tr_start+(tr_end-tr_start)*0.7) - lookback) // batch_size
            print("v", val_steps, tr_end, int(tr_start+(tr_end-tr_start)*0.7), lookback)
            np.random.seed(0)
            model = Sequential()
            model.add(layers.GRU(unit_i, return_sequences = True, input_shape=(None,normal_cycle.shape[-1])))
            model.add(layers.GRU(unit_i))
            model.add(layers.Dense(pred_length))

            model.compile(optimizer=RMSprop(), loss="mse")
            
            history = model.fit_generator(train_gen,
                                        steps_per_epoch=200,
                                        epochs=ep,
                                        validation_data=val_gen,
                                        validation_steps=val_steps)

            new_dir_path = f'image_gru1'
            os.makedirs(new_dir_path, exist_ok=True)                          
            plt.plot(history.history['loss'], color='b', label='acc')
            plt.plot(history.history['val_loss'], color='orange', label='val_acc')
            plt.close()
            #plt.show()
            print(history.history)

            test_gen_pred = generator(test_cycle, 
                            lookback=lookback,
                            pred_length=pred_length,
                            delay=delay,
                            min_index=te_start-te_start,
                            max_index=te_end-te_start,
                            step=step,
                            batch_size=batch_size)
            print(len(normal_cycle))
            test_steps = (te_end - te_start - lookback) // batch_size

            test_pred = model.predict_generator(test_gen_pred, steps=test_steps)
            
            test_gen_target = generator(test_cycle, 
                            lookback=lookback,
                            pred_length=pred_length,
                            delay=delay,
                            min_index=te_start-te_start,
                            max_index=te_end-te_start,
                            step=step,
                            batch_size=batch_size)

            test_target = np.zeros((test_steps * batch_size , pred_length))
            print(test_pred.shape, test_target.shape)
            pred0 = (test_pred - test_target)**2
            for i in range(test_steps):
                #print(i)
                test_target[i*batch_size:(i+1)*batch_size] = next(test_gen_target)[1]

            plt.plot(pred0)
            plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/lstm_{data_name}_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_error2.png')
            #plt.axvspan(start - shift, end - shift, color="lightcoral")
            #plt.show()
            plt.close()

            plt.plot(test_pred)
            plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/lstm_{data_name}_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_pred.png')
            #plt.axvspan(start, end, color="lightcoral")
            #plt.show()
            plt.close()


            plt.plot(test_target)
            plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/lstm_{data_name}_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_target.png')
            #plt.show()
            plt.close()

            error =  test_pred - test_target
            print("xxx", test_pred.shape, test_target.shape, error.shape)
            mean = np.mean(error, axis=0)
            print("m", mean)
            cov = np.cov(error, rowvar=False, bias=True)
            print("c", cov)

            detection_gen_pred = generator(test_cycle, 
                        lookback=lookback,
                        pred_length=pred_length,
                        delay=delay,
                        min_index=te_start-te_start,
                        max_index=te_end-te_start,
                        step=step,
                        batch_size=batch_size)

            detection_steps = (te_end - te_start - lookback) // batch_size

            detection_pred = model.predict_generator(detection_gen_pred, steps=detection_steps)
            detection_gen_target = generator(test_cycle, 
                        lookback=lookback,
                        pred_length=pred_length,
                        delay=delay,
                        min_index=te_start-te_start,
                        max_index=te_end-te_start,
                        step=step,
                        batch_size=batch_size)
            print("ddd", detection_steps, batch_size, pred_length)
            detection_target = np.zeros((detection_steps * batch_size , pred_length))

            for i in range(detection_steps):
                detection_target[i*batch_size:(i+1)*batch_size] = next(detection_gen_target)[1]

            error_detection = detection_pred - detection_target 
            print("er", error_detection.shape)
            m_dist = []

            for e in error_detection:
                #print("1", e)
                m_dist.append(Mahalanobis_dist(e, mean, cov))

            plt.axvspan(start, end, color="lightcoral")
            if data_name == 'nprs44':
                plt.axvspan(2087 - 0, 2553 - 0, color="lightcoral")
            plt.plot(m_dist)
            plt.savefig(f'{parentpath1(__file__, f=0)}/{new_dir_path}/lstm_{data_name}_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_u{unit_i}_mahala.png')

            print("start, end", start, end)
            #plt.show()
            plt.close()

            if data_name == 'chfdb_chf01_275_1':
                true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_1_label.csv', names=["A"]).values
                #print(1)
            elif data_name == 'chfdb_chf01_275_2':
                true = pd.read_csv(f'{parentpath1(__file__, f=0)}/correct_label/chfdb_chf01_275_2_label.csv', names=["A"]).values
                print(2)
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
            np.savetxt(f'{new_dir_path}/lstm_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_u{unit_i}_error2.csv', pred0, delimiter=',')
            np.savetxt(f'{new_dir_path}/lstm_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_u{unit_i}_mahala.csv', m_dist, delimiter=',')
            m_dist = np.array(m_dist)
            print(pred0.shape, true.shape, m_dist.shape)
            dif = true.shape[0] - m_dist.shape[0]
            #fpr0, tpr0, thresholds0 = metrics.roc_curve(true[dif:], pred0)
            #fpr1, tpr1, thresholds0 = metrics.roc_curve(true[:m_dist.shape[0]], pred0)
            fpr2, tpr2, thresholds2 = metrics.roc_curve(true[dif:], m_dist)
            fpr3, tpr3, thresholds3 = metrics.roc_curve(true[:m_dist.shape[0]], m_dist)
            #auc0 = metrics.auc(fpr0, tpr0)
            #auc1 = metrics.auc(fpr1, tpr1)
            auc2 = metrics.auc(fpr2, tpr2)
            auc3 = metrics.auc(fpr3, tpr3)
            #print(auc0)
            #print(auc1)
            
            youden = tpr3 - fpr3
            #print(pred.shape)
            #print(fpr[np.argmax(youden)], tpr[np.argmax(youden)], thresholds_pre[np.argmax(youden)])
            thresholds_best = thresholds3[np.argmax(youden)]
            pred_bin = np.zeros(m_dist.shape[0])
            for i in range(m_dist.shape[0]):
                if m_dist[i] > thresholds_best:
                    pred_bin[i] = 1
                else:
                    pred_bin[i] = 0

            fscore = f1_score(true[:m_dist.shape[0]], pred_bin)
            print(f'{data_name}_l{lookback}_p{pred_length}_b{batch_size}_e{ep}_u{unit_i}')
            print("auc", auc2)
            print("auc", auc3)
            print("fscore", fscore)
            print("")
            fscore_list0.append(fscore)
            #auc_list1.append(auc2)
            auc_list0.append(auc3)
    auc_list0 = np.array(auc_list0)
    fscore_list0 = np.array(fscore_list0)
    auc_list1.append(np.max(auc_list0))
    fscore_list1.append(fscore_list0[np.argmax(auc_list0)])
for i in range(len(auc_list1)):
    print(f'data:{data_list[i]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
#for i in range(len(fscore_list1)):
    #print(fscore_list1[i])

'''
print(f'data:{data_list[0]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[1]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[2]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[3]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[4]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[5]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
print(f'data:{data_list[6]}, AUC:{auc_list1[i]}, f_score:{fscore_list1[i]}')
'''