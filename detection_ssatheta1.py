'''
only direction
Use eigenvalues all over d(ex:1e-05)
detection type series: data renewal(train_length:1000(about 30%), test_length:2500(about 70%))
'''
import sys
import os
import os.path
import numpy as np
from utils.load_data import load
from utils.detection_ssa_theta1 import SSAtheta1
    
def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def main():
    
    #with open(f'{parentpath1(__file__, f=0)}/data_name_ver3.txt', "r") as f:
        #lines1 = f.readlines()

    #with open(f'{parentpath1(__file__, f=0)}/set_param_grassman_ver3.txt', "r") as f:
        #lines2 = f.readlines()
        
    for i in range(1):
        for i in range(1):
            data_list_dummy = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "chfdb_chf13_45590_1", "chfdb_chf13_45590_2", "chfdbchf15_1", "chfdbchf15_2", "mitdb__100_180_1", "mitdb__100_180_2"]
            data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
            me = 'ssa_theta1'
            window_list = [64, 128, 256]
            order_list = [64, 128, 256]
            lag_list = [0.7]
            M_list = [1, 5, 10, 15, 20, 25, 35, 40, 45, 50, 55, 60]
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
        
if __name__=='__main__':
    main()