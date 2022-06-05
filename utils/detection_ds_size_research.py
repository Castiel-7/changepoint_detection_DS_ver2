import numpy as np
import math

class SubspaceMethod(object):
    def __init__(self, threshold = 0.95, r = None):
        #print(1)
        self.threshold = threshold
        self.r = r
    
    def fit(self, train_X):
        if train_X.shape[0]<train_X.shape[1]:
            #print(2)
            self.fit_dual(train_X)
        else:
            #print(3)
            self.fit_primal(train_X)

    def _get_dim(self, e_val):
        if self.r is not None:
            #print(4)
            return self.r
        else:
            #print(5)
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1,len(e_val)+1)])
            r = np.min(np.where(sum_value>=self.threshold)[0])+1
            return r

    def fit_primal(self, X):
        #print(6)
        K = X.T@X/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        self.coef_ = e_val[:r]
        self.components_ = e_vec.T[:r].T

    def fit_dual(self, X):
        #print(7)
        #print(X.shape)
        K = X@X.T/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        V = X.T@e_vec/np.sqrt(e_val.reshape(1,-1)*X.shape[0])
        self.coef_ = e_val[:r]
        self.components_ = V.T[:r].T

    def score(self, test_X):
        #print(8)
        I = np.identity(test_X.shape[1])
        error = np.linalg.norm(test_X@(I-self.components_@self.components_.T), axis=1).reshape(-1)/np.linalg.norm(test_X, axis=1).reshape(-1)
        return np.fabs(error)

class SubspaceMethod1(object):
    def __init__(self, threshold = 0.95, r = None):
        #print(1)
        self.threshold = threshold
        self.r = r
    
    def fit(self, train_X):
        if train_X.shape[0]<train_X.shape[1]:
            #print(2)
            self.fit_dual(train_X)
        else:
            #print(3)
            self.fit_primal(train_X)

    def _get_dim(self, e_val):
        if self.r is not None:
            #print(4, self.r)
            return self.r
        else:
            #print(5)
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1,len(e_val)+1)])
            r = np.min(np.where(sum_value>=self.threshold)[0])+1
            return r

    def fit_primal(self, X):
        #print(6)
        K = X.T@X/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        #print(e_val.shape)
        r = self._get_dim(e_val)
        self.coef_ = e_val[:r]
        self.components_ = e_vec.T[:r].T

    def fit_dual(self, X):
        #print(7)
        #print(X.shape)
        K = X@X.T/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        V = X.T@e_vec/np.sqrt(e_val.reshape(1,-1)*X.shape[0])
        self.coef_ = e_val[:r]
        self.components_ = V.T[:r].T

    def score(self, test_X):
        #print(8)
        I = np.identity(test_X.shape[1])
        error = np.linalg.norm(test_X@(I-self.components_@self.components_.T), axis=1).reshape(-1)/np.linalg.norm(test_X, axis=1).reshape(-1)
        return np.fabs(error)

class DetectionDS(object):
    def __init__(self, window_length=128, order=64, lag = 64, M = 5, N = 10, DS_dim = 5, PS_dim = 30):
        self.window_length = window_length
        self.order = order
        self.lag = lag
        self.M = M
        self.N = N
        self.DS_dim = DS_dim
        self.PS_dim = PS_dim
        self.ds7 = np.zeros(10)
        
    def fit(self, x):
        start_idx = 0
        end_idx = len(x) - self.window_length - self.order - self.lag
        #print("K_num:", self.K_num)
        ds_list = []
        pds_list = []
        score_list = []
        vol_ruiwa_train = []
        vol_log_train = []
        count = 0
        #print(start_idx)
        #print(end_idx)
        for t in range(start_idx, end_idx):
            #print(t)
            train_H = self._get_hankel(x, order = self.order,
                                       start = t,
                                       end = t + self.window_length)
            test_H = self._get_hankel(x, order = self.order,
                                     start = t + self.lag,
                                     end = t + self.window_length + self.lag)
            
            sm = SubspaceMethod(r = self.M)
            sm.fit(train_H.T)
            subspace1 = sm.components_

            sm = SubspaceMethod(r = self.N)
            sm.fit(test_H.T)
            subspace2 = sm.components_
            
            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            #volume = np.sum(np.log(S))
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            #print(S, volume_ruiwa, volume_log)
            vol_ruiwa_train.append(volume_ruiwa)
            vol_log_train.append(volume_log)

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            #d = v[:, int(v.shape[1]/2):int(v.shape[1]/2)+self.DS_dim]
            d1 = v[:, self.DS_dim < w]
            #print("1", 0.0001 < w, d1.shape)
            w = w[0:d1.shape[1]]
            d2 = d1[:, 1 > w]
            #print("2", 1>w, d2.shape)


            #print(d2.shape)

            if t == 0:
                self.ds7 = d2
                #print(self.ds7)
            else:
                self.ds7 = np.concatenate([self.ds7, d2], 1)
            
            #print(self.ds7.shape)
            #print(d.shape)

            #print("1d", d.shape, int(v.shape[1]/2))
            '''
            d1 = v[:, 0.0001 < w]
            d = d1[:, d1.shape[1]-self.DS_dim:d1.shape[1]]
            '''

            '''
            d = v[:, w < 1]
            #print(v.shape)
            if d.shape[1] < self.M:
                continue
            '''
            #print(type(d))
            #print(w.shape)
            #print(np.sum(w > 1))
            #print(np.sum(w < 1))
            #print(d.shape)
            #ds_list.append(d)
        #change_vector = np.array(score_list)
        #print(change_vector.shape)
        #print(ds_list)
        #self.ds = np.array(ds_list)
        #ds_split = np.array_split(self.ds, 1)
        #print(self.ds7.T.shape, self.PS_dim)
        sm7 = SubspaceMethod1(r = self.PS_dim)
        sm7.fit(self.ds7.T)
        self.pds = sm7.components_
        #print("7", self.pds.shape)

        '''
        print(self.ds.shape)
        for _ds in ds_split:
            print(_ds.shape)
            _ds = _ds.transpose(0, 2, 1)
            _ds = _ds.reshape(-1, _ds.shape[-1])
            print(_ds.shape)
            sm = SubspaceMethod(r = self.M)
            sm.fit(_ds)
            subspace_ds = sm.components_
            print(subspace_ds)
            pds_list.append(subspace_ds)
        self.pds = np.array(pds_list)
        '''
        self.vol_ruiwa_train = np.array(vol_ruiwa_train)
        self.mean_ruiwa = np.mean(self.vol_ruiwa_train)
        self.vol_log_train = np.array(vol_log_train)
        self.mean_log = np.mean(self.vol_log_train)
        #print("pds", self.pds.shape)
        #print(self.ds.shape)
        #print(self.ds.shape)
        #e_val, e_vec = np.linalg.eigh(change_vector.T@change_vector/change_vector.shape[0])
        #e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        #print("e_val", e_val.shape, "e_vec", e_vec.shape)
        #V = e_vec.T[:29].T
        #print(V.shape)
        #self.Q = np.identity(e_vec.shape[0])-V@V.T
        #print(self.Q.shape)
        #score_list1 = np.array(score_list)
        #print(np.array(score_list).shape)
        return 1
            
            
    def _get_hankel(self, x, order, start, end):
        return np.array([x[start+i:end+i] for i in range(order)]).T
    
    def svd_score(self, s1, s2):
        _, S, _ = np.linalg.svd(s1.T@s2)
        #print(S.shape)
        return S
    
    def where(self, score):
        max_value = np.max(score, axis=0)
        min_value = np.min(score, axis=0)
        mean = np.mean(score, axis=0)
        std = np.std(score, axis=0)
        max_norm = (max_value-mean)/std
        min_norm = (min_value-mean)/std

        max_idx = np.argmax(max_norm+3)
        min_idx = np.argmin(min_norm-3)
        if np.abs(max_norm[max_idx])>np.abs(min_norm[min_idx]):
            return max_idx
        else:
            return min_idx

    def predict(self, x):
        start_idx = 0
        end_idx = len(x) - self.window_length - self.order - self.lag
        
        score_list = []
        similarity_list = []
        similarity_list_k1 = []
        vol_list = []
        count = 0
        #print(start_idx)
        #print(end_idx)
        for t in range(start_idx, end_idx):
            train_H = self._get_hankel(x, order = self.order,
                                       start = t,
                                       end = t + self.window_length)
            test_H = self._get_hankel(x, order = self.order,
                                     start = t + self.lag,
                                     end = t + self.window_length + self.lag)
                                                           
            sm = SubspaceMethod(r = self.M)
            sm.fit(train_H.T)
            subspace1 = sm.components_

            sm = SubspaceMethod(r = self.N)
            sm.fit(test_H.T)
            subspace2 = sm.components_

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            #volume = np.sum(np.log(S))
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            #print(S, volume_ruiwa, volume_log)
            vol_list.append(volume_ruiwa)
            dif_ruiwa = (volume_ruiwa - self.mean_ruiwa) ** 2
            dif_log = (volume_log - self.mean_log) ** 2
            #d1 = v[:, 0.0001 < w]
            #print(w)
            #print(w>0.0001)
            #print(w[])

            #d = v[:, int(v.shape[1]/2):int(v.shape[1]/2)+self.DS_dim]
            #print("2d", d.shape, int(v.shape[1]/2))
            '''
            d1 = v[:, 0.001 < w]
            d = d1[:, d1.shape[1]-self.DS_dim:d1.shape[1]]
            '''
            d3 = v[:, self.DS_dim < w]
            #print("1", 0.0001 < w, d1.shape)
            w = w[0:d3.shape[1]]
            d = d3[:, 1 > w]

            #print("d1", d1)
            #print("d", d)
            #print(d1.shape, d.shape)
            #print(w.shape)
            #print(t)
            #print(np.sum(w > 1))
            #print(np.sum(w < 1))
            #print(d.shape)
            #print(self.ds.shape)
            #similarity = [self.svd_score(d, _X) for _X in self.ds]
            try:
                _, similarity, _ = np.linalg.svd(d.T@self.pds)
            except:
                similarity_list = []
                A = d.T@self.pds
                for i in range(A.shape[0]):
                    w1, v1 = np.linalg.eigh(np.dot(A[i,:,:].T, A[i,:,:]))
                    w1 = w1[::-1]; v1 = v1[:,::-1]
                    similarity_pre = np.sqrt(w1)
                    similarity_list.append(similarity_pre)
                similarity = np.array(similarity_list)
            #print("sim", similarity.shape)
            #print(similarity.shape)
            similarity_list_k0 = []

            similarity_mean1 = 1 - np.mean(similarity[0:1])
            similarity_mean1_ruiwa = dif_ruiwa * similarity_mean1
            similarity_mean1_log = dif_log * similarity_mean1
            similarity_list_k0.append(similarity_mean1_ruiwa)
            similarity_list_k0.append(similarity_mean1_log)

            similarity_mean5 = 1 - np.mean(similarity[0:5])
            similarity_mean5_ruiwa = dif_ruiwa * similarity_mean5
            similarity_mean5_log = dif_log * similarity_mean5
            similarity_list_k0.append(similarity_mean5_ruiwa)
            similarity_list_k0.append(similarity_mean5_log)

            similarity_mean_all = 1 - np.mean(similarity)
            similarity_meanall_ruiwa = dif_ruiwa * similarity_mean_all
            similarity_meanall_log = dif_log * similarity_mean_all
            similarity_list_k0.append(similarity_meanall_ruiwa)
            similarity_list_k0.append(similarity_meanall_log)

            similarity_list_k1.append(similarity_list_k0)
            
        #similarity_list = np.array(similarity_list)
        similarity_list_k1 = np.array(similarity_list_k1)
        #similarity_list_k1 = np.array(vol_list)
        #print("k1_shape", similarity_list_k1.shape)
        #print("similarity_list", similarity_list.shape)
        #np.savetxt('Grassman_slide_w'+str(self.window_length)+'_o'+str(self.order)+'_l'+str(self.lag)+'_M'+str(self.M)+'.csv', similarity_list ,delimiter=',')
        #print("change_vector", change_vector.shape)
        #print("Q", self.Q.shape)
        #print(np.linalg.norm(change_vector@self.Q, axis=1).shape)
        return similarity_list_k1