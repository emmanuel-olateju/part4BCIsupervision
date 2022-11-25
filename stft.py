import numpy as np
import pandas as pd
import scipy

default_stft_params = {
    'fs':250,
    'window':'hamming',
    'nperseg':256,
    'nfft':256,
    'detrend':False,
    'return_onesided':True,
    'boundary':'zeros',
    'padded':True,
    'axis':-1
}

class STFT:
    def __init__(self,params):
        self.stft=dict()
        self.stft_params=params

    def compute(self,data):
        self.stft['f'],self.stft['t'],self.stft['Zxx']=scipy.signal.stft(data,**self.stft_params)

    def t_ch_f(self,f1,f2,t1,t2):
        return np.transpose(self.stft['Zxx'][:,:,f1:f2,t1:t2],[0,3,1,2])

    def t_f_ch(self,f1,f2,t1,t2):
        return np.transpose(self.stft['Zxx'][:,:,f1:f2,t1:t2],[0,3,2,1])
    
    def f_ch_t(self,f1,f2,t1,t2):
        return np.transpose(self.stft['Zxx'][:,:,f1:f2,t1:t2],[0,2,1,3])

    def f_t_ch(self,f1,f2,t1,t2):
        return np.transpose(self.stft['Zxx'][:,:,f1:f2,t1:t2],[0,2,3,1])
    
    def ch_t_f(self,f1,f2,t1,t2):
        return np.transpose(self.stft['Zxx'][:,:,f1:f2,t1:t2],[0,1,3,2])
    
    def ch_f_t(self,f1,f2,t1,t2):
        return self.stft['Zxx'][:,:,f1:f2,t1:t2]
    
    def __labelIndex(self):
        self.oneIndex=list()
        self.zeroIndex=list()
        for i in range(0,len(self.y)):
            if self.y[0]==1:
                self.oneIndex.append(i)
            else:
                self.zeroIndex.append(i)
    
    def averageSTFT(self,data,label):
        classes=list(pd.Series(label).unique())
        self.stft_class_average=dict.fromkeys(classes)
        for c in classes:
            count=0
            avg=0
            for i in range(0,len(label)):
                if label[i]==c:
                    count=count+1
                    avg=avg+data[i,:,:,:]
            avg=avg/count
            self.stft_class_average[c]=avg

def t_ch_f(data,f1,f2,t1,t2):
    return np.transpose(data[:,:,f1:f2,t1:t2],[0,3,1,2])

def t_f_ch(data,f1,f2,t1,t2):
    return np.transpose(data['Zxx'][:,:,f1:f2,t1:t2],[0,3,2,1])

def f_ch_t(data,f1,f2,t1,t2):
    return np.transpose(data[:,:,f1:f2,t1:t2],[0,2,1,3])

def f_t_ch(data,f1,f2,t1,t2):
    return np.transpose(data[:,:,f1:f2,t1:t2],[0,2,3,1])

def ch_t_f(data,f1,f2,t1,t2):
    return np.transpose(data['Zxx'][:,:,f1:f2,t1:t2],[0,1,3,2])

def ch_f_t(data,f1,f2,t1,t2):
    return data[:,:,f1:f2,t1:t2]