import numpy as np
import pywt
import scipy.signal as signal

def wavelet_coef_freq(signal, Fs=250, scale=(1,11), waveletname = 'cmor',method_='conv'):
  Ts=1/Fs
  scales=np.arange(scale[0],scale[1])
  f = pywt.scale2frequency(waveletname,scales)*Fs
  [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, 1/Fs, method=method_)
  return [coefficients, frequencies]

def D2_wavelet_coef_freq(X, Fs=250, scale=(1,11), waveletname = 'cmor',method_='conv'):
  assert X.ndim == 2
  coeffs = np.empty((X.shape[0],scale[1]-scale[0],X.shape[1]))

  for ch in range(X.shape[0]):
    coeffs[ch,:] = wavelet_coef_freq(X[ch,:],Fs,scale,waveletname,method_)[0]

  return coeffs

def D3_wavelet_coef_freq(X, Fs=250, scale=(1,11), waveletname = 'cmor',method_='conv'):
  assert X.ndim == 3
  
  coeffs = np.empty((X.shape[0],X.shape[1],scale[1]-scale[0],X.shape[2]))

  for ch in range(X.shape[0]):
    for i in range(X.shape[1]):
      coeffs[i,:,:] = D2_wavelet_coef_freq(X[i,:,:], Fs, scale, waveletname, method_)

  return coeffs

def sc_wavelet_coef_freq(X,width=np.arange(1,31)):
  return signal.cwt(X,signal.ricker,width)

def sc_D2_wavelet_coef_freq(X,width=np.arange(1,31)):
  assert X.ndim==2
  cwtmatr = np.empty((X.shape[0],len(width),X.shape[1]))

  for ch in range(X.shape[0]):
    cwtmatr[ch,:,:] = sc_wavelet_coef_freq(X[ch,:],width)
  
  return cwtmatr

def sc_D3_wavelet_coef_freq(X,width=np.arange(1,31)):
  assert X.ndim==3
  cwtmatr = np.empty((X.shape[0],X.shape[1],len(width),X.shape[2]))

  for ch in range(X.shape[0]):
    cwtmatr[ch,:,:,:] = sc_D2_wavelet_coef_freq(X[ch,:,:],width)

  return cwtmatr
