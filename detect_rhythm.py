import scikits.audiolab as au
import os
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from scipy.signal import fftconvolve, correlate2d, periodogram
from matplotlib.pylab import *
from scipy.signal import butter, lfilter, hilbert
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf

os.chdir("/home/frans/music/")

f = au.Sndfile('drumbeat.wav', 'r')
fs = f.samplerate
nc = f.channels
enc = f.encoding
nseconds = 50 # length of song


data = f.read_frames(fs*nseconds, dtype=np.float32)

downsample_rate = 4
dat = data[range(0,nseconds*fs,downsample_rate),:]

#length = fs*4
#clf()
#subplot(2,1,1)
#plot(dat[:length,0])
#subplot(2,1,2)
#plot(dat[:length,1])
#xticks(map(round,np.arange(0, length+1, fs/5.)),np.arange(0, length/fs, 0.2))
#show()

left_channel = dat[:,1]
fsd = fs/downsample_rate
#fft_size = 10
length = fsd*nseconds

#figure()
(spectrum,freqs,t,im) = specgram(left_channel[:length], NFFT=256, Fs=fsd,cmap='jet')
#show(im)

#D = cdist(spectrum.T, spectrum.T,lambda u, v: np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
D = cdist(spectrum.T, spectrum.T,'cosine')
#clf()
#show(matshow(D))


B = [D[range(D.shape[0]-l),range(l,D.shape[0])].sum() for l in range(D.shape[0])]
clf()
plot(t,B)
#xticks(map(round,np.arange(0, length+1, fsd/5.)),np.arange(0, length/fsd, 0.2))
show()

B2 = np.array([fftconvolve(D[i,:], D[i,::-1], mode = 'full') for i in range(D.shape[0])])
B2 = B2[:,B2.shape[0]:]
plot(B2.sum(axis=0))
figure()
plot(abs(hilbert(B2.sum(axis=0))))
show()

show(semilogy(*periodogram(B2[0,:],344)))
B3 = np.array([butter_bandpass_filter(B2[i,:],1,100,344,8) for i in range(D.shape[0])])
#show(plot(t[:-1],B3[2000,:]))
#show(plot(B2[0,B2.shape[0]:]))
#show(plot(B2[0,B2.shape[0]:]))
#show(matshow(B2))

#beatspectrum = B3[:10,:].sum(axis=0)
freq = int(float(spectrum.shape[1])/nseconds)
def beat_spectrogram(B2, windowsize):
    C = []
    i = 0
    for window in range(B2.shape[0]/windowsize):
        beatspectrum = B2[window*windowsize:(window+1)*windowsize,:].sum(axis=0)
        beatspectrum = beatspectrum/beatspectrum[0]
        clean_beat = beatspectrum/moving_average(beatspectrum,freq*10)
        clean_beat = clean_beat[::-1]
        C.append(clean_beat)
        i += 1
    return np.array(C)

C = beat_spectrogram(B2, 10)
show(plot(t[:4*freq], clean_beat[:4*freq]))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


beatcepstrum = fftconvolve(tmp.seasonal,tmp.seasonal[::-1], mode='full')
beatcepstrum = beatcepstrum[(beatcepstrum.shape[0]/2):]
show(plot(t[:freq*4], tmp.seasonal[:freq*4]))

accent_frames = [i[0] for i in sorted(enumerate(tmp.seasonal), key=lambda x:x[1])]

show(plot(t[:-1], beatcepstrum))
tmp.seasonal

beat = pacf(tmp.seasonal, nlags=freq*10)
markerline, stemlines, baseline = stem(range(len(beat)),beat, '-.')
setp(markerline, 'markerfacecolor', 'b')
setp(baseline, 'color','r', 'linewidth', 2)
show()

accent_frames = [i[0] for i in sorted(enumerate(beat), key=lambda x:x[1])]

[val, idx] = np.max(beat)

output = au.Sndfile('output.flac', 'w', au.Format('wav'), 2, fs*2/downsample_rate)
output.write_frames(dat)
output.close()

def butter_bandpass(lowcut,highcut,fs,order=8):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    b,a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=8):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)
    return lfilter(b,a,data) 