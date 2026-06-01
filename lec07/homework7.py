import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    duration = 0.5                    
    N = int(duration * Fs)            
    t = np.arange(N) / Fs              
    
    r4 = 2 ** (4 / 12)                
    r7 = 2 ** (7 / 12)                
    
    f2 = f * r4
    f3 = f * r7
    
    x = (np.sin(2 * np.pi * f * t) +
         np.sin(2 * np.pi * f2 * t) +
         np.sin(2 * np.pi * f3 * t))
    
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N)[:, np.newaxis]   
    n = np.arange(N)                  
    
    W = np.exp(-2j * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    N = len(x)
    X = np.fft.fft(x)
    mag = np.abs(X[:N//2])           
    freq = np.arange(N//2) * Fs / N  
    
    mag[0] = 0
    
    if len(mag) < 3:
        indices = np.argsort(mag)[::-1][:len(mag)]
    else:
        indices = np.argpartition(mag, -3)[-3:]   # last 3 indices in unsorted order
        indices = indices[np.argsort(mag[indices])[::-1]]
    
    freqs = freq[indices]
    f1, f2, f3 = np.sort(freqs)
    
    return f1, f2, f3
