import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    
    @param:
    duration (scalar) - length of the excitation, in samples
    F0 (scalar) - pitch frequency, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    excitation (np.ndarray) - the excitation signal, such that
      excitation[n] = -1 if n is an integer multiple of int(np.round(Fs/F0))
      excitation[n] = 0 otherwise
    '''
    T = int(np.round(Fs / F0))          
    excitation = np.zeros(duration)
    
    excitation[0::T] = -1              
    return excitation

def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    
    @param:
    x (np.ndarray(N)) - the excitation signal
    F (scalar) - resonant frequency, in Hertz
    BW (scalar) - resonant bandwidth, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    y (np.ndarray(N)) - resonant output
    '''
    N = len(x)
    y = np.zeros(N)
    # 极点半径 r 和角度 theta
    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs
    a0 = 1 - r**2
    b1 = 2 * r * np.cos(theta)
    b2 = -r**2
    
    y_1 = 0.0  
    y_2 = 0.0 
    for n in range(N):
        y_n = a0 * x[n] + b1 * y_1 + b2 * y_2
        y[n] = y_n
        y_2 = y_1
        y_1 = y_n
    return y

def synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel.
    
    @param:
    duration (scalar) - duration in samples
    F0 (scalar) - pitch frequency in Hertz
    F1 (scalar) - first formant frequency in Hertz
    F2 (scalar) - second formant frequency in Hertz
    F3 (scalar) - third formant frequency in Hertz
    F4 (scalar) - fourth formant frequency in Hertz
    BW1 (scalar) - first formant bandwidth in Hertz
    BW2 (scalar) - second formant bandwidth in Hertz
    BW3 (scalar) - third formant bandwidth in Hertz
    BW4 (scalar) - fourth formant bandwidth in Hertz
    Fs (scalar) - sampling frequency in samples/second
    
    @returns:
    speech (np.ndarray(samples)) - synthesized vowel
    '''
    
    excitation = voiced_excitation(duration, F0, Fs)
    
    speech = resonator(excitation, F1, BW1, Fs)
    speech = resonator(speech, F2, BW2, Fs)
    speech = resonator(speech, F3, BW3, Fs)
    speech = resonator(speech, F4, BW4, Fs)
    
    return speech
