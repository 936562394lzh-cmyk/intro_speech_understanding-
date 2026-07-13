import numpy as np

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''

    nframes = (len(speech) - frame_length) // frame_skip + 1
    if nframes <= 0:
        raise ValueError("信号长度小于帧长，无法分帧")
    
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        x = speech[start:start + frame_length]     
        
        R = np.array([np.sum(x[:len(x)-k] * x[k:]) for k in range(order + 1)])
        
        a = np.zeros(order + 1)
        a[0] = 1.0
        if order > 0:
            E = R[0]
            for m in range(1, order + 1):

                summ = R[m]
                for j in range(1, m):
                    summ += a[j] * R[m - j]
                k = -summ / E

                a_old = a.copy()
                for j in range(1, m):
                    a[j] = a_old[j] + k * a_old[m - j]
                a[m] = k
                E *= (1 - k * k)
        A[i] = a
        
        e_frame = np.zeros(frame_length)
        for n in range(frame_length):
            pred = 0.0
            for k in range(1, order + 1):
                if n - k >= 0:
                    pred += a[k] * x[n - k]
            e_frame[n] = x[n] - pred
        excitation[i] = e_frame
    
    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    order = A.shape[1] - 1
    total_samples = len(e)
    expected = nframes * frame_skip
    if total_samples != expected:
        raise ValueError(f"激励信号长度 {total_samples} 与期望 {expected} 不符")
    
    synthesis = np.zeros(total_samples)

    state = np.zeros(order)
    
    for i in range(nframes):
        start = i * frame_skip
        a = A[i]    
        for n in range(frame_skip):
            idx = start + n
            
            y_val = e[idx] - np.dot(a[1:], state)
            synthesis[idx] = y_val

            state = np.concatenate(([y_val], state[:-1]))
    
    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]

    gain = np.array([np.sqrt(np.mean(frame**2)) for frame in excitation])
    
    total = nframes * frame_skip
    e_robot = np.zeros(total)

    for n in range(total):
        if n % T0 == 0:
            frame_idx = n // frame_skip
            e_robot[n] = gain[frame_idx]
    
    return gain, e_robot
