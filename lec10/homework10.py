import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    '''
    alpha = 0.97                      
    frame_len_ms = 4                 
    step_ms = 2                       
    vad_win_ms = 25                   
    vad_step_ms = 10                  
    
    frame_len = int(frame_len_ms * Fs / 1000)
    step = int(step_ms * Fs / 1000)
    vad_win = int(vad_win_ms * Fs / 1000)
    vad_step = int(vad_step_ms * Fs / 1000)
    
    preemph = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])
    
    waveform_t = torch.from_numpy(preemph).float()
    window = torch.hann_window(frame_len)
    
    stft = torch.stft(waveform_t, n_fft=frame_len, hop_length=step, win_length=frame_len,
                      window=window, center=False, return_complex=True)

    spec = torch.abs(stft).numpy()   
   
    features = spec 
    N = features.shape[0]
    
    num_vad_frames = 0
    vad_energies = []
    start = 0
    while start + vad_win <= len(waveform):
        frame_pow = np.sum(waveform[start:start+vad_win] ** 2)
        vad_energies.append(frame_pow)
        start += vad_step
        num_vad_frames += 1
    vad_energies = np.array(vad_energies)
    
    threshold = 0.1 * np.mean(vad_energies)
    vad_active = (vad_energies > threshold).astype(int)
    
    labels_vad = np.zeros(len(vad_active), dtype=int)
    label_id = 1
    i = 0
    while i < len(vad_active):
        if vad_active[i] == 1:
            start_idx = i
            while i < len(vad_active) and vad_active[i] == 1:
                i += 1
            end_idx = i - 1
            labels_vad[start_idx:end_idx+1] = label_id
            label_id += 1
        else:
            i += 1
   
    labels_expanded = np.repeat(labels_vad, 5)  
    if len(labels_expanded) > N:
        labels = labels_expanded[:N]
    else:
        
        pad = N - len(labels_expanded)
        labels = np.pad(labels_expanded, (0, pad), constant_values=0)
    
    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    
    X = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    
    NFEATS = features.shape[1]
    output_dim = int(np.max(labels) + 1)  
    
    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, output_dim)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    lossvalues = []
    for it in range(iterations):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        lossvalues.append(loss.item())
    
    return model, np.array(lossvalues)


def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(features).float()
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()
