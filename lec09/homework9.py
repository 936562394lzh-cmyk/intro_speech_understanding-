import numpy as np

def VAD(waveform, Fs):
    """
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.

    @params:
        waveform (np.ndarray(N)) - the waveform
        Fs (scalar) - sampling rate

    @returns:
        segments (list of arrays) - list of the waveform segments where energy is greater than 10% of maximum energy
    """
    frame_len = int(Fs * 0.025)  
    frame_step = int(Fs * 0.01)  

    num_frames = max(1, (len(waveform) - frame_len) // frame_step + 1)
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_len
        frame = waveform[start:end]
        energies[i] = np.sum(frame ** 2)

    threshold = 0.1 * np.max(energies)
    above = energies > threshold

    segments = []
    i = 0
    while i < len(above):
        if above[i]:
            start_idx = i
            while i < len(above) and above[i]:
                i += 1
            end_idx = i - 1
            seg_start = start_idx * frame_step
            seg_end = min(end_idx * frame_step + frame_len, len(waveform))
            segments.append(waveform[seg_start:seg_end])
        else:
            i += 1
    return segments


def segments_to_models(segments, Fs):
    """
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra to make the model.

    @params:
        segments (list of arrays) - waveform segments that contain speech
        Fs (scalar) - sampling rate

    @returns:
        models (list of arrays) - average log spectra of pre-emphasized waveform segments
    """
    pre_emph = 0.97

    frame_len = int(Fs * 0.004)  
    frame_step = int(Fs * 0.002) 

    window = np.hamming(frame_len)

    models = []
    for seg in segments:
        if len(seg) > 1:
            seg_emph = seg - pre_emph * np.concatenate(([0], seg[:-1]))
        else:
            seg_emph = seg

        n_frames = max(1, (len(seg_emph) - frame_len) // frame_step + 1)
        n_fft = 1 << (frame_len - 1).bit_length()
        half_n = n_fft // 2
        spec_sum = np.zeros(half_n, dtype=np.float64)

        for i in range(n_frames):
            start = i * frame_step
            end = start + frame_len
            frame = seg_emph[start:end] * window
            fft_vals = np.fft.rfft(frame, n=n_fft) 
            power = np.abs(fft_vals[:half_n]) ** 2
            spec_sum += power

        avg_power = spec_sum / n_frames
        log_spec = np.log(avg_power + 1e-12)
        models.append(log_spec)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    """
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.

    @params:
        testspeech (array) - test waveform
        Fs (scalar) - sampling rate
        models (list of Y arrays) - list of model spectra
        labels (list of Y strings) - one label for each model

    @returns:
        sims (Y-by-K array) - cosine similarity of each model to each test segment
        test_outputs (list of strings) - recognized label of each test segment
    """
    test_segments = VAD(testspeech, Fs)
    if not test_segments:
        return np.array([]), []

    test_models = segments_to_models(test_segments, Fs)

    Y = len(test_models)  
    K = len(models)      
    sims = np.zeros((Y, K))

    for i, test_vec in enumerate(test_models):
        norm_test = np.linalg.norm(test_vec)
        if norm_test == 0:
            continue
        for j, model_vec in enumerate(models):
            norm_model = np.linalg.norm(model_vec)
            if norm_model == 0:
                sims[i, j] = 0.0
            else:
                sim = np.dot(test_vec, model_vec) / (norm_test * norm_model)
                sims[i, j] = sim

    test_outputs = []
    for i in range(Y):
        best_idx = np.argmax(sims[i, :])
        test_outputs.append(labels[best_idx])

    return sims, test_outputs
