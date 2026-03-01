import numpy as np
from scipy.signal import stft
from sklearn.metrics.pairwise import cosine_similarity


def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    # 1. 计算帧长和帧移（转成样本数）
    frame_len = int(0.025 * Fs)  # 25ms帧长
    frame_step = int(0.01 * Fs)  # 10ms帧移
    
    # 2. 分帧并计算每帧能量
    energy = []
    num_frames = 1 + (len(waveform) - frame_len) // frame_step
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_len
        frame = waveform[start:end]
        frame_energy = np.sum(frame **2)  # 能量定义为帧内样本平方和
        energy.append(frame_energy)
    
    # 3. 计算能量阈值（最大能量的10%）
    max_energy = np.max(energy) if energy else 0
    threshold = 0.1 * max_energy
    
    # 4. 找出能量超过阈值的连续帧，提取对应的波形段
    segments = []
    if max_energy == 0:
        return segments  # 空波形直接返回
    
    # 标记有效帧
    valid_frames = np.array(energy) > threshold
    # 找连续有效帧的起始和结束索引
    valid_intervals = []
    start_idx = None
    for i in range(len(valid_frames)):
        if valid_frames[i] and start_idx is None:
            start_idx = i
        elif not valid_frames[i] and start_idx is not None:
            # 计算该段的样本起始/结束位置
            seg_start = start_idx * frame_step
            seg_end = (i-1) * frame_step + frame_len
            segments.append(waveform[seg_start:seg_end])
            start_idx = None
    # 处理最后一段有效帧
    if start_idx is not None:
        seg_start = start_idx * frame_step
        seg_end = (len(valid_frames)-1) * frame_step + frame_len
        segments.append(waveform[seg_start:seg_end])
    
    return segments


def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    pre_emphasis_coeff = 0.97  # 预加重系数
    
    # 1. 定义谱图参数（4ms帧长，2ms帧移）
    frame_len = int(0.004 * Fs)  # 4ms帧长
    frame_step = int(0.002 * Fs)  # 2ms帧移
    nfft = max(512, 2** np.ceil(np.log2(frame_len)))  # FFT点数（取2的幂）
    
    for seg in segments:
        if len(seg) < frame_len:
            continue  # 跳过过短的段
        
        # 2. 预加重（y[n] = x[n] - α*x[n-1]）
        pre_emphasized = np.append(seg[0], seg[1:] - pre_emphasis_coeff * seg[:-1])
        
        # 3. 计算短时傅里叶变换（STFT）得到谱图
        f, t, Zxx = stft(
            pre_emphasized,
            fs=Fs,
            nperseg=frame_len,
            noverlap=frame_len - frame_step,  # 重叠数=帧长-帧移
            nfft=nfft,
            return_onesided=True
        )
        
        # 4. 计算幅度谱的对数（log频谱）
        mag_spec = np.abs(Zxx)
        log_spec = np.log10(mag_spec + 1e-10)  # 加小值避免log(0)
        
        # 5. 保留低频半部分（取频率轴的前1/2）
        low_freq_half = log_spec[:len(f)//2, :]
        
        # 6. 平均所有帧的低频谱，得到该段的模型
        avg_log_spec = np.mean(low_freq_half, axis=1)
        models.append(avg_log_spec)
    
    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
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
    '''
    # 1. 对测试语音做VAD，得到测试段
    test_segments = VAD(testspeech, Fs)
    # 2. 把测试段转成测试模型
    test_models = segments_to_models(test_segments, Fs)
    if not test_models or not models:
        return np.array([]), []
    
    # 3. 统一模型维度（避免维度不一致）
    max_dim = max([m.shape[0] for m in models + test_models])
    def pad_model(model):
        if len(model) < max_dim:
            return np.pad(model, (0, max_dim - len(model)), mode='constant')
        return model[:max_dim]
    
    padded_models = [pad_model(m) for m in models]
    padded_test_models = [pad_model(m) for m in test_models]
    
    # 4. 计算余弦相似度（Y个模型 × K个测试段）
    sims = cosine_similarity(padded_models, padded_test_models)
    
    # 5. 为每个测试段匹配最相似的模型标签
    test_outputs = []
    for i in range(sims.shape[1]):
        max_sim_idx = np.argmax(sims[:, i])
        test_outputs.append(labels[max_sim_idx])
    
    return sims, test_outputs


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 生成模拟语音数据（用于测试）
    Fs = 16000  # 采样率16kHz
    t = np.linspace(0, 2, 2*Fs)  # 2秒的时间轴
    
    # 生成两个模拟语音段（不同频率的正弦波，模拟不同语音）
    # 模型1：100Hz正弦波（模拟"你好"）
    speech1 = 0.5 * np.sin(2 * np.pi * 100 * t[:int(1*Fs)])  # 1秒
    # 模型2：200Hz正弦波（模拟"再见"）
    speech2 = 0.5 * np.sin(2 * np.pi * 200 * t[int(1*Fs):])  # 1秒
    
    # 步骤1：为两个语音段生成模型
    seg1 = VAD(speech1, Fs)
    seg2 = VAD(speech2, Fs)
    model1 = segments_to_models(seg1, Fs)[0]
    model2 = segments_to_models(seg2, Fs)[0]
    models = [model1, model2]
    labels = ["你好", "再见"]
    
    # 步骤2：生成测试语音（混合"你好"+"再见"）
    test_speech = np.concatenate([
        0.5 * np.sin(2 * np.pi * 100 * np.linspace(0, 1, Fs)),  # 你好
        0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, Fs))   # 再见
    ])
    
    # 步骤3：语音识别
    sims, test_outputs = recognize_speech(test_speech, Fs, models, labels)
    
    # 输出结果
    print("余弦相似度矩阵（行：模型，列：测试段）：")
    print(sims)
    print("\n识别结果：")
    for i, label in enumerate(test_outputs):
        print(f"测试段{i+1}：{label}")
