import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    # 计算总帧数
    num_frames = 1 + int((len(waveform) - frame_length) / step)
    # 初始化帧矩阵
    frames = np.zeros((num_frames, frame_length))
    
    # 填充每帧数据
    for i in range(num_frames):
        start = i * step
        end = start + frame_length
        frames[i] = waveform[start:end]
    
    return frames

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    # 对每一行（每一帧）做FFT，然后取绝对值（幅度）
    fft_result = np.fft.fft(frames, axis=1)
    mstft = np.abs(fft_result)
    return mstft

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    mstft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
    # 计算mstft的最大值
    max_mstft = np.amax(mstft)
    # 替换小于0.001*max_mstft的值，避免对数计算出错
    mstft_clipped = np.maximum(mstft, 0.001 * max_mstft)
    
    # 转换为分贝（20*log10）
    spectrogram = 20 * np.log10(mstft_clipped)
    
    # 确保最小值不超过最大值减60（动态范围限制在60dB）
    max_spec = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_spec - 60)
    
    return spectrogram

# 测试代码
if __name__ == "__main__":
    # 生成测试用的波形（正弦波混合，模拟音频信号）
    fs = 16000  # 采样率
    duration = 1  # 时长1秒
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 混合440Hz和880Hz的正弦波
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    # 帧长和步长（示例值，可根据需要调整）
    frame_length = 512
    step = 256
    
    # 执行完整流程
    frames = waveform_to_frames(waveform, frame_length, step)
    print(f"Frames shape: {frames.shape}")
    
    mstft = frames_to_mstft(frames)
    print(f"MSTFT shape: {mstft.shape}")
    
    spectrogram = mstft_to_spectrogram(mstft)
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram min: {np.amin(spectrogram):.2f}, max: {np.amax(spectrogram):.2f}")
    print(f"Max - Min: {np.amax(spectrogram) - np.amin(spectrogram):.2f} (should be ≤60)")
