import numpy as np
import librosa

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
    # 补零保证所有帧长度一致
    pad_length = (len(speech) + frame_skip - 1) // frame_skip * frame_skip
    speech_padded = np.pad(speech, (0, pad_length - len(speech)), mode='constant')
    
    nframes = (len(speech_padded) - frame_length) // frame_skip + 1
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    # 加窗（汉明窗）
    window = np.hamming(frame_length)
    
    for i in range(nframes):
        # 提取当前帧
        start = i * frame_skip
        end = start + frame_length
        frame = speech_padded[start:end] * window
        
        # 计算LPC系数（使用自相关法）
        r = librosa.core.autocorrelate(frame, max_size=order)
        R = np.zeros((order, order))
        for j in range(order):
            R[j, :] = r[j:j+order]
        
        # 求解Yule-Walker方程
        try:
            a = np.linalg.solve(R, r[1:order+1])
        except np.linalg.LinAlgError:
            a = np.zeros(order)
        
        # LPC系数：A[0]=1，后续为-a
        A[i, 0] = 1.0
        A[i, 1:] = -a
        
        # 计算残差（激励信号）
        # 线性预测：frame ≈ A[i] * 预测信号 → 残差 = frame - 预测值
        predicted = np.convolve(frame, A[i], mode='same')
        excitation[i] = frame - predicted
    
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (nframes,frame_length) - excitation signal frames
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes, frame_length = e.shape
    order = A.shape[1] - 1
    
    # 初始化合成语音数组
    total_length = (nframes - 1) * frame_skip + frame_length
    synthesis = np.zeros(total_length)
    
    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_length
        
        # 对当前帧激励信号做逆滤波（LPC合成）
        frame_synth = np.convolve(e[i], np.flip(A[i]), mode='same')
        
        # 重叠相加法合成
        synthesis[start:end] += frame_synth
    
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
    nframes, frame_length = excitation.shape
    gain = np.zeros(nframes)
    e_robot = np.zeros((nframes - 1) * frame_skip + frame_length)
    
    for i in range(nframes):
        # 计算每帧的增益（能量的平方根）
        gain[i] = np.sqrt(np.sum(excitation[i] ** 2) / frame_length)
        
        # 生成机器人语音的激励信号（周期性方波）
        start = i * frame_skip
        end = start + frame_length
        t = np.arange(frame_length)
        periodic = gain[i] * np.sign(np.sin(2 * np.pi * t / T0))
        
        # 只保留每帧最后frame_skip个有效样本
        valid_start = frame_length - frame_skip
        e_robot[start + valid_start : end] = periodic[valid_start:]
    
    # 截断到nframes*frame_skip长度
    e_robot = e_robot[:nframes * frame_skip]
    return gain, e_robot

# 测试代码（运行入口）
if __name__ == "__main__":
    # 1. 加载测试语音（librosa内置示例音频）
    y, sr = librosa.load(librosa.ex('trumpet'), duration=2.0)  # 取前2秒
    frame_length = 512  # 帧长（样本数）
    frame_skip = 128    # 帧移（样本数）
    order = 12          # LPC阶数
    T0 = 80             # 基音周期（样本数，需根据采样率调整）
    
    # 2. LPC分析
    print("开始LPC分析...")
    A, excitation = lpc(y, frame_length, frame_skip, order)
    print(f"LPC分析完成：帧数={A.shape[0]}, LPC阶数={order}")
    
    # 3. 生成机器人语音激励信号
    print("生成机器人语音激励信号...")
    gain, e_robot = robot_voice(excitation, T0, frame_skip)
    
    # 4. LPC合成（机器人语音）
    print("开始语音合成...")
    # 将e_robot重塑为帧格式（匹配synthesize输入）
    e_robot_frames = np.zeros_like(excitation)
    for i in range(len(e_robot_frames)):
        start = i * frame_skip
        end = start + frame_skip
        if end <= len(e_robot):
            e_robot_frames[i, -frame_skip:] = e_robot[start:end]
    
    synth_voice = synthesize(e_robot_frames, A, frame_skip)
    
    # 5. 保存合成后的语音
    import soundfile as sf
    sf.write("robot_voice.wav", synth_voice, sr)
    print("合成完成！结果已保存为 robot_voice.wav")
    
    # 可选：播放合成语音（需安装sounddevice）
    try:
        import sounddevice as sd
        print("正在播放合成的机器人语音...")
        sd.play(synth_voice, sr)
        sd.wait()
    except ImportError:
        print("未安装sounddevice，跳过播放步骤（可执行 pip install sounddevice 安装）")
