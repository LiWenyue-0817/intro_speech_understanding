import numpy as np

def minimum_Fs(f):
    '''
    Find the lowest sampling frequency that would avoid aliasing for a pure tone at f Hz.
    
    @param:
    f (scalar): frequency in Hz (cycles/second)
    
    @result:
    Fs (scalar): the lowest sampling frequency (samples/second) that would
    not cause aliasing at a tone of f Hz.
    '''
    # 根据奈奎斯特采样定理，最低采样频率是信号频率的2倍
    Fs = 2 * f  
    return Fs

def omega(f, Fs):
    '''
    Find the radial frequency (omega) that matches a given f and Fs.
    
    @param:
    f (scalar): frequency in Hz (cycles/second)
    Fs (scalar): sampling frequency in samples/second
    
    @result:
    omega (scalar): radial frequency in radians/sample
    '''
    # 径向频率计算公式：ω = 2πf / Fs
    omega = 2 * np.pi * f / Fs  
    return omega

def pure_tone(omega, N):
    '''
    Create a pure tone of N samples at omega radians/sample.
    
    @param:
    omega (scalar): radial frequency, radians/sample
    N (scalar): duration of the tone, in samples
    
    @result:
    x (array): N samples from the signal cos(omega*n)
    '''
    # 生成0到N-1的采样点，计算余弦信号
    n = np.arange(N)  
    x = np.cos(omega * n)  
    return x

# 测试代码（运行验证）
if __name__ == "__main__":
    # 1. 测试最小采样频率
    f_signal = 100  # 信号频率100Hz
    min_Fs = minimum_Fs(f_signal)
    print(f"信号频率{f_signal}Hz时，避免混叠的最小采样频率：{min_Fs}Hz")
    
    # 2. 测试径向频率计算
    Fs_used = 250  # 实际使用的采样频率250Hz
    omega_val = omega(f_signal, Fs_used)
    print(f"采样频率{Fs_used}Hz时，径向频率：{omega_val:.4f} 弧度/采样点")
    
    # 3. 生成纯音信号并输出前10个采样点
    N_samples = 100  # 采样点数
    tone_signal = pure_tone(omega_val, N_samples)
    print(f"\n纯音信号前10个采样点：")
    print(tone_signal[:10])
