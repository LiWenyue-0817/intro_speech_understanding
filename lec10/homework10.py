import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft
from librosa.effects import preemphasis
from librosa.feature import rms
from librosa.util import frame


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
    # 1. 预加重（Pre-emphasis）
    pre_emphasized = preemphasis(waveform, coef=0.97)

    # 2. 计算频谱图（4ms帧长，2ms步长）
    frame_len = int(0.004 * Fs)  # 4ms对应的采样点数
    hop_len = int(0.002 * Fs)    # 2ms对应的采样点数
    f, t, Zxx = stft(
        pre_emphasized,
        fs=Fs,
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        return_onesided=True
    )
    spectrogram = np.abs(Zxx)  # 取幅度谱

    # 3. 保留低频半部分（非混叠部分）
    n_freqs = spectrogram.shape[0]
    features = spectrogram[:n_freqs//2, :].T  # (NFRAMES, NFEATS)

    # 4. 计算VAD（25ms窗口，10ms步长）
    vad_window = int(0.025 * Fs)  # 25ms窗口
    vad_hop = int(0.010 * Fs)     # 10ms步长
    rms_vals = rms(
        y=pre_emphasized,
        frame_length=vad_window,
        hop_length=vad_hop
    ).squeeze()
    vad = (rms_vals > np.mean(rms_vals) * 0.1).astype(int)  # 简单能量阈值判断静音/非静音

    # 5. 为非静音段分配不同标签，并重复5次
    # 先将VAD对齐到features的帧数（features是2ms步长，VAD是10ms步长 → 1个VAD帧对应5个feature帧）
    vad_repeated = np.repeat(vad, 5)  # 每个VAD标签重复5次
    # 截断/补零到和features帧数一致
    if len(vad_repeated) < len(features):
        vad_repeated = np.pad(vad_repeated, (0, len(features) - len(vad_repeated)))
    else:
        vad_repeated = vad_repeated[:len(features)]

    # 标记非静音段
    labels = np.zeros_like(vad_repeated, dtype=int)
    current_label = 1
    in_speech = False
    for i in range(len(vad_repeated)):
        if vad_repeated[i] == 1 and not in_speech:
            in_speech = True
            current_label += 1
        elif vad_repeated[i] == 0:
            in_speech = False
        labels[i] = current_label if in_speech else 0

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels (integers)
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    # 1. 转换为PyTorch张量
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 2. 构建模型（LayerNorm + Linear）
    n_feats = features.shape[1]
    n_classes = 1 + np.max(labels)
    model = nn.Sequential(
        nn.LayerNorm(n_feats),
        nn.Linear(n_feats, n_classes)
    )

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 训练循环
    lossvalues = np.zeros(iterations)
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(features_tensor)  # 前向传播
        loss = criterion(outputs, labels_tensor)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        lossvalues[i] = loss.item()  # 记录损失

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    # 转换为张量
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # 转换为概率
    
    # 转为numpy数组
    return probabilities.detach().numpy()


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 生成模拟音频数据（1秒，采样率16kHz）
    Fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(Fs * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波（模拟语音）
    waveform = np.concatenate([
        waveform, 
        np.zeros(int(0.2 * Fs)),  # 静音段
        0.4 * np.sin(2 * np.pi * 880 * t[:int(0.8 * Fs)])  # 另一语音段
    ])

    # 1. 提取特征和标签
    features, labels = get_features(waveform, Fs)
    print(f"特征维度: {features.shape}")
    print(f"标签维度: {labels.shape}")
    print(f"唯一标签: {np.unique(labels)}")

    # 2. 训练神经网络
    iterations = 100
    model, loss_values = train_neuralnet(features, labels, iterations)
    print(f"\n训练完成，最后10轮损失: {loss_values[-10:]}")

    # 3. 测试模型
    probabilities = test_neuralnet(model, features)
    print(f"\n预测概率维度: {probabilities.shape}")
    print(f"第一帧概率分布: {probabilities[0]}")
