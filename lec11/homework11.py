import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio (e.g., 'zh-CN' for Chinese, 'en-US' for English)
    
    @returns:
    text (str) - the recognized speech
    '''
    # 初始化识别器
    r = sr.Recognizer()
    
    # 读取音频文件
    with sr.AudioFile(filename) as source:
        # 降噪（可选但推荐）
        r.adjust_for_ambient_noise(source)
        # 读取音频数据
        audio_data = r.record(source)
    
    try:
        # 使用Google语音识别（需要联网）
        text = r.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "无法识别音频内容"
    except sr.RequestError as e:
        return f"语音识别服务请求失败: {e}"
    except Exception as e:
        return f"其他错误: {e}"

# 测试示例（请替换为你的音频文件路径）
if __name__ == "__main__":
    # 示例：识别中文wav文件
    audio_file = "test.wav"  # 替换成你的音频文件路径
    language_code = "zh-CN"  # 中文：zh-CN，英文：en-US，根据音频语言调整
    
    result = transcribe_wavefile(audio_file, language_code)
    print("识别结果：", result)
