import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename (MP3).
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it (e.g., 'en' for English, 'zh-CN' for Chinese)
    filename (str) - the filename in which it should be saved (without .mp3 suffix)
    '''
    # 初始化gTTS对象
    tts = gtts.gTTS(text=text, lang=lang, slow=False)
    # 保存为MP3文件
    mp3_filename = f"{filename}.mp3"
    tts.save(mp3_filename)
    print(f"Synthesized MP3 saved to: {mp3_filename}")

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    # 校验输入列表长度一致
    if not (len(texts) == len(languages) == len(filenames)):
        raise ValueError("texts, languages, filenames must have the same length")
    
    recognized_texts = []
    recognizer = sr.Recognizer()  # 初始化识别器

    for text, lang, fname in zip(texts, languages, filenames):
        # 1. 合成MP3
        synthesize(text, lang, fname)
        mp3_path = f"{fname}.mp3"
        wav_path = f"{fname}.wav"

        # 2. 转换MP3到WAV（pydub需要ffmpeg支持）
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            print(f"Converted to WAV: {wav_path}")
        except Exception as e:
            print(f"Failed to convert {mp3_path} to WAV: {e}")
            recognized_texts.append(None)
            continue

        # 3. 语音识别（使用Google Web Speech API）
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)  # 读取音频文件
                # 识别（指定语言，与合成语言一致）
                recognized = recognizer.recognize_google(audio_data, language=lang)
                recognized_texts.append(recognized)
                print(f"Recognized text: {recognized}")
        except sr.UnknownValueError:
            print(f"Could not understand audio for {fname}")
            recognized_texts.append(None)
        except sr.RequestError as e:
            print(f"Could not request results from Google API: {e}")
            recognized_texts.append(None)
        except Exception as e:
            print(f"Recognition error for {fname}: {e}")
            recognized_texts.append(None)

    return recognized_texts

# 测试示例
if __name__ == "__main__":
    # 测试用例：中英文混合
    test_texts = ["Hello world", "你好，世界"]
    test_langs = ["en", "zh-CN"]
    test_filenames = ["english_hello", "chinese_hello"]

    # 执行合成+识别流程
    results = make_a_corpus(test_texts, test_langs, test_filenames)
    print("\nFinal recognition results:")
    for fname, res in zip(test_filenames, results):
        print(f"{fname}: {res}")
