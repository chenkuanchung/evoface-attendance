# src/utils/voice.py
import pyttsx3
import threading

def speak_success():
    """發出打卡成功的語音提示"""
    def _run():
        try:
            engine = pyttsx3.init()
            # 語速設定 (可選，預設通常剛好)
            # engine.setProperty('rate', 150) 
            engine.say("打卡成功")  # 這裡改為固定台詞
            engine.runAndWait()
        except Exception as e:
            print(f"語音播放失敗: {e}")
    
    # 使用執行緒播放，避免卡住畫面
    threading.Thread(target=_run, daemon=True).start()