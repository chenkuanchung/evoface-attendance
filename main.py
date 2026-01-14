import sys
import os
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def setup_environment():
    """
    確保系統執行所需的必要目錄結構已存在
    """
    required_dirs = [
        "data/faces",   # 存放員工註冊照片
        "data/logs",    # 存放打卡現場照片
        "models/liveness" # 存放活體偵測模型
    ]
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"已建立目錄: {d}")

def main():
    # 1. 確保必要的環境目錄已建立
    setup_environment()

    # 2. 初始化 PySide6 應用程式
    # 設定高 DPI 支援，確保在 4K 螢幕上不會模糊
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    
    # 3. 設定全域樣式 (可選，增加專業感)
    app.setStyle("Fusion") 

    # 4. 實例化並顯示主視窗
    try:
        window = MainWindow()
        window.show()
        
        # 5. 進入程式主迴圈
        sys.exit(app.exec())
    except Exception as e:
        print(f"系統啟動失敗: {e}")

if __name__ == "__main__":
    main()