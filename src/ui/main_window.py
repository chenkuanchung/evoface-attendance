import sys
import os
import cv2
import numpy as np
import yaml
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QListWidget, QFrame)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# 匯入專案核心模組
from src.core.detector import FaceDetector
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB
from src.utils.voice import speak_success

class VideoThread(QThread):
    """
    影像處理執行緒：負責擷取影像並執行 FaceDetector (含活體偵測)
    """
    change_pixmap_signal = Signal(np.ndarray, dict)

    def __init__(self, config):
        super().__init__()
        self.detector = FaceDetector()
        self.camera_index = config.get('system', {}).get('camera_index', 0)
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 執行偵測邏輯
                status, res = self.detector.process(frame)
                # 將影像與偵測結果傳回主視窗
                self.change_pixmap_signal.emit(frame, {"status": status, "res": res})
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 載入設定
        with open("config.yaml", 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.setWindowTitle(self.config['system']['app_name']) #
        self.setMinimumSize(1000, 700)

        # 初始化核心組件
        self.recognizer = FaceRecognizer()
        self.db = AttendanceDB()
        
        # 狀態控制
        self.is_processing = False # 避免重複辨識同一次打卡

        self.init_ui()
        
        # 啟動執行緒
        self.video_thread = VideoThread(self.config)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def init_ui(self):
        """建構 UI 佈局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左側：影像預覽區 ---
        left_layout = QVBoxLayout()
        self.video_label = QLabel("正在啟動攝影機...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # 狀態提示文字
        self.status_label = QLabel("請正對攝影機")
        self.status_label.setFont(QFont("Microsoft JhengHei", 18, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #555;")
        left_layout.addWidget(self.status_label)
        
        main_layout.addLayout(left_layout, stretch=2)

        # --- 右側：歷史紀錄區 ---
        right_layout = QVBoxLayout()
        log_title = QLabel("今日打卡清單") #
        log_title.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        right_layout.addWidget(log_title)

        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")
        right_layout.addWidget(self.log_list)
        
        main_layout.addLayout(right_layout, stretch=1)
        
        # 初始載入今日紀錄
        self.refresh_logs()

    @Slot(np.ndarray, dict)
    def update_image(self, frame, data):
        """處理每幀影像更新與辨識觸發"""
        status = data['status']
        res = data['res']
        h, w, _ = frame.shape

        if status == "SUCCESS":
            bbox = res['bbox']
            is_live = res['is_live']
            score = res['texture_score']

            # 繪製 UI 框：橘色代表掃描中，綠色代表活體通過
            color = (0, 255, 0) if is_live else (0, 165, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
    def update_image(self, frame, data):
        """處理每幀影像更新與辨識觸發"""
        status = data['status']
        res = data['res']
        h, w, _ = frame.shape

        if status == "SUCCESS":
            bbox = res['bbox']
            is_live = res['is_live']
            score = res['texture_score']

            # 繪製 UI 框：橘色代表掃描中，綠色代表活體通過
            color = (0, 255, 0) if is_live else (0, 165, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # --- 如果正在處理結果，就不要更新文字，讓「打卡成功」留著 ---
            if not self.is_processing:
                if not is_live:
                    self.status_label.setText(f"辨識中... {int(score*100)}%")
                    self.status_label.setStyleSheet("color: #FFA500;")
                else:
                    self.status_label.setText("活體檢測通過，正在比對身分...")
                    self.status_label.setStyleSheet("color: #008000;")
                    
                    # 觸發辨識邏輯
                    if res['face_img'] is not None:
                        self.perform_recognition(res['face_img'])

        elif status == "MULTIPLE_FACES":
            if not self.is_processing:
                self.status_label.setText("警示：偵測到多人，請單人打卡")
                self.status_label.setStyleSheet("color: red;")
        else:
            if not self.is_processing:
                self.status_label.setText("等待人臉入鏡...")
                self.status_label.setStyleSheet("color: #555;")

        # 轉換影像格式顯示在 QLabel
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def perform_recognition(self, face_img):
        """執行身分比對與考勤儲存"""
        self.is_processing = True
        
        # 1. 執行 1:N 比對
        emp_id, score, evolve, details, live_feat = self.recognizer.identify(face_img)
        print(f"🔍 [Debug] 比對結果: ID={emp_id}, 分數={score:.4f}, 詳細={details}")
        
        if emp_id:
            photo_name = f"data/logs/{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            success, message = self.recognizer.process_attendance(
                emp_id, score, evolve, live_feat, photo_name, details
            )
            if success:
                self.status_label.setText(f"打卡成功：{emp_id}")
                self.refresh_logs()
                speak_success()# 打卡成功語音
                # 2. 儲存打卡紀錄與處理特徵演進
                os.makedirs("data/logs", exist_ok=True)
                cv2.imwrite(photo_name, face_img)
            else:
                # 可能是觸發了 5 分鐘去抖動機制
                self.status_label.setText(message)
        else:
            # 如果失敗，顯示分數讓你知道差多少
            msg = f"辨識失敗 (分數: {score:.2f})"
            self.status_label.setText(msg)
            print(f"❌ {msg}")
        
        # 3. 重置偵測器狀態，準備下一次辨識
        QTimer.singleShot(5000, self.reset_recognition) # 5秒後恢復辨識功能

    def reset_recognition(self):
        """重置狀態供下一位員工打卡"""
        self.video_thread.detector.reset_liveness()
        self.is_processing = False

    def refresh_logs(self):
        """從資料庫抓取最新紀錄更新 UI"""
        self.log_list.clear()
        logs = self.db.get_recent_logs(limit=15)
        for log in logs:
            self.log_list.addItem(f"[{log['time']}] {log['name']} (得分: {log['score']})")

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())