import sys
import os
import cv2
import numpy as np
import yaml
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QDialog, QLineEdit, QFormLayout, 
                             QDialogButtonBox, QMessageBox)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# 匯入專案核心模組
from src.core.detector import FaceDetector
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB
from src.utils.voice import speak_success

# === 手動密碼驗證對話框 ===
class ManualLoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("輔助驗證 (信心度不足)")
        self.setFixedSize(300, 160)
        self.setStyleSheet("font-size: 14px;")
        
        layout = QFormLayout(self)
        
        self.info_label = QLabel("系統無法確認您的身分，\n請輸入密碼進行驗證。")
        self.info_label.setStyleSheet("color: #E67E22; font-weight: bold; margin-bottom: 10px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addRow(self.info_label)
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("請輸入員工 ID")
        
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd_input.setPlaceholderText("請輸入密碼")
        
        layout.addRow("員工 ID:", self.id_input)
        layout.addRow("密碼:", self.pwd_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

class VideoThread(QThread):
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
                status, res = self.detector.process(frame)
                self.change_pixmap_signal.emit(frame, {"status": status, "res": res})
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        with open("config.yaml", 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.setWindowTitle(self.config['system']['app_name'])
        # 高度可以縮小了，因為文字疊在影像上
        self.resize(800, 600) 

        self.recognizer = FaceRecognizer()
        self.db = AttendanceDB()
        self.is_processing = False 

        self.init_ui()
        
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()
        
        self.video_thread = VideoThread(self.config)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 20, 0, 20) # 上下留白

        # 1. 頂部時鐘
        self.clock_label = QLabel("00:00:00")
        self.clock_label.setFont(QFont("Consolas", 24, QFont.Bold))
        self.clock_label.setAlignment(Qt.AlignCenter)
        self.clock_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        layout.addWidget(self.clock_label)

        # 2. 影像預覽區 (Container)
        self.video_label = QLabel("正在啟動攝影機...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 4px solid #333; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # === 3. 狀態文字 (Overlay HUD) ===
        # 將 status_label 的父物件設為 video_label，這樣它就會「黏」在影像上
        self.status_label = QLabel(self.video_label)
        # 設定位置：x=10, y=380 (底部), w=620, h=90
        self.status_label.setGeometry(10, 380, 620, 90)
        self.status_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        # 設定半透明黑底 + 白字
        self.status_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160); 
            color: white; 
            border-radius: 5px;
            padding: 5px;
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setText("請正對攝影機打卡")
        self.status_label.show() # 必須手動 show 因為它是子元件

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S"))

    @Slot(np.ndarray, dict)
    def update_image(self, frame, data):
        status = data['status']
        res = data['res']
        h, w, _ = frame.shape

        if status == "SUCCESS":
            bbox = res['bbox']
            is_live = res['is_live']
            score = res['texture_score']

            color = (0, 255, 0) if is_live else (0, 165, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if not self.is_processing:
                if not is_live:
                    self.status_label.setText(f"活體檢測中... {int(score*100)}%")
                    # 活體檢測中顯示橘字 (背景維持半透明黑)
                    self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #F39C12; border-radius: 5px;")
                else:
                    self.status_label.setText("檢測通過，正在識別身份...")
                    self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #2ECC71; border-radius: 5px;")
                    
                    if res['face_img'] is not None:
                        self.perform_recognition(res['face_img'])
                        
        # === 處理人臉過小的情況 ===
        elif status == "FACE_TOO_SMALL":
            bbox = res['bbox']
            # 顯示紅色框
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            if not self.is_processing:
                self.status_label.setText("請靠近鏡頭 (人臉過小)")
                self.status_label.setStyleSheet("color: red;")

        elif status == "MULTIPLE_FACES":
            if not self.is_processing:
                self.status_label.setText("偵測到多人，請單人打卡")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #E74C3C; border-radius: 5px;")
        else:
            if not self.is_processing:
                self.status_label.setText("請正對攝影機打卡")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: white; border-radius: 5px;")

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def perform_recognition(self, face_img):
        self.is_processing = True
        
        emp_id, score, evolve, details, live_feat = self.recognizer.identify(face_img)
        print(f"🔍 [Debug] Match Result: ID={emp_id}, Score={score:.4f}")
        
        if emp_id:
            # === 辨識成功 ===
            photo_name = f"data/logs/{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            success, message = self.recognizer.process_attendance(
                emp_id, score, evolve, live_feat, photo_name, details
            )
            
            if success:
                # [修改] 改用 helper function 顯示成功，方便手動驗證也能共用
                self.show_success_feedback(emp_id, details)
                os.makedirs("data/logs", exist_ok=True)
                cv2.imwrite(photo_name, face_img)
            else:
                self.status_label.setText(message)
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #E67E22; border-radius: 5px;")
                QTimer.singleShot(3000, self.reset_recognition)
                
        else:
            # === 辨識失敗 ===
            # 輔助驗證：如果分數在模糊地帶 (> 0.4)，啟動密碼驗證
            if score > 0.4:
                self.handle_manual_login(score)
            else:
                msg = f"辨識失敗 (信心度不足: {score:.2f})"
                self.status_label.setText(msg)
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #E74C3C; border-radius: 5px;")
                QTimer.singleShot(3000, self.reset_recognition)

    def handle_manual_login(self, score):
        """處理手動密碼驗證流程"""
        self.status_label.setText(f"信心度 {score:.2f} (不足)\n請進行輔助驗證...")
        
        # 彈出對話框 (會暫停 UI 互動，但 VideoThread 仍會跑，造成畫面凍結是正常的)
        dialog = ManualLoginDialog(self)
        if dialog.exec() == QDialog.Accepted:
            uid = dialog.id_input.text().strip()
            pwd = dialog.pwd_input.text().strip()
            
            # 呼叫資料庫驗證密碼
            if self.db.verify_password(uid, pwd):
                # 驗證成功：手動寫入 Log (標記 photo_path 為 MANUAL_PWD)
                self.db.add_attendance_log(uid, 1.0, "MANUAL_PWD", {'base_score': 0, 'dynamic_score': 0})
                
                # 顯示成功 (使用空的 details)
                self.show_success_feedback(uid, {})
            else:
                self.status_label.setText("❌ 密碼驗證失敗，請重試")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: red; border-radius: 5px;")
                QTimer.singleShot(3000, self.reset_recognition)
        else:
            # 使用者按取消
            self.status_label.setText("驗證取消")
            QTimer.singleShot(3000, self.reset_recognition)

    def show_success_feedback(self, emp_id, details):
        """顯示打卡成功訊息 (抽離出來共用)"""
        current_time = datetime.now().strftime("%H:%M:%S")
        is_warning = details.get('warning', False)
        
        if is_warning:
            display_text = f"✅ 打卡成功\n⚠️ 證件照差異過大，請通知管理員\nID: {emp_id}"
            text_color = "#F1C40F" 
        else:
            display_text = f"✅ 打卡成功\nID: {emp_id}\n時間: {current_time}"
            text_color = "#2ECC71"
        
        self.status_label.setText(display_text)
        self.status_label.setStyleSheet(f"background-color: rgba(0,0,0,180); color: {text_color}; border-radius: 5px; font-weight: bold;")
        
        speak_success()
        QTimer.singleShot(3000, self.reset_recognition)

    def reset_recognition(self):
        self.video_thread.detector.reset_liveness()
        self.is_processing = False
        self.status_label.setText("請正對攝影機打卡")
        self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: white; border-radius: 5px;")

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())