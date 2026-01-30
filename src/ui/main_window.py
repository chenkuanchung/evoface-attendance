import sys
import os
import cv2
import numpy as np
import yaml
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QDialog, QLineEdit, QFormLayout, 
                             QDialogButtonBox, QMessageBox)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer, QMutex
from PySide6.QtGui import QImage, QPixmap, QFont

# 匯入專案核心模組
from src.core.detector import FaceDetector
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB
from src.utils.voice import speak_success

# === 手動密碼驗證對話框 (完全保留) ===
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

# === [核心修改] 全能辨識工作執行緒 (取代原本的 VideoThread) ===
class RecognitionWorker(QThread):
    # Signal: (顯示用的QImage, 辨識結果與狀態包, 原始frame用於存檔)
    result_signal = Signal(QImage, dict, np.ndarray)

    def __init__(self, config, detector, recognizer):
        super().__init__()
        self.detector = detector
        self.recognizer = recognizer
        self.camera_index = config.get('system', {}).get('camera_index', 0)
        self.running = True
        self.mutex = QMutex()

    def stop(self):
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        self.wait()

    def run(self):
        # 1. 在子執行緒開啟攝影機
        cap = cv2.VideoCapture(self.camera_index)
        # 設定解析度 (降低解析度可提升 FPS，這裡維持 640x480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            self.mutex.lock()
            if not self.running:
                self.mutex.unlock()
                break
            self.mutex.unlock()

            ret, frame = cap.read()
            if not ret:
                continue

            # 2. 影像前處理 (新增：鏡像翻轉，體驗更好)
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # 3. 執行人臉偵測 (MediaPipe)
            status, res = self.detector.process(frame)
            
            # 準備回傳的數據包
            result_data = {
                "status": status,
                "res": res,
                "recognition": None # 預設無辨識結果
            }

            # 4. 如果偵測到活體人臉，直接在這裡進行辨識 (InsightFace)
            # 這原本是在主執行緒做的，現在移到這裡就不會卡 UI 了
            if status == "SUCCESS" and res['is_live']:
                face_img = res['face_img']
                if face_img is not None:
                    # 呼叫辨識核心 (這是最耗時的步驟)
                    emp_id, score, evolve, details, live_feat = self.recognizer.identify(face_img)
                    
                    result_data["recognition"] = {
                        "emp_id": emp_id,
                        "score": score,
                        "evolve": evolve,
                        "details": details,
                        "live_feat": live_feat
                    }

            # 5. 在背景先畫好框 (減輕 UI 負擔)
            if status == "SUCCESS":
                bbox = res['bbox']
                is_live = res['is_live']
                # 綠色=活體, 橘色=偽造/檢測中
                color = (0, 255, 0) if is_live else (0, 165, 255)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # 如果有辨識出名字，順便寫上去
                rec_res = result_data.get("recognition")
                if rec_res and rec_res["emp_id"]:
                    text = f"{rec_res['emp_id']} ({rec_res['score']:.2f})"
                    cv2.putText(display_frame, text, (bbox[0], bbox[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif status == "FACE_TOO_SMALL":
                bbox = res['bbox']
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            # 6. 轉換格式給 Qt 顯示
            rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 7. 發送結果回主執行緒
            # 注意：frame (原始無框圖) 也傳回去，因為存檔需要乾淨的照片
            self.result_signal.emit(qt_image, result_data, frame)

        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        with open("config.yaml", 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.setWindowTitle(self.config['system']['app_name'])
        self.resize(800, 600) 

        # 初始化核心組件
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.db = AttendanceDB()
        
        # UI 初始化
        self.init_ui()
        
        # 時鐘 Timer
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()
        
        # 啟動全能辨識 Worker
        self.success_cooldown = False 
        self.start_worker()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 20, 0, 20)

        # 1. 頂部時鐘
        self.clock_label = QLabel("00:00:00")
        self.clock_label.setFont(QFont("Consolas", 24, QFont.Bold))
        self.clock_label.setAlignment(Qt.AlignCenter)
        self.clock_label.setStyleSheet("color: #2C3E50; margin-bottom: 10px;")
        layout.addWidget(self.clock_label)

        # 2. 影像預覽區
        self.video_label = QLabel("正在啟動系統核心...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 4px solid #333; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # 3. 狀態文字 (Overlay)
        self.status_label = QLabel(self.video_label)
        self.status_label.setGeometry(10, 380, 620, 90)
        self.status_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: rgba(0, 0, 0, 160); color: white; border-radius: 5px; padding: 5px;")
        self.status_label.setWordWrap(True)
        self.status_label.setText("系統初始化中...")
        self.status_label.show()

    def start_worker(self):
        # 建立並啟動子執行緒
        self.worker = RecognitionWorker(self.config, self.detector, self.recognizer)
        self.worker.result_signal.connect(self.update_ui)
        self.worker.start()

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(now.strftime("%H:%M:%S"))

    @Slot(QImage, dict, np.ndarray)
    def update_ui(self, qt_image, result_data, raw_frame):
        """
        主執行緒 Slot：只負責更新畫面與處理打卡邏輯，不進行運算
        """
        # 1. 顯示畫面 (這會非常流暢)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        # 2. 如果處於「打卡成功冷卻期」，就不更新狀態文字，讓使用者看清楚成功訊息
        if self.success_cooldown:
            return

        # 3. 解析 Worker 傳來的結果
        status = result_data['status']
        res = result_data['res']
        rec_data = result_data.get('recognition')

        if status == "SUCCESS":
            if not res['is_live']:
                # 活體檢測失敗/進行中
                score = res.get('texture_score', 0)
                self.status_label.setText(f"活體檢測中... {int(score*100)}%")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #F39C12; border-radius: 5px;")
            
            elif rec_data:
                # === 活體通過 + 辨識完成 ===
                self.handle_recognition_result(rec_data, raw_frame)
            else:
                # 活體通過，但辨識無結果 (可能是 None)
                self.status_label.setText("身分識別中...")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #2ECC71; border-radius: 5px;")

        elif status == "FACE_TOO_SMALL":
            self.status_label.setText("請靠近鏡頭 (人臉過小)")
            self.status_label.setStyleSheet("color: red;")
            
        elif status == "MULTIPLE_FACES":
            self.status_label.setText("偵測到多人，請單人打卡")
            self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #E74C3C; border-radius: 5px;")
            
        else:
            self.status_label.setText("請正對攝影機打卡")
            self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: white; border-radius: 5px;")

    def handle_recognition_result(self, rec_data, raw_frame):
        emp_id = rec_data['emp_id']
        score = rec_data['score']
        details = rec_data['details']
        
        # (A) 猶豫機制檢查
        if details.get("warning") and details.get("reason") == "ambiguous_gap":
            self.status_label.setText(f"⚠️ 系統猶豫中... 請轉動頭部或靠近")
            self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: orange; border-radius: 5px;")
            return

        # (B) 辨識成功
        if emp_id:
            # 存檔路徑
            os.makedirs("data/logs", exist_ok=True)
            photo_name = f"data/logs/{emp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # 寫入資料庫
            success, message = self.recognizer.process_attendance(
                emp_id, score, rec_data['evolve'], rec_data['live_feat'], photo_name, details
            )

            if success:
                # 儲存照片
                cv2.imwrite(photo_name, raw_frame)
                # 顯示成功訊息並進入冷卻
                self.show_success_feedback(emp_id, details)
            else:
                # 重複打卡 (Debounce)
                self.status_label.setText(f"ℹ️ {message}")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #3498DB; border-radius: 5px;")

        # (C) 辨識失敗
        else:
            # 檢查是否因為冷卻期而導致的低分重複掃描
            candidate_id = details.get("candidate_id") # 從 details 拿出最像的候選人
            if candidate_id:
                is_cd, remaining = self.db.is_cooling_down(candidate_id)
                if is_cd:
                    # 如果該候選人還在冷卻中，即使分數低，也只要顯示冷卻訊息，不要跳驗證視窗
                    self.status_label.setText(f"ℹ️ 打卡過於頻繁，請於 {remaining} 秒後再試。")
                    self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #3498DB; border-radius: 5px;")
                    return # 直接結束，不跳出對話框

            # 輔助驗證：分數不低但沒過門檻 (且不在冷卻期)
            if score > 0.4:
                self.handle_manual_login(score)
            else:
                self.status_label.setText(f"辨識失敗 (信心度不足: {score:.2f})")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: #E74C3C; border-radius: 5px;")

    def show_success_feedback(self, emp_id, details):
        """顯示打卡成功訊息"""
        current_time = datetime.now().strftime("%H:%M:%S")
        is_warning = details.get('warning', False)
        
        if is_warning:
            display_text = f"✅ 打卡成功\n⚠️ 證件照差異過大\nID: {emp_id}"
            text_color = "#F1C40F"
        else:
            display_text = f"✅ 打卡成功\nID: {emp_id}\n時間: {current_time}"
            text_color = "#2ECC71"
        
        self.status_label.setText(display_text)
        self.status_label.setStyleSheet(f"background-color: rgba(0,0,0,190); color: {text_color}; border-radius: 5px; font-weight: bold;")
        
        speak_success()
        
        # 啟動冷卻 (3秒後恢復正常掃描顯示)
        self.success_cooldown = True
        QTimer.singleShot(3000, self.end_cooldown)

    def end_cooldown(self):
        self.success_cooldown = False
        if hasattr(self, 'worker'):
            self.worker.detector.reset_liveness() 
            
        # 重置顯示
        self.status_label.setText("請正對攝影機打卡")
        self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: white; border-radius: 5px;")

    def handle_manual_login(self, score):
        # 暫時停止接收 Worker 更新
        self.success_cooldown = True 
        
        dialog = ManualLoginDialog(self)
        if dialog.exec() == QDialog.Accepted:
            uid = dialog.id_input.text().strip()
            pwd = dialog.pwd_input.text().strip()
            
            if self.db.verify_password(uid, pwd):
                self.db.add_attendance_log(uid, 1.0, "MANUAL_PWD", {'base_score': 0, 'dynamic_score': 0})
                self.show_success_feedback(uid, {})
            else:
                self.status_label.setText("❌ 密碼驗證失敗")
                self.status_label.setStyleSheet("background-color: rgba(0,0,0,160); color: red;")
                QTimer.singleShot(2000, self.end_cooldown)
        else:
            self.status_label.setText("驗證取消")
            QTimer.singleShot(1000, self.end_cooldown)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())