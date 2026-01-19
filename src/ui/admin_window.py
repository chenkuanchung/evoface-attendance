import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
                             QLineEdit, QFileDialog, QMessageBox, QGroupBox, QFormLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont

# 引用核心模組
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB

class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EvoFace - 員工資料管理後台")
        self.resize(900, 600)
        
        # 初始化核心
        self.db = AttendanceDB()
        self.recognizer = FaceRecognizer()
        
        # 暫存變數
        self.current_feature = None    # 暫存特徵向量 (給資料庫用)
        self.current_face_img = None   # 暫存裁切後的人臉照片 (給存檔用)
        
        self.init_ui()
        self.refresh_employee_list()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # === 左側：員工列表 ===
        left_panel = QGroupBox("現有員工名單")
        left_layout = QVBoxLayout()
        
        self.emp_list = QListWidget()
        self.emp_list.setStyleSheet("font-size: 14px;")
        left_layout.addWidget(self.emp_list)
        
        self.btn_delete = QPushButton("刪除選取員工")
        self.btn_delete.setStyleSheet("background-color: #ffcccc; color: red; padding: 8px;")
        self.btn_delete.clicked.connect(self.delete_employee)
        left_layout.addWidget(self.btn_delete)
        
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, stretch=1)
        
        # === 右側：新增註冊區 ===
        right_panel = QGroupBox("新增/註冊員工")
        right_layout = QVBoxLayout()
        
        # 1. 照片預覽
        self.lbl_preview = QLabel("請上傳照片")
        self.lbl_preview.setFixedSize(200, 200)
        self.lbl_preview.setStyleSheet("border: 2px dashed #aaa; background-color: #eee;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)
        
        self.btn_upload = QPushButton("選擇證件照/生活照...")
        self.btn_upload.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_upload)
        
        # 2. 資料輸入表單
        form_layout = QFormLayout()
        self.input_id = QLineEdit()
        self.input_name = QLineEdit()
        form_layout.addRow("員工編號 (ID):", self.input_id)
        form_layout.addRow("員工姓名 (Name):", self.input_name)
        right_layout.addLayout(form_layout)
        
        # 3. 註冊按鈕
        self.btn_register = QPushButton("確認註冊")
        self.btn_register.setStyleSheet("background-color: #ccffcc; color: green; font-weight: bold; padding: 10px;")
        self.btn_register.clicked.connect(self.register_employee)
        self.btn_register.setEnabled(False) # 一開始先停用
        right_layout.addWidget(self.btn_register)
        
        right_layout.addStretch() # 往上頂
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, stretch=1)

    def refresh_employee_list(self):
        """從資料庫讀取並更新列表"""
        self.emp_list.clear()
        employees = self.db.load_all_employees()
        for emp_id, data in employees.items():
            self.emp_list.addItem(f"{emp_id} - {data['name']}")

    def load_image(self):
        """選擇照片並進行前處理預覽"""
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇照片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return

        # 讀取並偵測
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "錯誤", "無法讀取照片檔案！")
            return

        # 使用 FaceAnalysis (app.get) 進行全圖偵測與對齊
        faces = self.recognizer.app.get(img)
        
        if len(faces) == 0:
            QMessageBox.warning(self, "偵測失敗", "照片中找不到人臉，請換一張清晰的照片。")
            self.reset_preview()
            return
        
        if len(faces) > 1:
             QMessageBox.information(self, "提示", "偵測到多張人臉，系統自動選取最大的一張。")
             # 選最大張的臉
             faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)

        # 取得對齊後的特徵
        face = faces[0]
        self.current_feature = face.normed_embedding 
        
        # 取得裁切後的臉 (用於預覽與存檔)
        b = list(map(int, face.bbox))
        face_crop = img[max(0,b[1]):b[3], max(0,b[0]):b[2]]
        
        if face_crop.size > 0:
            # 將裁切好的照片存入 Class 變數，供稍後存檔使用
            self.current_face_img = face_crop 
            
            # 轉成 Qt 影像顯示
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w, ch = face_crop_rgb.shape
            qimg = QImage(face_crop_rgb.data, w, h, w*ch, QImage.Format_RGB888)
            self.lbl_preview.setPixmap(QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio))
            self.lbl_preview.setText("")
            self.btn_register.setEnabled(True)
        else:
             self.lbl_preview.setText("預覽失敗")

    def register_employee(self):
        """寫入資料庫並儲存照片"""
        emp_id = self.input_id.text().strip()
        name = self.input_name.text().strip()
        
        if not emp_id or not name:
            QMessageBox.warning(self, "資料不全", "請輸入員工編號與姓名！")
            return
            
        if self.current_feature is None:
             QMessageBox.warning(self, "錯誤", "請先上傳照片！")
             return

        # 寫入資料庫
        try:
            # 1. 存入 SQLite
            self.db.register_employee(emp_id, name, self.current_feature)
            
            # 2. 儲存實際照片檔案
            if self.current_face_img is not None:
                save_dir = "data/faces"
                os.makedirs(save_dir, exist_ok=True)
                
                # 檔名格式：ID_Name.jpg (簡單過濾掉空格，避免檔名問題)
                safe_name = name.replace(" ", "_")
                filename = f"{emp_id}_{safe_name}.jpg"
                save_path = os.path.join(save_dir, filename)
                
                # 寫入硬碟
                cv2.imwrite(save_path, self.current_face_img)
                print(f"照片已儲存至: {save_path}")

            QMessageBox.information(self, "成功", f"員工 {name} ({emp_id}) 註冊成功！")
            
            # 重置介面
            self.input_id.clear()
            self.input_name.clear()
            self.reset_preview()
            self.refresh_employee_list()
            
        except Exception as e:
            QMessageBox.critical(self, "系統錯誤", f"註冊失敗：{str(e)}")

    def delete_employee(self):
        """刪除員工 (防呆機制)"""
        current_item = self.emp_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "提示", "請先從列表中點選要刪除的員工。")
            return
            
        # 解析字串 "101 - Andy" -> 取得 ID
        raw_text = current_item.text()
        emp_id = raw_text.split(" - ")[0]
        emp_name = raw_text.split(" - ")[1]
        
        # 防呆對話框
        reply = QMessageBox.question(
            self, 
            "確認刪除", 
            f"您確定要刪除員工：{emp_name} ({emp_id}) 嗎？\n\n注意：此操作將一併刪除該員工的所有打卡歷史紀錄，且無法復原！",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success, msg = self.db.delete_employee(emp_id)
            if success:
                # 嘗試刪除對應的照片檔 (Optional)
                try:
                    target_file = f"data/faces/{emp_id}_{emp_name.replace(' ', '_')}.jpg"
                    if os.path.exists(target_file):
                        os.remove(target_file)
                except:
                    pass # 照片刪除失敗不影響資料庫操作
                
                QMessageBox.information(self, "成功", "員工資料已刪除。")
                self.refresh_employee_list()
            else:
                QMessageBox.critical(self, "失敗", msg)

    def reset_preview(self):
        self.lbl_preview.clear()
        self.lbl_preview.setText("請上傳照片")
        self.current_feature = None
        self.current_face_img = None # 重置暫存照片
        self.btn_register.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminWindow()
    window.show()
    sys.exit(app.exec())