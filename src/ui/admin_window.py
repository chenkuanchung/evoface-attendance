import sys
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
                             QLineEdit, QFileDialog, QMessageBox, QGroupBox, 
                             QFormLayout, QTabWidget, QComboBox, QListWidgetItem,
                             QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
                             QDateEdit)
from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

# 引用核心模組
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB
from src.core.calculator import AttendanceCalculator # 引入計算核心

class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EvoFace - 考勤管理後台")
        self.resize(1200, 800)
        
        # 初始化核心
        self.db = AttendanceDB()
        self.recognizer = FaceRecognizer()
        self.calc = AttendanceCalculator() # 初始化計算機
        
        # 暫存變數
        self.current_feature = None
        self.current_face_img = None
        
        self.init_ui()
        self.refresh_employee_list()
        self.refresh_approval_list()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 建立分頁
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # --- Tab 1: 員工資料管理 ---
        self.tab_emp = QWidget()
        self.init_employee_tab()
        self.tabs.addTab(self.tab_emp, "👥 員工資料管理")
        
        # --- Tab 2: 補登簽核中心 ---
        self.tab_approval = QWidget()
        self.init_approval_tab()
        self.tabs.addTab(self.tab_approval, "📝 補登簽核中心")

        # --- Tab 3: 考勤報表 (新增) ---
        self.tab_report = QWidget()
        self.init_report_tab()
        self.tabs.addTab(self.tab_report, "📊 考勤報表與匯出")

        # 建立一個水平佈局 (HBoxLayout) 來放按鈕，避免按鈕被拉得太長
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch() # 彈簧，把按鈕頂到右邊 (可選)
        
        self.btn_backup = QPushButton("💾 立即備份資料庫")
        self.btn_backup.setFixedWidth(200) # (可選) 設定固定寬度比較美觀
        self.btn_backup.setStyleSheet("""
            QPushButton {
                background-color: #D6EAF8; 
                color: #21618C; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #AED6F1;
            }
        """)
        self.btn_backup.clicked.connect(self.perform_backup)
        
        bottom_layout.addWidget(self.btn_backup) # 加入水平佈局
        main_layout.addLayout(bottom_layout)     # 將按鈕佈局加入主畫面

    # ==========================================
    # Tab 1: 員工管理
    # ==========================================
    def init_employee_tab(self):
        layout = QHBoxLayout(self.tab_emp)
        
        # 左側：員工列表
        left_panel = QGroupBox("現有員工名單")
        left_layout = QVBoxLayout()
        self.emp_list = QListWidget()
        self.emp_list.setStyleSheet("font-size: 14px;")
        self.emp_list.itemClicked.connect(self.on_emp_selected)
        left_layout.addWidget(self.emp_list)
        
        self.btn_delete = QPushButton("刪除選取員工")
        self.btn_delete.setStyleSheet("background-color: #ffcccc; color: red; padding: 8px;")
        self.btn_delete.clicked.connect(self.delete_employee)
        left_layout.addWidget(self.btn_delete)
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, stretch=1)
        
        # 右側：新增/編輯
        right_panel = QGroupBox("新增 / 註冊員工")
        right_layout = QVBoxLayout()
        
        self.lbl_preview = QLabel("請上傳照片")
        self.lbl_preview.setFixedSize(200, 200)
        self.lbl_preview.setStyleSheet("border: 2px dashed #aaa; background-color: #eee;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_preview, alignment=Qt.AlignCenter)
        
        self.btn_upload = QPushButton("📷 選擇證件照...")
        self.btn_upload.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_upload)
        
        form_layout = QFormLayout()
        self.input_id = QLineEdit()
        self.input_name = QLineEdit()
        self.input_pwd = QLineEdit()
        self.input_pwd.setEchoMode(QLineEdit.Password)
        self.input_pwd.setPlaceholderText("留空則預設為員工ID")
        
        self.combo_shift = QComboBox()
        self.combo_shift.addItem("未指定 (自動判斷)", None)
        self.combo_shift.addItem("早班 (08:00-17:00)", "morning")
        self.combo_shift.addItem("晚班 (16:00-01:00)", "evening")
        self.combo_shift.addItem("大夜班 (00:00-09:00)", "night")
        
        form_layout.addRow("員工編號 (ID):", self.input_id)
        form_layout.addRow("員工姓名 (Name):", self.input_name)
        form_layout.addRow("登入密碼 (Pwd):", self.input_pwd)
        form_layout.addRow("預設班別 (Shift):", self.combo_shift)
        right_layout.addLayout(form_layout)
        
        self.btn_register = QPushButton("確認註冊 / 更新資料")
        self.btn_register.setStyleSheet("background-color: #ccffcc; color: green; font-weight: bold; padding: 10px;")
        self.btn_register.clicked.connect(self.register_employee)
        self.btn_register.setEnabled(False) 
        right_layout.addWidget(self.btn_register)
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, stretch=1)

    # ==========================================
    # Tab 2: 簽核中心
    # ==========================================
    def init_approval_tab(self):
        layout = QVBoxLayout(self.tab_approval)
        
        top_bar = QHBoxLayout()
        btn_refresh = QPushButton("🔄 重新整理")
        btn_refresh.clicked.connect(self.refresh_approval_list)
        top_bar.addWidget(QLabel("待審核申請列表 (請勾選要處理的項目)"))
        top_bar.addStretch()
        top_bar.addWidget(btn_refresh)
        layout.addLayout(top_bar)
        
        self.table_approval = QTableWidget()
        self.table_approval.setColumnCount(7)
        self.table_approval.setHorizontalHeaderLabels(["選取", "ID", "申請人", "日期", "類型", "時間", "理由"])
        self.table_approval.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_approval.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_approval.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_approval.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table_approval)
        
        action_layout = QHBoxLayout()
        self.btn_approve = QPushButton("✅ 核准選取項目")
        self.btn_approve.setStyleSheet("background-color: #E8F8F5; color: #117A65; font-weight: bold; padding: 10px;")
        self.btn_approve.clicked.connect(self.approve_request)
        
        self.btn_reject = QPushButton("❌ 駁回選取項目")
        self.btn_reject.setStyleSheet("background-color: #FDEBD0; color: #9A7D0A; font-weight: bold; padding: 10px;")
        self.btn_reject.clicked.connect(self.reject_request)
        
        action_layout.addWidget(self.btn_approve)
        action_layout.addWidget(self.btn_reject)
        layout.addLayout(action_layout)

    # ==========================================
    # Tab 3: 報表中心 (New!)
    # ==========================================
    def init_report_tab(self):
        layout = QVBoxLayout(self.tab_report)
        
        # 1. 查詢控制列
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("查詢區間:"))
        
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate.currentDate().addDays(-30)) # 預設查前30天
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate.currentDate())
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        
        ctrl_layout.addWidget(self.date_start)
        ctrl_layout.addWidget(QLabel("至"))
        ctrl_layout.addWidget(self.date_end)
        
        btn_query = QPushButton("🔍 產生報表")
        btn_query.clicked.connect(self.generate_report)
        ctrl_layout.addWidget(btn_query)
        
        btn_export = QPushButton("📥 匯出 Excel")
        btn_export.setStyleSheet("background-color: #2E86C1; color: white; font-weight: bold;")
        btn_export.clicked.connect(self.export_report)
        ctrl_layout.addWidget(btn_export)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # 2. 報表顯示表格
        self.table_report = QTableWidget()
        self.table_report.setColumnCount(8)
        self.table_report.setHorizontalHeaderLabels(["日期", "員工ID", "姓名", "班別", "上班", "下班", "工時", "狀態"])
        self.table_report.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_report)

    # --- 功能實作: 報表 ---
    
    def generate_report(self):
        """計算並顯示報表"""
        start_date = self.date_start.date().toPython()
        end_date = self.date_end.date().toPython()
        
        # 1. 準備數據
        self.report_data = [] # 暫存供匯出使用
        
        # 取得所有員工
        employees = self.db.load_all_employees()
        
        # 擴大搜尋範圍 (前後多一天避免跨日遺漏)
        query_start = datetime.combine(start_date - timedelta(days=1), datetime.min.time())
        query_end = datetime.combine(end_date + timedelta(days=2), datetime.min.time())
        
        # 撈出所有 Logs (效能優化點：這裡撈出全部再過濾，量大時建議改 SQL)
        all_logs_raw = self.db.get_logs_by_range(query_start, query_end)
        # 轉成 (dt, emp_id) 列表
        all_logs = []
        for l in all_logs_raw:
            # l 可能是 tuple 或 string，視 database.py 實作而定
            # 根據上一版 database.py get_logs_by_range 回傳的是 timestamp string list (若無 employee_id 參數)
            # 但這裡我們需要知道是誰打的卡，所以需要修正 get_logs_by_range 或 這裡多做處理
            pass 

        # 修正策略：我們用更簡單的方式，依員工迴圈計算 (小規模適用)
        self.table_report.setRowCount(0)
        row_idx = 0
        
        for emp_id, emp_data in employees.items():
            # 針對每個員工撈 Logs
            logs = self.db.get_logs_by_range(query_start, query_end, emp_id)
            logs_dt = []
            for t_str in logs:
                try:
                    logs_dt.append(datetime.strptime(t_str.split('.')[0], '%Y-%m-%d %H:%M:%S'))
                except:
                    pass
            
            # 逐日計算
            current_d = start_date
            while current_d <= end_date:
                # 篩選當日 Logs
                daily_logs = [log for log in logs_dt if self.calc.get_logical_day(log) == current_d]
                
                # 計算
                stats = self.calc.calculate_daily_stats(
                    current_d.strftime("%Y-%m-%d"), 
                    daily_logs, 
                    emp_data.get('default_shift')
                )
                
                # 填入表格
                self.table_report.insertRow(row_idx)
                self.table_report.setItem(row_idx, 0, QTableWidgetItem(stats['date']))
                self.table_report.setItem(row_idx, 1, QTableWidgetItem(emp_id))
                self.table_report.setItem(row_idx, 2, QTableWidgetItem(emp_data['name']))
                self.table_report.setItem(row_idx, 3, QTableWidgetItem(stats['shift']))
                self.table_report.setItem(row_idx, 4, QTableWidgetItem(stats['in'].strftime("%H:%M") if stats['in'] else "--"))
                self.table_report.setItem(row_idx, 5, QTableWidgetItem(stats['out'].strftime("%H:%M") if stats['out'] else "--"))
                self.table_report.setItem(row_idx, 6, QTableWidgetItem(str(stats['hours'])))
                
                status_item = QTableWidgetItem(stats['status'])
                # 紅字標記異常
                if "異常" in stats['status'] or "缺" in stats['status'] or "遲到" in stats['status']:
                    status_item.setForeground(QColor("red"))
                    status_item.setFont(QFont("Arial", 9, QFont.Bold))
                elif "正常" in stats['status']:
                    status_item.setForeground(QColor("green"))
                
                self.table_report.setItem(row_idx, 7, status_item)
                
                # 儲存資料供匯出
                self.report_data.append({
                    "日期": stats['date'], "ID": emp_id, "姓名": emp_data['name'],
                    "班別": stats['shift'], "上班": stats['in'], "下班": stats['out'],
                    "工時": stats['hours'], "狀態": stats['status']
                })
                
                row_idx += 1
                current_d += timedelta(days=1)

    def export_report(self):
        """匯出成 Excel"""
        if not hasattr(self, 'report_data') or not self.report_data:
            QMessageBox.warning(self, "提示", "請先執行查詢產生報表！")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "匯出報表", "考勤報表.xlsx", "Excel Files (*.xlsx)")
        if not file_path:
            return
            
        try:
            df = pd.DataFrame(self.report_data)
            # 格式化一下時間，避免顯示完整 datetime
            df['上班'] = df['上班'].apply(lambda x: x.strftime("%H:%M:%S") if x else "")
            df['下班'] = df['下班'].apply(lambda x: x.strftime("%H:%M:%S") if x else "")
            
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "成功", "報表匯出成功！")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"匯出失敗：{str(e)}\n請確認是否已安裝 openpyxl")

    # ==========================================
    # 以下為 Tab 1 & Tab 2 的原有邏輯 (保持不變)
    # ==========================================

    def refresh_employee_list(self):
        self.emp_list.clear()
        employees = self.db.load_all_employees()
        for emp_id, data in employees.items():
            shift_info = data.get('default_shift')
            shift_str = f"[{shift_info}]" if shift_info else ""
            item_text = f"{emp_id} - {data['name']} {shift_str}"
            self.emp_list.addItem(item_text)

    def on_emp_selected(self, item):
        text = item.text()
        parts = text.split(" - ")
        if len(parts) >= 2:
            self.input_id.setText(parts[0])
            self.input_name.setText(parts[1].split(" [")[0])
            self.btn_register.setText("更新資料 (需重新上傳照片)")
            self.btn_register.setEnabled(True)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇照片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path: return
        try:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            img = None
        if img is None: return
        faces = self.recognizer.app.get(img)
        if len(faces) == 0:
            QMessageBox.warning(self, "失敗", "找不到人臉")
            return
        if len(faces) > 1:
             faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        self.current_feature = face.normed_embedding 
        b = list(map(int, face.bbox))
        face_crop = img[max(0,b[1]):b[3], max(0,b[0]):b[2]]
        if face_crop.size > 0:
            self.current_face_img = face_crop 
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w, ch = face_crop_rgb.shape
            qimg = QImage(face_crop_rgb.data, w, h, w*ch, QImage.Format_RGB888)
            self.lbl_preview.setPixmap(QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio))
            self.btn_register.setEnabled(True)

    def register_employee(self):
        emp_id = self.input_id.text().strip()
        name = self.input_name.text().strip()
        pwd = self.input_pwd.text().strip()
        shift = self.combo_shift.currentData() 
        if not emp_id or not name: return
        if self.current_feature is None: return
        try:
            self.db.register_employee(emp_id, name, self.current_feature, password=pwd if pwd else None, default_shift=shift)
            if self.current_face_img is not None:
                os.makedirs("data/faces", exist_ok=True)
                safe_name = name.replace(" ", "_")
                filename = f"{emp_id}_{safe_name}.jpg"
                save_path = os.path.join("data/faces", filename)
                cv2.imencode('.jpg', self.current_face_img)[1].tofile(save_path)
            QMessageBox.information(self, "成功", f"員工 {name} 資料已更新！")
            self.refresh_employee_list()
            self.input_id.clear(); self.input_name.clear(); self.lbl_preview.clear(); self.current_feature = None
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def delete_employee(self):
        current_item = self.emp_list.currentItem()
        if not current_item: return
        emp_id = current_item.text().split(" - ")[0]
        if QMessageBox.question(self, "確認", "確定刪除？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            with self.db._get_connection() as conn:
                conn.execute("DELETE FROM employees WHERE employee_id=?", (emp_id,))
                conn.execute("DELETE FROM logs WHERE employee_id=?", (emp_id,))
            self.refresh_employee_list()

    def refresh_approval_list(self):
        self.table_approval.setRowCount(0)
        requests = self.db.get_pending_requests()
        for row_idx, req in enumerate(requests):
            self.table_approval.insertRow(row_idx)
            item_check = QTableWidgetItem()
            item_check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item_check.setCheckState(Qt.Unchecked)
            item_check.setData(Qt.UserRole, req['id'])
            self.table_approval.setItem(row_idx, 0, item_check)
            self.table_approval.setItem(row_idx, 1, QTableWidgetItem(str(req.get('employee_id', ''))))
            self.table_approval.setItem(row_idx, 2, QTableWidgetItem(req['name']))
            self.table_approval.setItem(row_idx, 3, QTableWidgetItem(req['date']))
            self.table_approval.setItem(row_idx, 4, QTableWidgetItem(req['type']))
            self.table_approval.setItem(row_idx, 5, QTableWidgetItem(req['time']))
            self.table_approval.setItem(row_idx, 6, QTableWidgetItem(req['reason']))

    def process_request(self, decision):
        target_ids = []
        for row in range(self.table_approval.rowCount()):
            item = self.table_approval.item(row, 0)
            if item.checkState() == Qt.Checked:
                target_ids.append(item.data(Qt.UserRole))
        if not target_ids:
            QMessageBox.warning(self, "提示", "請先勾選項目")
            return
        if QMessageBox.question(self, "確認", f"確定處理這 {len(target_ids)} 筆？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            for req_id in target_ids:
                self.db.approve_request(req_id, decision)
            QMessageBox.information(self, "完成", "處理完畢")
            self.refresh_approval_list()

    def perform_backup(self):
        import sqlite3
        
        db_path = "data/attendance.db"
        backup_dir = "backup"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"attendance_backup_{timestamp}.db")
        
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        try:
            # 連接到現有資料庫
            src_conn = sqlite3.connect(db_path)
            # 連接到備份目標檔案 (會自動建立)
            dst_conn = sqlite3.connect(backup_path)
            
            with dst_conn:
                # 使用 SQLite 的 Online Backup API
                # 這會自動處理鎖定問題，確保備份的一致性
                src_conn.backup(dst_conn)
            
            dst_conn.close()
            src_conn.close()
            
            QMessageBox.information(self, "成功", f"安全備份完成！\n{backup_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "失敗", str(e))

    def approve_request(self): self.process_request('approved')
    def reject_request(self): self.process_request('rejected')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminWindow()
    window.show()
    sys.exit(app.exec())