import sys
import os
import cv2
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
                             QLineEdit, QFileDialog, QMessageBox, QGroupBox, 
                             QFormLayout, QTabWidget, QComboBox, QListWidgetItem,
                             QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
                             QDateEdit, QListView, QCheckBox)
from PySide6.QtCore import Qt, QDate, QSortFilterProxyModel, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QStandardItemModel, QStandardItem

# 引用核心模組
from src.core.recognizer import FaceRecognizer
from src.core.database import AttendanceDB
from src.core.calculator import AttendanceCalculator # 引入計算核心

class BackupWorker(QThread):
    """獨立的備份執行緒，避免卡住 UI"""
    finished_signal = Signal(bool, str) # 回傳 (是否成功, 訊息)

    def __init__(self, db_path, backup_dir="backup"):
        super().__init__()
        self.db_path = db_path
        self.backup_dir = backup_dir

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"attendance_backup_{timestamp}.db")
        
        if not os.path.exists(self.backup_dir):
            try:
                os.makedirs(self.backup_dir)
            except Exception as e:
                self.finished_signal.emit(False, f"無法建立目錄: {str(e)}")
                return

        try:
            # 建立獨立連線進行備份
            src_conn = sqlite3.connect(self.db_path)
            dst_conn = sqlite3.connect(backup_path)
            
            with dst_conn:
                src_conn.backup(dst_conn)
            
            dst_conn.close()
            src_conn.close()
            
            self.finished_signal.emit(True, backup_path)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class AdminWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EvoFace - 考勤管理後台")
        self.resize(1200, 800)
        
        # 1. 初始化核心與資料庫
        self.db = AttendanceDB()
        self.recognizer = FaceRecognizer()
        self.calc = AttendanceCalculator()
        
        # 2. 先建立 Model 與 Proxy Model (放在 init_ui 之前！)
        self.emp_model = QStandardItemModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.emp_model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setFilterKeyColumn(0)
        
        # 暫存變數
        self.current_feature = None
        self.current_face_img = None
        
        # 3. 最後才初始化 UI
        self.init_ui()
        
        # 4. 載入資料
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
        
        left_panel = QGroupBox("現有員工名單")
        left_layout = QVBoxLayout()
        
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍 搜尋員工編號或姓名...")
        self.edit_search.textChanged.connect(self.proxy_model.setFilterFixedString) 
        left_layout.addWidget(self.edit_search)
        
        # 使用 QListView 搭配代理模型
        self.emp_view = QListView()
        self.emp_view.setModel(self.proxy_model)
        self.emp_view.setStyleSheet("font-size: 14px;")
        # 連結到正確的 v2 方法
        self.emp_view.clicked.connect(self.on_emp_selected_v2) 
        left_layout.addWidget(self.emp_view)
        
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

        self.btn_reset = QPushButton("↺ 取消編輯 / 回到新增模式")
        self.btn_reset.setStyleSheet("background-color: #f0f0f0; color: #555; padding: 8px;")
        self.btn_reset.clicked.connect(self.reset_form) # 連結到現有的 reset_form 方法

        right_layout.addWidget(self.btn_register)
        right_layout.addWidget(self.btn_reset)

        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, stretch=1)

    # ==========================================
    # Tab 2: 簽核中心
    # ==========================================
    def init_approval_tab(self):
        layout = QVBoxLayout(self.tab_approval)
        
        top_bar = QHBoxLayout()
        
        # 全選控制項
        self.chk_select_all = QCheckBox("全選所有項目")
        self.chk_select_all.stateChanged.connect(self.toggle_select_all)
        top_bar.addWidget(self.chk_select_all)
        
        top_bar.addStretch()
        
        btn_refresh = QPushButton("🔄 重新整理")
        btn_refresh.clicked.connect(self.refresh_approval_list)
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

    def refresh_employee_list(self):
        self.emp_model.clear()
        employees = self.db.load_all_employees()
        for emp_id, data in employees.items():
            shift_info = data.get('default_shift')
            shift_str = f"[{shift_info}]" if shift_info else ""
            item_text = f"{emp_id} - {data['name']} {shift_str}"
            
            item = QStandardItem(item_text)
            # 關鍵：儲存 ID，這樣過濾後才抓得對人
            item.setData(emp_id, Qt.UserRole) 
            self.emp_model.appendRow(item)

    def on_emp_selected_v2(self, index):
        """處理模型視圖點擊，支援搜尋後的正確映射"""
        source_index = self.proxy_model.mapToSource(index)
        item = self.emp_model.itemFromIndex(source_index)
        emp_id = item.data(Qt.UserRole)
        
        parts = item.text().split(" - ")
        if len(parts) >= 2:
            name = parts[1].split(" [")[0]
            self.input_id.setText(emp_id)
            self.input_id.setReadOnly(True) # 鎖定 ID
            self.input_id.setStyleSheet("background-color: #e9ecef;")
            self.input_name.setText(name)
            self.btn_register.setText("更新員工資料 (需重新上傳照片)")
            self.btn_register.setEnabled(False)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇照片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path: return
        try:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except:
            img = None
            
        if img is None: return
        
        # 1. 偵測人臉
        faces = self.recognizer.app.get(img)
        if len(faces) == 0:
            QMessageBox.warning(self, "失敗", "找不到人臉，請更換照片")
            return
            
        # 2. 取最大人臉
        if len(faces) > 1:
             faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # 3. 提取特徵
        self.current_feature = face.normed_embedding 
        
        # === 防呆機制：全庫特徵比對 ===
        try:
            employees = self.db.load_all_employees()
            max_score = 0.0
            similar_emp_name = ""
            similar_emp_id = ""
            
            # 遍歷所有員工進行 1:N 比對
            for eid, data in employees.items():
                # 如果是「更新模式」且比對到自己，就跳過 (自己跟自己像很正常)
                if self.input_id.isReadOnly() and eid == self.input_id.text():
                    continue
                    
                # 計算相似度 (直接呼叫 recognizer 的數學函式)
                score = self.recognizer.compute_similarity(self.current_feature, data['base'])
                if score > max_score:
                    max_score = score
                    similar_emp_name = data['name']
                    similar_emp_id = eid
            
            # 門檻值判斷 (0.5 為 InsightFace 的危險區)
            if max_score > 0.5:
                QMessageBox.warning(self, "相似度過高警告", 
                    f"⚠️ 注意：這張照片與現有員工高度相似！\n\n"
                    f"相似對象：{similar_emp_name} ({similar_emp_id})\n"
                    f"相似分數：{max_score:.2f}\n\n"
                    f"請確認該員工是否重複註冊，或照片是否混淆。")
                    
        except Exception as e:
            print(f"相似度檢查錯誤: {e}")
        # ===================================

        # 4. 顯示預覽圖 (原邏輯)
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
        
        if not emp_id or not name: 
            QMessageBox.warning(self, "提示", "ID 與 姓名 為必填欄位")
            return
            
        if self.current_feature is None: 
            QMessageBox.warning(self, "提示", "請先上傳並確認證件照")
            return
        
        # === 防呆機制：ID 重複檢查 ===
        # 只有在「新增模式」(ID 可編輯) 時才需要檢查
        # 如果是「更新模式」(ID 唯讀)，代表使用者本來就是要更新這個人，不用擋
        if not self.input_id.isReadOnly():
            existing_employees = self.db.load_all_employees()
            if emp_id in existing_employees:
                old_name = existing_employees[emp_id]['name']
                # 跳出確認視窗
                reply = QMessageBox.question(self, "ID 已存在", 
                    f"員工編號 {emp_id} 已經存在！\n"
                    f"原登記姓名：{old_name}\n\n"
                    f"您確定要「覆蓋」並更新這位員工的資料嗎？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.No:
                    return # 使用者按取消，中止註冊
        # ===================================
        
        try:
            # 1. 寫入資料庫
            self.db.register_employee(emp_id, name, self.current_feature, password=pwd if pwd else None, default_shift=shift)
            
            # 2. 儲存照片
            if self.current_face_img is not None:
                os.makedirs("data/faces", exist_ok=True)
                safe_name = name.replace(" ", "_")
                filename = f"{emp_id}_{safe_name}.jpg"
                save_path = os.path.join("data/faces", filename)
                cv2.imencode('.jpg', self.current_face_img)[1].tofile(save_path)
            
            # 3. 立即重載特徵庫，讓新員工生效
            self.recognizer.reload_employees() 

            QMessageBox.information(self, "成功", f"員工 {name} 資料已更新！")
            
            # 4. 成功後：重新整理清單 + 重置表單
            self.refresh_employee_list()
            self.reset_form() 
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def delete_employee(self):
        # 修正：從 QListView 的 selectionModel 獲取選中的索引
        selection_model = self.emp_view.selectionModel()
        selected_indexes = selection_model.selectedIndexes()
        
        if not selected_indexes:
            QMessageBox.warning(self, "提示", "請先選取要刪除的員工")
            return
            
        # 取得第一個選中項目的 ID (透過 Proxy Model 映射回 Source Model)
        index = selected_indexes[0]
        source_index = self.proxy_model.mapToSource(index)
        item = self.emp_model.itemFromIndex(source_index)
        
        # 讀取隱藏在 Item 中的員工 ID
        emp_id = item.data(Qt.UserRole)
        
        # 為了安全，顯示員工姓名給使用者確認
        display_text = item.text()
        name = display_text.split(" - ")[1].split(" [")[0] if " - " in display_text else emp_id

        if QMessageBox.question(self, "確認", f"確定刪除員工 {name} ({emp_id})？\n這將一併刪除其所有打卡紀錄且無法恢復。", 
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                with self.db._get_connection() as conn:
                    conn.execute("DELETE FROM employees WHERE employee_id=?", (emp_id,))
                    conn.execute("DELETE FROM logs WHERE employee_id=?", (emp_id,))
                
                self.recognizer.reload_employees() # 刪除後也要重載，清除記憶體中的特徵
                self.refresh_employee_list()
                self.reset_form() # 刪除後清空表單，避免畫面上殘留已不存在的資料
                QMessageBox.information(self, "成功", "資料已移除")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"刪除失敗：{str(e)}")

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
        """啟動非同步備份"""
        self.btn_backup.setEnabled(False)
        self.btn_backup.setText("⏳ 備份進行中...請稍候")
        
        # 實例化 Worker (傳入 db_path)
        db_path = self.db.db_path if hasattr(self.db, 'db_path') else "data/attendance.db"
        self.backup_thread = BackupWorker(db_path)
        self.backup_thread.finished_signal.connect(self.on_backup_finished)
        self.backup_thread.start()

    def on_backup_finished(self, success, result_msg):
        """備份完成後的 Callback"""
        self.btn_backup.setEnabled(True)
        self.btn_backup.setText("💾 立即備份資料庫")
        
        if success:
            QMessageBox.information(self, "備份成功", f"資料庫已安全備份至：\n{result_msg}")
        else:
            QMessageBox.critical(self, "備份失敗", f"發生錯誤：{result_msg}")

    def reset_form(self):
        """回到新增模式"""
        self.input_id.clear()
        self.input_id.setReadOnly(False)
        self.input_id.setStyleSheet("")
        self.input_name.clear()
        self.input_pwd.clear()
        self.lbl_preview.clear()
        self.lbl_preview.setText("請上傳證件照")
        self.current_feature = None
        self.current_face_img = None
        self.btn_register.setText("確認註冊新員工")
        self.btn_register.setEnabled(False)

    def toggle_select_all(self, state):
        """批次勾選/取消勾選"""
        is_checked = (state == Qt.Checked)
        for row in range(self.table_approval.rowCount()):
            item = self.table_approval.item(row, 0)
            # 只有在 Item 啟用的狀態下才勾選 (避免勾選到無效項目)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)

    def approve_request(self): self.process_request('approved')
    def reject_request(self): self.process_request('rejected')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdminWindow()
    window.show()
    sys.exit(app.exec())