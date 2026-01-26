import sqlite3
import json
import numpy as np
import os
import yaml
import hashlib
from datetime import datetime, date, timedelta, time

class AttendanceDB:
    def __init__(self, config_path="config.yaml"):
        # 載入設定
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {}
            
        self.db_path = self.config.get('database', {}).get('db_path', 'data/attendance.db')
        self.debounce_min = self.config.get('attendance', {}).get('debounce_minutes', 1)
        # 讀取日切點設定，預設 04:00
        self.day_cutoff = self.config.get('attendance', {}).get('day_cutoff', "04:00")
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._migrate_db() # 自動檢查並升級欄位

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化基礎表結構"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 員工表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    password_hash TEXT,
                    default_shift TEXT,              
                    base_feature TEXT NOT NULL,      
                    dynamic_feature TEXT,            
                    last_updated TIMESTAMP
                )
            ''')
            
            # 2. 原始打卡流水帳
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    timestamp DATETIME,
                    confidence REAL,
                    base_score REAL,
                    dynamic_score REAL,
                    photo_path TEXT,
                    FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
                )
            ''')

            # 3. 每日考勤結算表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,           
                    employee_id TEXT,
                    shift_name TEXT,              
                    clock_in DATETIME,            
                    clock_out DATETIME,           
                    work_hours REAL,              
                    status TEXT,                  
                    is_modified BOOLEAN DEFAULT 0,
                    UNIQUE(date, employee_id)
                )
            ''')

            # 4. 補登申請表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS manual_requests (
                    req_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    target_date TEXT,      -- 補登日期 YYYY-MM-DD
                    request_type TEXT,     -- 'IN' (補上班) or 'OUT' (補下班)
                    request_time TEXT,     -- 補登時間 HH:MM
                    reason TEXT,
                    status TEXT DEFAULT 'pending', -- pending, approved, rejected
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()

    def _migrate_db(self):
        """檢查並補上舊版資料庫缺少的欄位"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(employees)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'password_hash' not in columns:
                print("DB Migration: 新增 password_hash")
                cursor.execute("ALTER TABLE employees ADD COLUMN password_hash TEXT")
                cursor.execute("UPDATE employees SET password_hash = employee_id")
            
            if 'default_shift' not in columns:
                print("DB Migration: 新增 default_shift")
                cursor.execute("ALTER TABLE employees ADD COLUMN default_shift TEXT")
                
            conn.commit()

    # --- 員工管理 ---
    def register_employee(self, emp_id, name, feature_vector, password=None, default_shift=None):
        feature_json = json.dumps(feature_vector.tolist())
        pwd = password if password else emp_id
        pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO employees (employee_id, name, password_hash, default_shift, base_feature, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (emp_id, name, pwd_hash, default_shift, feature_json, datetime.now()))
            conn.commit()

    def update_employee_shift(self, emp_id, shift_code):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE employees SET default_shift = ? WHERE employee_id = ?", (shift_code, emp_id))
            conn.commit()

    def verify_password(self, emp_id, input_password):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM employees WHERE employee_id = ?', (emp_id,))
            row = cursor.fetchone()
            if row:
                input_hash = hashlib.sha256(input_password.encode()).hexdigest()
                return input_hash == row[0]
        return False

    def load_all_employees(self):
        employees = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, base_feature, dynamic_feature, default_shift FROM employees')
            for row in cursor.fetchall():
                eid, name, base_json, dynamic_json, def_shift = row
                employees[eid] = {
                    'name': name,
                    'base': np.array(json.loads(base_json), dtype=np.float32),
                    'dynamic': np.array(json.loads(dynamic_json), dtype=np.float32) if dynamic_json else None,
                    'default_shift': def_shift
                }
        return employees

    # --- 補登申請功能 ---
    def create_request(self, emp_id, date_str, req_type, time_str, reason):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO manual_requests (employee_id, target_date, request_type, request_time, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (emp_id, date_str, req_type, time_str, reason, datetime.now()))
            conn.commit()
            return True

    def get_pending_requests(self):
        """管理員取得待審核清單 (已修正：包含 employee_id)"""
        requests = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.req_id, e.name, r.target_date, r.request_type, r.request_time, r.reason, r.employee_id
                FROM manual_requests r
                JOIN employees e ON r.employee_id = e.employee_id
                WHERE r.status = 'pending'
                ORDER BY r.created_at DESC
            ''')
            for row in cursor.fetchall():
                requests.append({
                    'id': row[0], 'name': row[1], 'date': row[2],
                    'type': row[3], 'time': row[4], 'reason': row[5],
                    'employee_id': row[6]
                })
        return requests

    def approve_request(self, req_id, status='approved'):
        """
        審核申請 (核心修正：核准時自動寫入 Logs)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 取得申請單詳細資料
            cursor.execute("SELECT employee_id, target_date, request_time FROM manual_requests WHERE req_id = ?", (req_id,))
            req_data = cursor.fetchone()
            
            if not req_data:
                return False
                
            emp_id, date_str, time_str = req_data
            
            # 2. 如果是核准，則計算正確的時間戳記並插入 logs
            if status == 'approved':
                # 判斷是否為跨日打卡 (透過 day_cutoff)
                # 例如 cutoff=04:00, 申請時間=01:00, 申請日期(歸屬日)=10/01
                # 則實際時間應該是 10/02 的 01:00
                
                try:
                    req_t = datetime.strptime(time_str, "%H:%M").time()
                    target_d = datetime.strptime(date_str, "%Y-%m-%d").date()
                    cutoff_h = int(self.day_cutoff.split(':')[0])
                    
                    final_dt = None
                    # 如果申請的時間 小於 日切點 (例如 01:00 < 04:00)
                    # 代表這是「隔天凌晨」的打卡，但歸屬於 target_d
                    if req_t.hour < cutoff_h:
                        real_date = target_d + timedelta(days=1)
                        final_dt = datetime.combine(real_date, req_t)
                    else:
                        # 正常當日打卡
                        final_dt = datetime.combine(target_d, req_t)
                    
                    # 3. 寫入 Logs 表 (標記 photo_path 為 MANUAL_APPROVAL)
                    # 直接寫入，不經過去抖動檢查 (因為是人工補登)
                    cursor.execute('''
                        INSERT INTO logs (employee_id, timestamp, confidence, base_score, dynamic_score, photo_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (emp_id, final_dt, 1.0, 0.0, 0.0, 'MANUAL_APPROVAL'))
                    
                except Exception as e:
                    print(f"Approval Error: {e}")
                    return False

            # 4. 更新申請單狀態
            cursor.execute("UPDATE manual_requests SET status = ? WHERE req_id = ?", (status, req_id))
            conn.commit()
            return True

    # --- 打卡紀錄與日結 ---
    def add_attendance_log(self, emp_id, confidence, photo_path, details=None):
        now = datetime.now()
        base_s = details.get('base_score', 0.0) if details else 0.0
        dyn_s = details.get('dynamic_score', 0.0) if details else 0.0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 去抖動檢查
            cursor.execute('SELECT timestamp FROM logs WHERE employee_id = ? ORDER BY timestamp DESC LIMIT 1', (emp_id,))
            last_log = cursor.fetchone()
            
            if last_log:
                try:
                    last_time = datetime.strptime(last_log[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S.%f')

                elapsed = (now - last_time).total_seconds()
                limit_seconds = self.debounce_min * 60
                
                if elapsed < limit_seconds:
                    remaining = int(limit_seconds - elapsed)
                    return False, f"打卡過於頻繁，請於 {remaining} 秒後再試。"

            cursor.execute('''
                INSERT INTO logs (employee_id, timestamp, confidence, base_score, dynamic_score, photo_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (emp_id, now, float(confidence), base_s, dyn_s, photo_path))
            conn.commit()
            return True, "打卡成功"
            
    def update_dynamic_feature(self, emp_id, new_feature_vector):
        feature_json = json.dumps(new_feature_vector.tolist())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE employees SET dynamic_feature = ?, last_updated = ? WHERE employee_id = ?', 
                          (feature_json, datetime.now(), emp_id))
            conn.commit()

    def get_logs_by_range(self, start_time, end_time, emp_id=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT timestamp FROM logs WHERE timestamp BETWEEN ? AND ?"
            params = [start_time, end_time]
            if emp_id:
                query += " AND employee_id = ?"
                params.append(emp_id)
            query += " ORDER BY timestamp ASC"
            cursor.execute(query, params)
            return [row[0] for row in cursor.fetchall()]

    def get_recent_logs(self, limit=10):
        logs = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.name, l.timestamp, l.confidence
                FROM logs l
                JOIN employees e ON l.employee_id = e.employee_id
                ORDER BY l.timestamp DESC LIMIT ?
            ''', (limit,))
            for row in cursor.fetchall():
                logs.append({'name': row[0], 'time': row[1].split('.')[0], 'score': round(row[2], 2)})
        return logs