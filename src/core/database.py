import sqlite3
import json
import numpy as np
import os
import yaml
from datetime import datetime, date

class AttendanceDB:
    """
    負責 SQLite 資料庫的初始化、員工特徵管理與打卡紀錄存取。
    V2.1 支援詳細得分 (Base/Dynamic) 儲存與特徵演進紀錄。
    """
    def __init__(self, config_path="config.yaml"):
        # 載入設定檔
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # 備用預設值，避免設定檔不存在時崩潰
            self.config = {
                'database': {'db_path': 'data/attendance.db'},
                'attendance': {'debounce_minutes': 5}
            }
            
        self.db_path = self.config['database']['db_path']
        self.debounce_min = self.config['attendance']['debounce_minutes']
        
        # 確保資料庫目錄存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        """建立資料庫連線"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化資料表結構"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 員工表 (儲存基本資訊與 512 維特徵向量)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_feature TEXT NOT NULL,      
                    dynamic_feature TEXT,            
                    last_updated TIMESTAMP
                )
            ''')
            
            # 2. 打卡紀錄表 (V2.1 新增 base_score, dynamic_score 用於效能分析)
            # 注意：若已存在舊資料庫，請手動刪除 data/attendance.db 以套用新欄位
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
            conn.commit()

    def register_employee(self, emp_id, name, feature_vector):
        """
        註冊新員工
        :param emp_id: 員工編號
        :param name: 員工姓名
        :param feature_vector: numpy array (512,)
        """
        feature_json = json.dumps(feature_vector.tolist())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO employees (employee_id, name, base_feature, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (emp_id, name, feature_json, datetime.now()))
            conn.commit()

    def delete_employee(self, emp_id):
        """
        註銷/刪除員工及其所有打卡紀錄
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('DELETE FROM logs WHERE employee_id = ?', (emp_id,))
                cursor.execute('DELETE FROM employees WHERE employee_id = ?', (emp_id,))
                conn.commit()
                return True, f"員工 {emp_id} 已成功註銷。"
            except Exception as e:
                return False, f"註銷失敗: {str(e)}"

    def update_dynamic_feature(self, emp_id, new_feature_vector):
        """
        執行特徵演進：更新動態特徵
        """
        feature_json = json.dumps(new_feature_vector.tolist())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE employees 
                SET dynamic_feature = ?, last_updated = ?
                WHERE employee_id = ?
            ''', (feature_json, datetime.now(), emp_id))
            conn.commit()

    def load_all_employees(self):
        """
        讀取所有員工資料用於辨識比對
        """
        employees = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, base_feature, dynamic_feature FROM employees')
            for row in cursor.fetchall():
                eid, name, base_json, dynamic_json = row
                employees[eid] = {
                    'name': name,
                    'base': np.array(json.loads(base_json), dtype=np.float32),
                    'dynamic': np.array(json.loads(dynamic_json), dtype=np.float32) if dynamic_json else None
                }
        return employees

    def add_attendance_log(self, emp_id, confidence, photo_path, details=None):
        """
        記錄一次打卡，包含去抖動邏輯與詳細得分紀錄
        :param details: 包含 base_score 與 dynamic_score 的字典
        """
        now = datetime.now()
        
        # 提取詳細分數 (由 Recognizer 提供)
        base_s = details.get('base_score', 0.0) if details else 0.0
        dyn_s = details.get('dynamic_score', 0.0) if details else 0.0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 檢查最後一次打卡時間 (去抖動)
            cursor.execute('''
                SELECT timestamp FROM logs 
                WHERE employee_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (emp_id,))
            last_log = cursor.fetchone()
            
            if last_log:
                try:
                    last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    last_time = datetime.strptime(last_log[0], '%Y-%m-%d %H:%M:%S')
                
                elapsed_seconds = (now - last_time).total_seconds()
                debounce_seconds = self.debounce_min * 60
                
                if elapsed_seconds < debounce_seconds:
                    remaining = int(debounce_seconds - elapsed_seconds)
                    return False, f"打卡過於頻繁，請於 {remaining} 秒後再試。"

            # 插入新紀錄 (包含新欄位)
            cursor.execute('''
                INSERT INTO logs (employee_id, timestamp, confidence, base_score, dynamic_score, photo_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (emp_id, now, float(confidence), base_s, dyn_s, photo_path))
            conn.commit()
            return True, "打卡成功"

    def get_today_stats(self):
        """
        獲取今日所有員工的考勤統計
        """
        today_start = datetime.combine(date.today(), datetime.min.time())
        stats = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.name, MIN(l.timestamp), MAX(l.timestamp)
                FROM logs l
                JOIN employees e ON l.employee_id = e.employee_id
                WHERE l.timestamp >= ?
                GROUP BY l.employee_id
            ''', (today_start,))
            
            for row in cursor.fetchall():
                name, first_in, last_out = row
                stats.append({
                    'name': name,
                    'first_in': first_in.split('.')[0] if first_in else "N/A",
                    'last_out': last_out.split('.')[0] if first_in != last_out else "尚未下班"
                })
        return stats

    def get_recent_logs(self, limit=10):
        """
        獲取最近的打卡清單，用於 UI 畫面的結果反饋
        """
        logs = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.name, l.timestamp, l.confidence
                FROM logs l
                JOIN employees e ON l.employee_id = e.employee_id
                ORDER BY l.timestamp DESC
                LIMIT ?
            ''', (limit,))
            for row in cursor.fetchall():
                logs.append({
                    'name': row[0], 
                    'time': row[1].split('.')[0],
                    'score': round(row[2], 2)
                })
        return logs

if __name__ == "__main__":
    db = AttendanceDB()
    print("資料庫模組載入成功。")