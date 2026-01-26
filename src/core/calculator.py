import yaml
from datetime import datetime, timedelta, date

class AttendanceCalculator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.cutoff_str = self.config.get('attendance', {}).get('day_cutoff', "04:00")
        self.shifts = self.config.get('shifts', {})

    def get_logical_day(self, dt_obj):
        """根據日切點 (04:00) 決定歸屬日期"""
        cutoff_hour = int(self.cutoff_str.split(':')[0])
        if dt_obj.hour < cutoff_hour:
            return (dt_obj - timedelta(days=1)).date()
        return dt_obj.date()

    def determine_shift(self, first_log_dt, default_shift_code=None):
        """
        決定班別：
        1. 若員工有綁定 default_shift (如 morning)，強制使用該班別設定。
        2. 若無，則根據打卡時間自動判斷。
        """
        # 策略 1: 使用預設班別
        if default_shift_code and default_shift_code in self.shifts:
            return self.shifts[default_shift_code]['name'], self.shifts[default_shift_code]

        # 策略 2: 自動判斷
        if not first_log_dt: 
            return "未排班", None

        in_time_str = first_log_dt.strftime("%H:%M")
        for code, shift in self.shifts.items():
            start = shift['range_start']
            end = shift['range_end']
            
            # 處理跨日區間 (例如 21:00 ~ 03:00)
            if start > end: 
                if in_time_str >= start or in_time_str <= end:
                    return shift['name'], shift
            else:
                if start <= in_time_str <= end:
                    return shift['name'], shift
                    
        return "排班外", None

    def calculate_daily_stats(self, date_str, logs, default_shift_code=None):
        """
        計算單日考勤
        logs: 該歸屬日的所有打卡時間 (datetime list)
        default_shift_code: 員工的預設班別 (optional)
        """
        if not logs:
            return {
                'date': date_str, 'shift': 'N/A', 'in': None, 'out': None,
                'hours': 0, 'status': '缺卡'
            }

        first_in = min(logs)
        last_out = max(logs) if len(logs) > 1 else None

        # 判斷班別
        shift_name, shift_conf = self.determine_shift(first_in, default_shift_code)
        
        status_list = []
        
        # 1. 異常檢查：缺卡
        if not last_out:
            status_list.append("缺下班卡")
        
        # 2. 異常檢查：遲到 (如果有班別設定)
        if shift_conf:
            std_in_str = shift_conf['start_time']
            std_in_t = datetime.strptime(std_in_str, "%H:%M").time()
            
            # 簡化邏輯：如果打卡時間 "HH:MM" 晚於標準時間 + 10分鐘 Buffer
            # 注意：這裡不處理跨日日期的比對，僅比對時間字串，適用於大部分場景
            buffer_min = 10
            
            # 將時間轉為當日分鐘數進行比較，避免跨日問題複雜化
            def to_mins(t): return t.hour * 60 + t.minute
            
            first_mins = to_mins(first_in)
            std_mins = to_mins(std_in_t)
            
            # 特殊處理：如果標準上班是大夜 (例如 00:00)，而打卡是 23:50 (前一天)
            # 這裡簡化：只針對 "晚於" 進行遲到判斷
            
            # 如果是跨日班 (例如 21:00 上班)，且打卡是 02:00 (凌晨)，這絕對遲到
            # 但如果是 20:55 打卡，則正常
            
            # 這裡採用簡單比對：若依靠 default_shift，則直接比對時間差
            # 若無 default_shift，因為已經落入 range，通常不會遲到太誇張
            
            if shift_conf and first_in:
                std_in = datetime.strptime(shift_conf['start_time'], "%H:%M").time()
                # 將時間轉為分鐘數比較
                chk_min = first_in.hour * 60 + first_in.minute
                std_min = std_in.hour * 60 + std_in.minute
                # 給予 30 分鐘緩衝
                if chk_min > (std_min + 30):
                    status_list.append("遲到")

        # 3. 工時計算 (核心)
        work_hours = 0.0
        if last_out:
            duration = (last_out - first_in).total_seconds() / 3600.0
            
            # 規則：滿 4 小時扣 1 小時休息
            if duration >= 5.0:
                work_hours = duration - 1.0
            else:
                work_hours = duration
            
            if work_hours < 0: work_hours = 0
            
            # 簡單防呆：早退檢查
            if work_hours < 8.0:
                 status_list.append("早退/工時不足")

        final_status = ", ".join(status_list) if status_list else "正常"
        
        return {
            'date': date_str,
            'shift': shift_name,
            'in': first_in,
            'out': last_out,
            'hours': round(work_hours, 2),
            'status': final_status
        }