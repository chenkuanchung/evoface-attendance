# webapp.py
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta, time
from src.core.database import AttendanceDB
from src.core.calculator import AttendanceCalculator

# 1. 網頁基本設定
st.set_page_config(
    page_title="EvoFace 員工考勤中心", 
    page_icon="📅", 
    layout="centered"
)

# 2. 初始化核心模組 (使用 Session State 避免重複載入)
if 'db' not in st.session_state:
    st.session_state.db = AttendanceDB()
if 'calc' not in st.session_state:
    st.session_state.calc = AttendanceCalculator()

# --- 介面函式 ---

def login_page():
    """登入畫面"""
    st.title("🔐 EvoFace 員工考勤系統")
    st.markdown("請輸入您的員工編號與密碼進行登入。")
    
    with st.form("login_form"):
        emp_id = st.text_input("員工編號 (ID)")
        password = st.text_input("密碼 (預設為 ID)", type="password")
        submit = st.form_submit_button("登入系統", width="stretch")
        
        if submit:
            if st.session_state.db.verify_password(emp_id, password):
                # 登入成功，記錄 Session
                st.session_state.logged_in = True
                st.session_state.user_id = emp_id
                
                # 讀取員工基本資料
                all_emps = st.session_state.db.load_all_employees()
                if emp_id in all_emps:
                    st.session_state.user_name = all_emps[emp_id]['name']
                    st.session_state.default_shift = all_emps[emp_id].get('default_shift')
                else:
                    st.session_state.user_name = "員工"
                    st.session_state.default_shift = None
                    
                st.success("登入成功！")
                st.rerun()
            else:
                st.error("❌ 帳號或密碼錯誤")

def calculate_period_stats(start_date, end_date):
    """計算指定區間內的考勤資料"""
    data_list = []
    current_d = start_date
    
    # 讀取所有 Logs (為了效能，實務上可優化 SQL 範圍查詢)
    # 這裡我們為了確保 "跨日" 邏輯正確，前後多抓一天 buffer
    query_start = datetime.combine(start_date - timedelta(days=1), time(0,0))
    query_end = datetime.combine(end_date + timedelta(days=2), time(0,0))
    
    # 從 DB 撈取該員工的所有原始打卡時間
    raw_logs_str = st.session_state.db.get_logs_by_range(query_start, query_end, st.session_state.user_id)
    
    # 字串轉 datetime
    raw_logs_dt = []
    for t_str in raw_logs_str:
        try:
            raw_logs_dt.append(datetime.strptime(t_str.split('.')[0], '%Y-%m-%d %H:%M:%S'))
        except:
            pass
            
    # 逐日計算
    while current_d <= end_date:
        # 使用 Calculator 的 "日切" 邏輯，篩選出屬於 current_d 的打卡紀錄
        daily_logs = []
        for log_dt in raw_logs_dt:
            if st.session_state.calc.get_logical_day(log_dt) == current_d:
                daily_logs.append(log_dt)
        
        # 呼叫核心計算
        stats = st.session_state.calc.calculate_daily_stats(
            current_d.strftime("%Y-%m-%d"), 
            daily_logs, 
            st.session_state.default_shift
        )
        
        # 整理成表格顯示用的格式
        data_list.append({
            "日期": stats['date'],
            "班別": stats['shift'],
            "上班卡": stats['in'].strftime("%H:%M") if stats['in'] else "--:--",
            "下班卡": stats['out'].strftime("%H:%M") if stats['out'] else "--:--",
            "工時": f"{stats['hours']} hr",
            "狀態": stats['status']
        })
        
        current_d += timedelta(days=1)
        
    return pd.DataFrame(data_list)

def main_dashboard():
    """登入後的主控台"""
    # 側邊欄：個人資訊
    with st.sidebar:
        st.title(f"👤 {st.session_state.user_name}")
        st.caption(f"ID: {st.session_state.user_id}")
        st.divider()
        if st.button("🚪 登出系統", width="stretch"):
            st.session_state.logged_in = False
            st.rerun()

    st.subheader(f"👋 {st.session_state.user_name}，歡迎使用考勤查詢系統!")

    tab1, tab2 = st.tabs(["📅 我的考勤紀錄", "📝 申請補登/請假"])

    # --- Tab 1: 考勤查詢 ---
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            # 預設查詢最近 7 天
            today = date.today()
            date_range = st.date_input(
                "選擇查詢區間",
                (today - timedelta(days=6), today),
                max_value=today
            )
        
        if len(date_range) == 2:
            start_d, end_d = date_range
            df = calculate_period_stats(start_d, end_d)
            
            # 樣式設定：異常顯示紅色背景
            def highlight_status(val):
                if '正常' in val:
                    return 'color: green; font-weight: bold'
                elif '缺' in val or '異常' in val:
                    return 'color: red; font-weight: bold'
                elif '遲到' in val or '早退' in val:
                    return 'color: orange; font-weight: bold'
                return ''

            st.dataframe(
                df.style.map(highlight_status, subset=['狀態']),
                width="stretch",
                height=400
            )
        else:
            st.info("請選擇完整的起始與結束日期。")

    # --- Tab 2: 補登申請 ---
    with tab2:
        st.markdown("#### 📝 考勤異常補登申請")
        st.info("若您發現考勤狀態異常（如忘記打卡），請在此提交申請。申請後需等待管理員核准。")
        
        with st.form("request_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                req_date = st.date_input("補登日期", max_value=date.today())
                req_type = st.selectbox("補登類型", ["上班卡 (IN)", "下班卡 (OUT)"])
            with col_b:
                req_time = st.time_input("正確時間")
                
            req_reason = st.text_area("申請理由", placeholder="例如：忘記打卡、外出公務...")
            
            submitted = st.form_submit_button("送出申請", width="stretch")
            
            if submitted:
                if not req_reason:
                    st.warning("請填寫申請理由。")
                else:
                    success = st.session_state.db.create_request(
                        st.session_state.user_id,
                        req_date.strftime("%Y-%m-%d"),
                        "IN" if "上班" in req_type else "OUT",
                        req_time.strftime("%H:%M"),
                        req_reason
                    )
                    if success:
                        st.success("✅ 申請已送出！請等待管理員審核。")
                    else:
                        st.error("❌ 系統忙碌中，請稍後再試。")

# 3. 路由控制
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    main_dashboard()