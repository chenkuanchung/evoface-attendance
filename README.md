# EvoFace - 企業級 AI 人臉辨識考勤系統
### Enterprise AI Face Recognition Attendance System

EvoFace 是一套整合 **即時人臉辨識**、**活體防偽偵測** 與 **自動化考勤計算** 的全方位解決方案。
系統採用 InsightFace (ArcFace) 作為辨識核心，並結合 Silent-Face (MiniFASNet) 進行防照片/影片攻擊，適合中小企業進行打卡管理。

EvoFace is a comprehensive solution integrating **Real-time Face Recognition**, **Liveness Detection**, and **Automated Attendance Calculation**.
Powered by **InsightFace (ArcFace)** for recognition and **Silent-Face (MiniFASNet)** for anti-spoofing, it effectively defends against photo/video attacks, making it an ideal attendance management system for SMBs.

## 🌟 核心功能 (Key Features)

### 1. 高精度人臉辨識與活體防禦
- **雙重引擎**：使用 MediaPipe 進行快速人臉定位，搭配 InsightFace (Buffalo_L) 提取特徵。
- **強健的遮擋適應性 (Occlusion Robustness)**：
  - **口罩/眼鏡支援**：經參數調校，系統可在佩戴眼鏡或口罩的情況下完成辨識，但註冊後的首次辨識在**不配戴口罩**的情況下較容易成功。
  - **智慧演進保護**：當偵測到面部有遮擋（如口罩）導致信心度下降，但仍高於最低辨識門檻時，系統會**允許打卡但暫停特徵更新 (Feature Update)**，防止遮擋的特徵汙染資料庫中的演進特徵。
- **活體檢測 (Anti-Spoofing)**：內建 Silent-Face 活體分析引擎，有效防禦手機翻拍、照片列印等攻擊。
- **動態特徵演進 (Dynamic Evolution)**：系統會隨著員工日常打卡，自動更新人臉特徵模型（權重融合），適應髮型或樣貌的微小變化。

### 2. 彈性考勤邏輯
- **三班制支援**：內建早班、晚班、大夜班設定，支援 **跨日打卡** (例如 16:00 上班，隔日 01:00 下班)。
- **智慧工時計算**：
  - 自動判斷遲到 (緩衝 30 分鐘)。
  - 符合勞基法邏輯的休息扣除 (滿 5 小時扣 1 小時休息，加班不另扣除休息時間)。
  - 每日 04:00 自動換日切結。

### 3. 雙平台整合介面
- **管理端 (Admin Dashboard - PySide6)**：
  - 員工資料註冊 (**支援照片檔案上傳，不須請員工另行現場拍照**)。
  - 考勤異常補登簽核 (註：正常商業邏輯應寄送 Email 由主管核准，此處簡化由管理員直接核准)。
  - 考勤報表查詢與 Excel 匯出。
- **員工端 (Web Portal - Streamlit)**：
  - 個人考勤紀錄查詢。
  - 考勤報表查詢與 Excel 匯出。
  - 線上申請補登 (忘記打卡/請假)。

---

## 🛠️ 安裝與環境設定 (Installation)

### 1. 系統需求
- Python 3.8+
- 建議使用 NVIDIA GPU (需安裝 CUDA) 以獲得最佳效能，亦可使用 CPU 模式。
- 網路攝影機 (Webcam)

### 2. 安裝步驟

```bash
# 1. 下載專案
git clone https://github.com/chenkuanchung/evoface-attendance.git
cd evoface-attendance

# 2. 建立虛擬環境 (建議)
python -m venv .venv
# Windows 啟用:
.venv\Scripts\activate
# Mac/Linux 啟用:
source .venv/bin/activate

# 3. 安裝依賴套件
pip install -r requirements.txt
```
---

## 🚀 快速啟動 (Usage)

本系統分為 **前台打卡端** 與 **後台管理端**，建議使用內建腳本啟動。

### 🟢 啟動考勤主系統 (打卡鐘 + 員工網頁)
執行 `Start_System.bat`
- 這將同時啟動：
  1. **人臉辨識主視窗**：員工打卡用。
  2. **員工考勤查詢網頁**：
     - **本機操作**：瀏覽器會自動開啟 (或輸入 `http://localhost:8501`)。
     - **區網查詢**：同網域內的其他電腦/手機，可透過 `http://<主機IP>:8501` 進行遠端連線查詢。

### 🔵 啟動管理後台 (Admin)
執行 `Start_Admin.bat`
- 進入管理介面進行員工註冊、班表設定與報表匯出。
- **首次使用請先進入後台註冊員工資料。**

### 🟠 資料備份 (Backup)
執行 `Backup_Data.bat`
- 系統會自動將資料庫 (`attendance.db`) 複製到 `backup/` 資料夾，並加上日期戳記，防止資料遺失。
- **建議定期執行此腳本 (例如每週一次)。**

---

## ⚙️ 參數設定 (Configuration)

所有核心參數皆位於 `config.yaml`，可依場域需求調整：
```yaml
system:
  device_mode: "auto"       # auto/gpu/cpu (推論硬體設定)
  camera_index: 0           # 攝影機編號 (若抓不到畫面可改為 1)

attendance:
  debounce_minutes: 1       # 防止重複打卡間隔 (分)
  day_cutoff: "04:00"       # 每日換日結算點

# 三班制定義 (Format: HH:MM)
# 邏輯：一班 9 小時 (含休 1 小時)。
shifts:
  morning:
    name: "早班"
    start_time: "08:00"  # 標準上班
    end_time: "17:00"    # 標準下班
    range_start: "05:00" # 自動判斷區間起
    range_end: "11:00"   # 自動判斷區間止
  
  evening:
    name: "晚班"
    start_time: "16:00"
    end_time: "01:00"    # 跨日
    range_start: "13:00"
    range_end: "19:00"

  night:
    name: "大夜班"
    start_time: "00:00"
    end_time: "09:00"
    range_start: "21:00"
    range_end: "03:00"

thresholds:
  recognition_confidence: 0.5  # 辨識門檻
  texture_liveness: 0.98       # 活體防偽嚴格度 (越高越嚴)
  warning_base_score: 0.3      # 相似度過低警告
  evolution_min_base: 0.5      # 特徵演進最低門檻

recognition:
  min_face_ratio: 0.2          # 最小人臉佔比 (太遠不偵測)
```

---

## 📂 專案結構

```plaintext
evoface-attendance/
├── data/
│   ├── attendance.db          # SQLite 資料庫 (自動生成)
│   ├── faces/                 # 員工註冊底圖
│   └── logs/                  # 打卡現場快照
├── models/
│   ├── liveness/              # Silent-Face 活體偵測模型 (.onnx)
│   └── face_landmarker.task   # MediaPipe 人臉關鍵點模型
├── src/
│   ├── core/                  # [核心] 系統邏輯層
│   │   ├── calculator.py      # 考勤與工時計算邏輯
│   │   ├── database.py        # 資料庫 CRUD 操作
│   │   ├── detector.py        # MediaPipe 人臉偵測與追蹤
│   │   ├── liveness_engine.py # Silent-Face 活體防偽引擎
│   │   └── recognizer.py      # ArcFace 辨識與特徵演進
│   ├── ui/                    # [介面] PySide6 視窗
│   │   ├── admin_window.py    # 後台管理介面
│   │   └── main_window.py     # 前台打卡主視窗
│   └── utils/                 # [工具] 通用模組
│       ├── image_tool.py      # 影像對齊與裁切工具
│       └── voice.py           # 語音提示功能 (TTS)
├── admin.py                   # [入口] 管理後台啟動檔
├── main.py                    # [入口] 打卡系統啟動檔
├── webapp.py                  # [入口] 員工網頁端 (Streamlit)
├── config.yaml                # 系統全域設定檔
├── requirements.txt           # Python 依賴套件清單
├── Start_System.bat           # [腳本] 一鍵啟動系統
├── Start_Admin.bat            # [腳本] 一鍵啟動後台
├── Backup_Data.bat            # [腳本] 資料庫備份工具
├── README.md                  # 專案說明文件
└── LICENSE                    # 授權條款
```

---

## 📝 授權與聲明

本專案僅供學術研究與內部使用。

- Face Detection model based on Google MediaPipe.

- Anti-Spoofing model based on MiniFASNet (Silent-Face).

- Recognition model based on InsightFace.

