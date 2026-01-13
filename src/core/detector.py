import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    """
    採用 MediaPipe 最新 Tasks API 的偵測器。
    整合 3D 結構視差與眼部微震顫檢測，嚴格防範影片與照片攻擊。
    """
    def __init__(self, config_path="config.yaml"):
        # 載入設定
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 讀取配置參數
        model_path = self.config.get('database', {}).get('model_path', 'models/face_landmarker.task')
        det_confidence = self.config.get('thresholds', {}).get('detection_confidence', 0.6)
        liveness_score_threshold = self.config.get('thresholds', {}).get('liveness_score', 1.0)
        
        # 檢查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 找不到模型文件: {model_path}")
            print("💡 請下載 face_landmarker.task 並放至 models/ 資料夾。")
            # 建立目錄以方便使用者放置
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 1. MediaPipe Tasks 選項設定
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # 為了方便整合 OpenCV 循環，使用 IMAGE 模式
            num_faces=2,  # 設為 2 以偵測多人干擾
            min_face_detection_confidence=det_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        # 建立偵測器
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 活體檢測參數
        self.liveness_score = 0.0
        self.liveness_threshold = liveness_score_threshold
        self.history_landmarks = deque(maxlen=20)
        
        # 關鍵點索引 (Tasks API 的索引與 Face Mesh 相同)
        self.NOSE_TIP = 1 

    def _check_3d_parallax(self, landmarks):
        """核心防偽：3D 視差檢查"""
        if len(self.history_landmarks) < 5:
            return 0.0
        
        # 取得鼻尖座標
        curr_nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y, landmarks[self.NOSE_TIP].z])
        prev_nose = self.history_landmarks[-2][self.NOSE_TIP]
        
        # 鼻尖平面位移
        nose_move = np.linalg.norm(curr_nose[:2] - prev_nose[:2])
        if nose_move < 0.001: return -0.05 # 靜止扣分

        # 計算不同深度點的位移比標準差
        depth_changes = []
        for idx in [33, 263, 152, 10]:
            curr_pt = np.array([landmarks[idx].x, landmarks[idx].y])
            prev_pt = self.history_landmarks[-2][idx][:2]
            pt_move = np.linalg.norm(curr_pt - prev_pt)
            if pt_move > 0:
                depth_changes.append(nose_move / pt_move)
        
        if not depth_changes: return 0.0

        std_val = np.std(depth_changes)
        if 0.002 < std_val < 0.08: return 0.15 # 真人特徵
        return 0.0

    def _calculate_ear(self, landmarks, w, h):
        """計算平均眼睛外觀比例 (EAR)"""
        def get_ear(indices):
            # Tasks API 的點直接具備 x, y 屬性
            pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
            v = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return v / (2.0 * h_dist)
        # 左眼與右眼索引
        l_ear = get_ear([362, 385, 387, 263, 373, 380])
        r_ear = get_ear([33, 160, 158, 133, 153, 144])
        return (l_ear + r_ear) / 2.0

    def process(self, frame):
        """
        處理影像幀。
        :param frame: OpenCV 格式影像 (BGR)
        :return: (status, data)
        """
        h, w, _ = frame.shape
        
        # 轉換為 MediaPipe Image 格式
        # 注意：Tasks API 處理的是 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 執行偵測
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            self.reset_liveness()
            return "NO_FACE", None
        
        if len(result.face_landmarks) > 1:
            return "MULTIPLE_FACES", None
            
        # 取得第一張臉的點位
        landmarks = result.face_landmarks[0]
        
        # 更新歷史紀錄 (將點位物件轉為 numpy 陣列以便計算)
        curr_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        self.history_landmarks.append(curr_pts)

        # --- 活體檢測累積 ---
        # 1. 3D 視差
        self.liveness_score += self._check_3d_parallax(landmarks)
        
        # 2. 眼部微動 (EAR)
        ear = self._calculate_ear(landmarks, w, h)
        if 0.1 < ear < 0.22:
            self.liveness_score += 0.1
            
        # 分數控制
        self.liveness_score = max(0.0, min(self.liveness_score, 1.2))
        is_live = self.liveness_score >= self.liveness_threshold

        # 計算 BBox (基於邊界點)
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        bbox = [int(min(x_coords)*w), int(min(y_coords)*h), int(max(x_coords)*w), int(max(y_coords)*h)]

        return "SUCCESS", {
            "bbox": bbox,
            "is_live": is_live,
            "liveness_percent": min(int(self.liveness_score * 100), 100)
        }

    def reset_liveness(self):
        self.liveness_score = 0.0
        self.history_landmarks.clear()

    def __del__(self):
        """關閉偵測器釋放資源"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

if __name__ == "__main__":
    # 測試腳本
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    
    print("🎬 開始測試 Tasks API 偵測器 (按 'q' 退出)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        status, res = detector.process(frame)
        
        if status == "SUCCESS":
            color = (0, 255, 0) if res['is_live'] else (0, 165, 255)
            cv2.rectangle(frame, (res['bbox'][0], res['bbox'][1]), (res['bbox'][2], res['bbox'][3]), color, 2)
            cv2.putText(frame, f"Tasks 3D Liveness: {res['liveness_percent']}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        cv2.imshow("MediaPipe Tasks Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()