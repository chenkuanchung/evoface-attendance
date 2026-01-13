import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .liveness_engine import SilentFaceAnalyzer

class FaceDetector:
    """
    採用 MediaPipe 最新 Tasks API 的偵測器。
    整合 3D 結構視差、眼部微震顫以及 Silent-Face 深度學習紋理檢測，
    雙重過濾影片、照片與 3D 翻拍攻擊。
    """
    def __init__(self, config_path="config.yaml"):
        # 載入設定
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 讀取配置參數
        model_path = self.config.get('database', {}).get('model_path', 'models/face_landmarker.task')
        det_confidence = self.config.get('thresholds', {}).get('detection_confidence', 0.6)
        liveness_score_threshold = self.config.get('thresholds', {}).get('liveness_score', 1.0)
        # 讀取 Silent-Face 的門檻值 (預設 0.9)
        self.texture_threshold = self.config.get('thresholds', {}).get('texture_liveness', 0.9)
        
        # 檢查模型文件
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 1. MediaPipe Tasks 設定
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=det_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 2. 初始化 Silent-Face 紋理分析器 (傳入 config_path 以讀取 device_mode)
        self.silent_face_analyzer = SilentFaceAnalyzer(config_path=config_path)
        
        # 活體檢測參數
        self.liveness_score = 0.0
        self.liveness_threshold = liveness_score_threshold
        self.history_landmarks = deque(maxlen=20)
        self.is_locked = False
        self.NOSE_TIP = 1 

    def _check_3d_parallax(self, landmarks):
        """核心防偽：3D 視差檢查"""
        if len(self.history_landmarks) < 5:
            return 0.0
        
        curr_nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y, landmarks[self.NOSE_TIP].z])
        prev_nose = self.history_landmarks[-2][self.NOSE_TIP]
        
        nose_move = np.linalg.norm(curr_nose[:2] - prev_nose[:2])
        if nose_move < 0.001: return -0.05 # 靜止扣分

        depth_changes = []
        for idx in [33, 263, 152, 10]:
            curr_pt = np.array([landmarks[idx].x, landmarks[idx].y])
            prev_pt = self.history_landmarks[-2][idx][:2]
            pt_move = np.linalg.norm(curr_pt - prev_pt)
            if pt_move > 0:
                depth_changes.append(nose_move / pt_move)
        
        if not depth_changes: return 0.0

        std_val = np.std(depth_changes)
        # 強化門檻：避開螢幕晃動的小位移
        if 0.005 < std_val < 0.08: return 0.15 
        return 0.0

    def _calculate_ear(self, landmarks, w, h):
        """計算平均眼睛外觀比例 (EAR)"""
        def get_ear(indices):
            pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
            v = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return v / (2.0 * h_dist)
        l_ear = get_ear([362, 385, 387, 263, 373, 380])
        r_ear = get_ear([33, 160, 158, 133, 153, 144])
        return (l_ear + r_ear) / 2.0

    def process(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            self.reset_liveness()
            return "NO_FACE", None
        
        if len(result.face_landmarks) > 1:
            return "MULTIPLE_FACES", None
            
        landmarks = result.face_landmarks[0]
        curr_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        self.history_landmarks.append(curr_pts)

        # 計算 BBox (用於後續 Silent-Face 裁剪)
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        bbox = [int(min(x_coords)*w), int(min(y_coords)*h), int(max(x_coords)*w), int(max(y_coords)*h)]

        # --- 雙重活體檢測邏輯 ---
        if not self.is_locked:
            # 1. 第一層：幾何動作分析 (CPU 輕量計算)
            self.liveness_score += self._check_3d_parallax(landmarks)
            ear = self._calculate_ear(landmarks, w, h)
            if 0.1 < ear < 0.22:
                self.liveness_score += 0.1
            
            self.liveness_score = max(0.0, min(self.liveness_score, 1.2))

            # 2. 第二層：當幾何分數達標，觸發深度學習紋理檢查
            if self.liveness_score >= self.liveness_threshold:
                # 裁剪臉部影像
                x1, y1, x2, y2 = bbox
                # 確保裁剪座標不超出範圍
                face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if face_crop.size > 0:
                    # 執行 Silent-Face 推論 (此部分會讀取 config.yaml 決定使用 CPU/GPU)
                    texture_score = self.silent_face_analyzer.predict(face_crop)
                    
                    if texture_score >= self.texture_threshold:
                        self.is_locked = True # 雙重通過，鎖定
                    else:
                        # 幾何雖過但紋理不合格 (手機錄影常見特徵)
                        self.liveness_score = 0.4 # 強制降分重測
                        self.is_locked = False
        else:
            self.liveness_score = self.liveness_threshold

        return "SUCCESS", {
            "bbox": bbox,
            "is_live": self.is_locked,
            "liveness_percent": min(int(self.liveness_score * 100), 100)
        }

    def reset_liveness(self):
        self.liveness_score = 0.0
        self.is_locked = False
        self.history_landmarks.clear()

    def __del__(self):
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