import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from liveness_engine import SilentFaceAnalyzer

class FaceDetector:
    """
    極簡優化 + CLAHE 強化版偵測器。
    MediaPipe：負責穩定的人臉追蹤與裁剪。
    CLAHE：強化低畫質鏡頭下的皮膚紋理細節。
    Silent-Face：負責核心真偽判定。
    """
    def __init__(self, config_path="config.yaml"):
        # 載入設定
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 讀取配置參數
        model_path = self.config.get('database', {}).get('model_path', 'models/face_landmarker.task')
        det_confidence = self.config.get('thresholds', {}).get('detection_confidence', 0.6)
        
        # 針對低畫質鏡頭，門檻建議設在 0.85 ~ 0.9 之間
        self.texture_threshold = self.config.get('thresholds', {}).get('texture_liveness', 0.95)
        
        # 1. MediaPipe Tasks 設定 (僅用於定位)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=det_confidence
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 2. 初始化 Silent-Face 紋理分析器
        self.silent_face_analyzer = SilentFaceAnalyzer(config_path=config_path)
        
        # 3. 初始化 CLAHE 工具 (用於預處理)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 狀態變數
        self.is_locked = False
        self.texture_pass_count = 0  # 連續通過計數器
        self.REQUIRED_PASS_FRAMES = 10 # 需連續通過 10 幀

    def _preprocess_face(self, face_crop):
        """影像預處理：強化紋理以補償筆電鏡頭的不足"""
        try:
            # 1. 轉到 LAB 色彩空間強化亮度對比
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
            
            # 2. 輕微銳化：抵消筆電鏡頭過度降噪導致的「油畫感」
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_bgr = cv2.filter2D(enhanced_bgr, -1, kernel)
            
            return enhanced_bgr
        except:
            return face_crop

    def process(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            self.reset_liveness()
            return "NO_FACE", None
            
        landmarks = result.face_landmarks[0]
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x1, y1, x2, y2 = int(min(x_coords)*w), int(min(y_coords)*h), int(max(x_coords)*w), int(max(y_coords)*h)
        bbox = [x1, y1, x2, y2]

        if not self.is_locked:
            face_w = x2 - x1
            if face_w < (w * 0.2):
                return "TOO_FAR", {"bbox": bbox}

            face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if face_crop.size > 0:
                # --- 執行 CLAHE 與銳化預處理 ---
                processed_face = self._preprocess_face(face_crop)
                
                # 計算亮度進行動態補償
                avg_brightness = np.mean(cv2.cvtColor(processed_face, cv2.COLOR_BGR2GRAY))
                current_threshold = self.texture_threshold
                if avg_brightness < 70: # 環境太暗時自動微降門檻
                    current_threshold -= 0.05

                texture_score = self.silent_face_analyzer.predict(processed_face)
                
                if texture_score >= current_threshold:
                    self.texture_pass_count += 1
                else:
                    self.texture_pass_count = 0 
                
                if self.texture_pass_count >= self.REQUIRED_PASS_FRAMES:
                    self.is_locked = True
            
        progress = min(int((self.texture_pass_count / self.REQUIRED_PASS_FRAMES) * 100), 100)
        if self.is_locked: progress = 100

        return "SUCCESS", {
            "bbox": bbox,
            "is_live": self.is_locked,
            "liveness_percent": progress
        }

    def reset_liveness(self):
        self.is_locked = False
        self.texture_pass_count = 0

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
            # 綠框代表真人鎖定，橘框代表判定中
            color = (0, 255, 0) if res['is_live'] else (0, 165, 255)
            cv2.rectangle(frame, (res['bbox'][0], res['bbox'][1]), (res['bbox'][2], res['bbox'][3]), color, 2)
            cv2.putText(frame, f"Liveness: {res['liveness_percent']}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        cv2.imshow("MediaPipe Tasks Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()