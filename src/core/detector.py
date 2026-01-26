import cv2
import mediapipe as mp
import numpy as np
import yaml
from src.utils.image_tool import ImagePreprocessor 
from src.core.liveness_engine import SilentFaceAnalyzer

class FaceDetector:
    def __init__(self, config_path="config.yaml"):
        # 1. 載入設定檔
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 2. 初始化影像工具與活體分析器
        self.img_tool = ImagePreprocessor() 
        self.silent_face_analyzer = SilentFaceAnalyzer(config_path=config_path)
        
        # 3. 初始化 MediaPipe Face Landmarker
        model_path = self.config['database']['model_path']
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_face_detection_confidence=self.config['thresholds']['detection_confidence'],
            min_face_presence_confidence=self.config['thresholds']['detection_confidence'],
            min_tracking_confidence=self.config['thresholds']['tracking_confidence'],
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        
        # 4. 讀取門檻值與狀態控制
        self.texture_threshold = self.config.get('thresholds', {}).get('texture_liveness', 0.95)
        # 讀取使用者指定的佔比門檻 (預設 0.4)
        self.min_face_ratio = self.config.get('recognition', {}).get('distance_threshold', 0.4)
        
        self.is_locked = False
        self.texture_pass_count = 0 
        self.REQUIRED_PASS_FRAMES = 10

    def process(self, frame):
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            
            if not result.face_landmarks:
                self.reset_liveness()
                return "NO_FACE", None
            
            if len(result.face_landmarks) > 1:
                self.reset_liveness()
                return "MULTIPLE_FACES", None
                
            points = result.face_landmarks[0]
            x_coords = [p.x * w for p in points]
            y_coords = [p.y * h for p in points]
            
            # 原始的 BBox
            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            actual_bbox = [x1, y1, x2, y2]

            # === 人臉佔比檢查邏輯 ===
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            ratio = face_area / frame_area
            
            # 如果佔比小於設定值，則回傳過小狀態
            if ratio < self.min_face_ratio:
                self.reset_liveness()
                return "FACE_TOO_SMALL", {"bbox": actual_bbox}
            # ==============================

            # === BBox Padding 邏輯 ===
            padding_ratio = 0.6
            face_w = x2 - x1
            face_h = y2 - y1
            
            pad_w = int(face_w * padding_ratio)
            pad_h = int(face_h * padding_ratio)
            
            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(w, x2 + pad_w)
            py2 = min(h, y2 + pad_h)
            
            recognition_face = None
            if self.is_locked:
                landmarks_5pt = [
                    [points[468].x * w, points[468].y * h], [points[473].x * w, points[473].y * h],
                    [points[4].x * w, points[4].y * h], [points[61].x * w, points[61].y * h],
                    [points[291].x * w, points[291].y * h]
                ]
                aligned_face = self.img_tool.align_face(frame, landmarks_5pt, is_masked=False)
                recognition_face = aligned_face 

            if not self.is_locked:
                face_roi = frame[py1:py2, px1:px2]
                if face_roi.size > 0:
                    raw_score = self.silent_face_analyzer.predict(face_roi)
                    
                    if raw_score >= self.texture_threshold:
                        self.texture_pass_count += 1
                    else:
                        self.texture_pass_count = 0 
                    
                    if self.texture_pass_count >= self.REQUIRED_PASS_FRAMES:
                        self.texture_pass_count = self.REQUIRED_PASS_FRAMES
                        self.is_locked = True
            
            display_score = self.texture_pass_count / self.REQUIRED_PASS_FRAMES

            return "SUCCESS", {
                "bbox": actual_bbox,
                "is_live": self.is_locked,
                "texture_score": display_score,
                "face_img": recognition_face if self.is_locked else None 
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
            # 使用實際計算出的 BBox
            cv2.rectangle(frame, (res['bbox'][0], res['bbox'][1]), (res['bbox'][2], res['bbox'][3]), color, 2)
            
            # 顯示活體百分比
            score_text = f"Liveness: {res['texture_score']*100:.1f}%"
            cv2.putText(frame, score_text, (res['bbox'][0], res['bbox'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        cv2.imshow("EvoFace Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()