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
        self.is_locked = False
        self.texture_pass_count = 0 
        self.REQUIRED_PASS_FRAMES = 5

    def check_mask_status(self, landmarks, frame_h, frame_w):
        """
        簡單的口罩判斷邏輯：檢查鼻子與中心區域
        """
        # 索引 4 為鼻尖
        nose = landmarks[4]
        if nose.y > 0.9 or nose.y < 0.1:
            return True
        return False

    def process(self, frame):
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            
            if not result.face_landmarks:
                self.reset_liveness()
                return "NO_FACE", None
                
            points = result.face_landmarks[0]
            
            # 1. 座標計算與辨識路徑 (保持原樣，確保辨識品質)
            x_coords = [p.x * w for p in points]
            y_coords = [p.y * h for p in points]
            actual_bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
            is_masked = self.check_mask_status(points, h, w)
            
            landmarks_5pt = [
                [points[468].x * w, points[468].y * h], [points[473].x * w, points[473].y * h],
                [points[4].x * w, points[4].y * h], [points[61].x * w, points[61].y * h],
                [points[291].x * w, points[291].y * h]
            ]
            aligned_face = self.img_tool.align_face(frame, landmarks_5pt, is_masked=is_masked)
            recognition_face = self.img_tool.enhance_face(aligned_face)

            # 2. 活體路徑
            
            # 3. 活體計次邏輯 (進度累加制)
            if not self.is_locked:
                # 執行推論取得原始分數
                raw_score = self.silent_face_analyzer.predict(frame)
                
                # 判斷這一幀是否「通過真人門檻」
                if raw_score >= self.texture_threshold:
                    self.texture_pass_count += 1
                else:
                    # 若有一幀沒過，進度歸零 (確保連續性，增加安全性)
                    self.texture_pass_count = 0 
                
                # 達到 5 幀後鎖定
                if self.texture_pass_count >= self.REQUIRED_PASS_FRAMES:
                    self.texture_pass_count = self.REQUIRED_PASS_FRAMES # 封頂
                    self.is_locked = True
            
            # 4. 計算 UI 顯示分數 (每幀 20%)
            # 就算沒鎖定，也要回傳 0%, 20%, 40%, 60%, 80% 的進度感
            display_score = self.texture_pass_count / self.REQUIRED_PASS_FRAMES

            return "SUCCESS", {
                "bbox": actual_bbox,
                "is_live": self.is_locked,
                "texture_score": display_score, # 回傳 0.0 ~ 1.0 的百分比進度
                "face_img": recognition_face if self.is_locked else None 
            }

    def reset_liveness(self):
        self.is_locked = False
        self.texture_pass_count = 0

    def __del__(self):
        """釋放 MediaPipe 資源"""
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