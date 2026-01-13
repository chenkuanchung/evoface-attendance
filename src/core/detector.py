import cv2
import mediapipe as mp
import numpy as np
import yaml
from src.utils.image_tool import ImagePreprocessor 
from src.core.liveness_engine import SilentFaceAnalyzer

class FaceDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化模型與工具
        self.img_tool = ImagePreprocessor() 
        self.silent_face_analyzer = SilentFaceAnalyzer(config_path=config_path)
        
        # MediaPipe 初始化 (省略部分重複代碼，保持與您原始檔案結構一致)
        # ... 
        
        self.texture_threshold = self.config.get('thresholds', {}).get('texture_liveness', 0.95)
        self.is_locked = False
        self.texture_pass_count = 0 
        self.REQUIRED_PASS_FRAMES = 10

    def check_mask_status(self, landmarks, frame_h, frame_w):
        """
        簡單的口罩判斷邏輯：檢查鼻子與嘴角關鍵點的偵測信心或位置
        (實務上建議使用專門的分類器，此處示範 logic-based 判斷)
        """
        # 如果鼻子 (index 4) 或 嘴巴週邊點位在畫面外或異常偏移，則視為口罩遮擋
        nose = landmarks[4]
        if nose.y > 0.9 or nose.y < 0.1: # 範例判斷
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
        
        # 1. 提取 5 個核心對齊點 [左瞳, 右瞳, 鼻尖, 左嘴角, 右嘴角]
        # MediaPipe 索引：左眼(468), 右眼(473), 鼻尖(4), 左嘴角(61), 右嘴角(291)
        landmarks_5pt = [
            [points[468].x * w, points[468].y * h],
            [points[473].x * w, points[473].y * h],
            [points[4].x * w, points[4].y * h],
            [points[61].x * w, points[61].y * h],
            [points[291].x * w, points[291].y * h]
        ]
        
        # 2. 判斷口罩狀態
        is_masked = self.check_mask_status(points, h, w)

        if not self.is_locked:
            # 3. 執行仿射變換與影像強化 (第一步改寫重點)
            aligned_face = self.img_tool.align_face(frame, landmarks_5pt, is_masked=is_masked)
            processed_face = self.img_tool.enhance_face(aligned_face)
            
            # 4. 活體檢測 (使用處理過的標準影像)
            avg_brightness = self.img_tool.get_brightness(processed_face)
            current_threshold = self.texture_threshold
            if avg_brightness < 70: current_threshold -= 0.05

            texture_score = self.silent_face_analyzer.predict(processed_face)
            
            if texture_score >= current_threshold:
                self.texture_pass_count += 1
            else:
                self.texture_pass_count = 0 
            
            if self.texture_pass_count >= self.REQUIRED_PASS_FRAMES:
                self.is_locked = True
        
        # ... 回傳邏輯 [cite: 36]
        return "SUCCESS", {
            "bbox": [0,0,0,0], # 簡化
            "is_live": self.is_locked,
            "face_img": processed_face if self.is_locked else None # 傳遞給 recognizer
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