import cv2
import numpy as np
import onnxruntime as ort
import yaml
import os

class SilentFaceAnalyzer:
    def __init__(self, config_path="config.yaml"):
        # 1. 讀取設定檔
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        # 2. 決定裝置模式 (系統預設 auto)
        device_mode = self.config.get('system', {}).get('device_mode', 'auto')
        model_path = self.config.get('database', {}).get('liveness_model', 'models/liveness/2.7_80x80_MiniFASNetV2.onnx')
        
        # 3. 配置推論後端 (Providers)
        available_providers = ort.get_available_providers()
        
        if device_mode == "gpu" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        elif device_mode == "cpu":
            providers = ["CPUExecutionProvider"]
        else: # auto 模式
            # 優先使用 GPU，若無則回退至 CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

        # 初始化 Session
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, face_img):
        # 影像預處理邏輯
        face_img = cv2.resize(face_img, (80, 80))
        face_img = face_img.astype(np.float32)
        
        # 轉換為 NCHW 格式 (1, 3, 80, 80)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)

        # 執行推論
        preds = self.session.run(None, {self.input_name: face_img})
        
        # Softmax 解析結果
        score = np.exp(preds[0]) / np.sum(np.exp(preds[0]))
        return score[0][1] # 真人機率