import numpy as np
import cv2
import yaml
import insightface
from insightface.app import FaceAnalysis
from src.core.database import AttendanceDB
from src.utils.image_tool import ImagePreprocessor

class FaceRecognizer:
    """
    辨識核心 (V2.1)：支援特徵加權融合、口罩模式對齊與詳細來源追蹤。
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 1. 初始化 InsightFace (Buffalo_L)
        device_mode = self.config.get('system', {}).get('device_mode', 'auto')
        ctx_id = 0 if device_mode == "gpu" else -1
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # 2. 初始化相關組件
        self.db = AttendanceDB(config_path=config_path)
        self.img_tool = ImagePreprocessor()
        
        # 3. 讀取門檻值
        self.rec_threshold = self.config.get('thresholds', {}).get('recognition_confidence', 0.5)
        self.evo_threshold = self.config.get('thresholds', {}).get('evolution_confidence', 0.92)

        # 4. 讀取辨識權重與距離門檻
        self.base_weight = self.config.get('recognition', {}).get('base_weight', 0.4)
        self.dynamic_weight = self.config.get('recognition', {}).get('dynamic_weight', 0.6)
        self.dist_threshold = self.config.get('recognition', {}).get('distance_threshold', 0.4)

    def extract_feature(self, aligned_face):
        """從對齊後的 112x112 影像提取 512 維特徵"""
        faces = self.app.get(aligned_face)
        if not faces:
            return None
        return faces[0].normed_embedding

    def compute_similarity(self, feat1, feat2):
        """計算餘弦相似度"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def identify(self, processed_face):
        """
        執行 1:N 加權比對邏輯 (使用 config.yaml 設定之權重)
        """
        live_feat = self.extract_feature(processed_face)
        if live_feat is None:
            return None, 0.0, False, {}

        all_employees = self.db.load_all_employees()
        best_match_id = None
        max_fused_score = -1.0
        final_details = {}
        
        for emp_id, data in all_employees.items():
            base_feat = data['base']
            dynamic_feat = data['dynamic']
            
            # --- 動態權重融合邏輯 ---
            if dynamic_feat is not None:
                # 使用 config 中的權重進行融合 
                fused_feat = (base_feat * self.base_weight) + (dynamic_feat * self.dynamic_weight)
                # 融合後必須重新歸一化以維持單位向量 
                fused_feat = fused_feat / np.linalg.norm(fused_feat)
            else:
                fused_feat = base_feat
            
            # 計算分數
            fused_score = self.compute_similarity(live_feat, fused_feat)
            base_score = self.compute_similarity(live_feat, base_feat)
            dyn_score = self.compute_similarity(live_feat, dynamic_feat) if dynamic_feat is not None else 0.0

            if fused_score > max_fused_score:
                max_fused_score = fused_score
                best_match_id = emp_id
                final_details = {
                    "base_score": float(base_score),
                    "dynamic_score": float(dyn_score),
                    "fused_score": float(fused_score)
                }

        # 檢查是否達到辨識門檻
        if max_fused_score >= self.rec_threshold:
            should_evolve = max_fused_score >= self.evo_threshold
            return best_match_id, max_fused_score, should_evolve, final_details
        
        return None, max_fused_score, False, final_details

    def process_attendance(self, emp_id, score, should_evolve, face_img, photo_path, details):
        """處理打卡儲存與演進"""
        # 調用資料庫記錄 (注意：需配合稍後第三步修改的 database.py)
        success, message = self.db.add_attendance_log(emp_id, score, photo_path, details)
        
        if success and should_evolve:
            new_feat = self.extract_feature(face_img)
            if new_feat is not None:
                self.db.update_dynamic_feature(emp_id, new_feat) 
                message += " (特徵已進化)"
        
        return success, message
    
if __name__ == "__main__":
    # 測試腳本：驗證辨識邏輯與加權融合
    import os
    
    # 1. 初始化辨識器
    # 確保您的路徑正確，若在 src/core 下執行，可能需要調整 config_path
    recognizer = FaceRecognizer(config_path="config.yaml")
    print("✅ 辨識引擎初始化成功 (Buffalo_L)。")

    # 2. 模擬一張測試影像 (隨機產生或讀取實際檔案)
    # 實務上您應該放一張 112x112 的人臉裁切圖進行測試
    test_img = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print("\n[開始進行 1:N 辨識測試...]")
    
    # 3. 執行辨識
    emp_id, score, evolve, details = recognizer.identify(test_img)

    # 4. 輸出結果分析
    if emp_id:
        print(f"🎯 辨識結果: 員工編號 {emp_id}")
        print(f"📈 最終融合分數 (Fused Score): {score:.4f}")
        print(f"🔍 得分細節: {details}")
        if evolve:
            print("🚀 狀態: 信心度極高，建議觸發特徵演進。")
    else:
        print(f"❌ 辨識失敗: 未達門檻值 (最高得分: {score:.4f})")
        print(f"🔍 嘗試得分詳情: {details}")

    # 5. 測試特徵提取功能
    feat = recognizer.extract_feature(test_img)
    if feat is not None:
        print(f"\n✅ 特徵提取正常，維度: {feat.shape}")