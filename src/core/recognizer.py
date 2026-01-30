import numpy as np
import cv2
import yaml
#import insightface
from datetime import datetime
from insightface.app import FaceAnalysis
from src.core.database import AttendanceDB
from src.utils.image_tool import ImagePreprocessor

class FaceRecognizer:
    """
    辨識核心：
    1. 修正特徵向量未正規化導致分數暴衝 (15.x) 的問題。
    2. 確保特徵演進時權重正確 (原本因向量過大導致 Soft Update 失效)。
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
        # 啟動時預先載入所有特徵到記憶體
        self.reload_employees()
        
        # 3. 讀取門檻值
        self.rec_threshold = self.config.get('thresholds', {}).get('recognition_confidence', 0.5)
        self.evo_threshold = self.config.get('thresholds', {}).get('evolution_confidence', 0.5)
        self.warn_base_th = self.config.get('thresholds', {}).get('warning_base_score', 0.3)
        self.evo_min_base = self.config.get('thresholds', {}).get('evolution_min_base', 0.5)
        self.evo_min_dyn = self.config.get('thresholds', {}).get('evolution_min_dynamic', 0.85)

        # 4. 讀取辨識權重與距離門檻
        self.base_weight = self.config.get('recognition', {}).get('base_weight', 0.4)
        self.dynamic_weight = self.config.get('recognition', {}).get('dynamic_weight', 0.6)

    def reload_employees(self):
        """
        將所有員工資料轉為 Numpy 矩陣 (Cache)。
        """
        all_data = self.db.load_all_employees()
        
        self.emp_ids = []           # 順序對應的 ID 列表
        self.base_feats = []        # 原始特徵列表
        self.dynamic_feats = []     # 動態特徵列表
        self.has_dynamic_flags = [] # 標記該員工是否有動態特徵
        
        for eid, data in all_data.items():
            self.emp_ids.append(eid)
            self.base_feats.append(data['base'])
            
            # 處理動態特徵：如果沒有，就暫時用 base 填補
            if data['dynamic'] is not None:
                self.dynamic_feats.append(data['dynamic'])
                self.has_dynamic_flags.append(True)
            else:
                self.dynamic_feats.append(data['base'])
                self.has_dynamic_flags.append(False)
                
        # 轉為 Numpy 矩陣，形狀為 (N, 512)
        if self.emp_ids:
            self.base_matrix = np.array(self.base_feats)
            self.dynamic_matrix = np.array(self.dynamic_feats)
            self.has_dynamic_flags = np.array(self.has_dynamic_flags)
        else:
            self.base_matrix = np.empty((0, 512))
            self.dynamic_matrix = np.empty((0, 512))
            self.has_dynamic_flags = np.array([])

        print(f"✅ 特徵庫載入完成，共 {len(self.emp_ids)} 人。")

    def extract_feature(self, aligned_face):
        """
        從已對齊的 112x112 影像中提取特徵向量。
        """
        if aligned_face is None: 
            return None
        
        # 繞過 FaceAnalysis 的封裝，直接取得內部的 ArcFace 辨識模型
        rec_model = self.app.models['recognition'] 
        feat = rec_model.get_feat(aligned_face)
        
        if feat is not None:
            return feat.flatten()
            
        return None

    def compute_similarity(self, feat1, feat2):
        """計算餘弦相似度 (單一比對用)"""
        # 這裡原本就有除以 norm，所以單獨算 base_score 時是對的 (0.54)
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def identify(self, processed_face):
        """
        執行 1:N 加權比對邏輯 (矩陣加速版)
        """
        # 1. 提取鏡頭前的人臉特徵 (原始長度約 20~25)
        live_feat_raw = self.extract_feature(processed_face)
        if live_feat_raw is None:
            return None, 0.0, False, {}, None

        # [CRITICAL FIX] 必須先將 live_feat 正規化 (長度變為 1)
        # 否則矩陣乘法出來的分數會暴衝到 10~20
        norm = np.linalg.norm(live_feat_raw)
        live_feat = live_feat_raw / (norm + 1e-10)

        # 如果沒人或矩陣沒初始化
        if not hasattr(self, 'base_matrix') or self.base_matrix.shape[0] == 0:
             return None, 0.0, False, {}, live_feat
        
        # A. 融合特徵 (一次算出所有人的融合特徵)
        fused_matrix = (self.base_matrix * self.base_weight) + (self.dynamic_matrix * self.dynamic_weight)
        
        # B. 矩陣正規化 (L2 Norm)，確保資料庫裡的特徵長度也是 1
        norms = np.linalg.norm(fused_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10 
        fused_matrix = fused_matrix / norms
        
        # C. 計算相似度 (Normalized Dot Product)
        # 現在兩邊長度都是 1，算出來一定是 -1 ~ 1 之間
        fused_scores = np.dot(fused_matrix, live_feat)
        
        # D. 猶豫邏輯與選出最佳者
        best_idx = 0
        score_1st = 0.0

        if len(self.emp_ids) >= 2:
            sorted_indices = np.argsort(fused_scores)[::-1]
            best_idx = sorted_indices[0]
            second_idx = sorted_indices[1]
            
            score_1st = float(fused_scores[best_idx])
            score_2nd = float(fused_scores[second_idx])
            
            ambiguity_th = self.config.get('thresholds', {}).get('ambiguity_gap', 0.05)
            
            if (score_1st - score_2nd) < ambiguity_th:
                print(f"⚠️ [猶豫] Top1:{self.emp_ids[best_idx]}({score_1st:.2f}) vs Top2:{self.emp_ids[second_idx]}({score_2nd:.2f})")
                return None, score_1st, False, {"warning": True, "reason": "ambiguous_gap"}, live_feat
        else:
            best_idx = np.argmax(fused_scores)
            score_1st = float(fused_scores[best_idx])

        max_fused_score = score_1st
        best_emp_id = self.emp_ids[best_idx]
        best_base_feat = self.base_matrix[best_idx]
        best_dyn_feat = self.dynamic_matrix[best_idx]
        has_dynamic = self.has_dynamic_flags[best_idx]
        
        # 這裡傳入 normalized 的 live_feat，確保 base_score 計算正確
        base_score = float(self.compute_similarity(live_feat, best_base_feat))
        
        dyn_score = 0.0
        if has_dynamic:
            dyn_score = float(self.compute_similarity(live_feat, best_dyn_feat))
            
        # 診斷輸出 (現在應該會看到 0.7, 0.8 這種正常分數了)
        if max_fused_score > 0.4:
             print(f"📊 [診斷] ID: {best_emp_id} | 總分: {max_fused_score:.2f} | Base: {base_score:.2f} | Dynamic: {dyn_score:.2f}")

        should_evolve = False
        if has_dynamic:
            if base_score > self.evo_min_base or dyn_score > self.evo_min_dyn:
                should_evolve = True
        else:
            if max_fused_score > self.evo_threshold:
                should_evolve = True

        low_base_warning = (base_score < self.warn_base_th)

        final_details = {
            "base_score": base_score,
            "dynamic_score": dyn_score,
            "fused_score": max_fused_score,
            "warning": low_base_warning,
            "matched_old_dynamic": best_dyn_feat if has_dynamic else None,
            "candidate_id": best_emp_id
        }

        if max_fused_score >= self.rec_threshold:
            return best_emp_id, max_fused_score, should_evolve, final_details, live_feat
        
        return None, max_fused_score, False, final_details, live_feat

    def process_attendance(self, emp_id, score, should_evolve, live_feat, photo_path, details):
        """
        處理打卡儲存與演進 (含記憶體熱更新)
        """
        # 注意：若有 debounce (打卡太頻繁)，success 會是 False，演進會被跳過
        success, message = self.db.add_attendance_log(emp_id, score, photo_path, details)
        
        if success and should_evolve:
            base_s = details.get('base_score', 0.0)
            if base_s < 0.4:
                return success, message + " (辨識成功，跳過演進: 差異過大)"

            if live_feat is not None:          
                old_dynamic = details.get("matched_old_dynamic")

                if old_dynamic is not None:
                    alpha = 0.1 
                    new_dynamic = (alpha * live_feat) + ((1 - alpha) * old_dynamic)
                    new_dynamic = new_dynamic / np.linalg.norm(new_dynamic)
                    print(f"🌊 [Soft Update] 融合舊特徵 (Alpha={alpha})")
                else:
                    new_dynamic = live_feat # 冷啟動
                    print(f"🌱 [Cold Start] 建立初始動態特徵")

                # 1. 寫入資料庫
                self.db.update_dynamic_feature(emp_id, new_dynamic)
                
                # 2. 同步更新記憶體中的矩陣
                if emp_id in self.emp_ids:
                    idx = self.emp_ids.index(emp_id)
                    self.dynamic_matrix[idx] = new_dynamic
                    self.has_dynamic_flags[idx] = True
                    print(f"🧠 [Memory Update] 記憶體特徵已同步 (ID: {emp_id})")

                message += " (特徵已柔和演進)"

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
    emp_id, score, evolve, details, live_feat = recognizer.identify(test_img)

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