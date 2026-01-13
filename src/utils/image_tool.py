import cv2
import numpy as np

class ImagePreprocessor:
    """
    統一影像處理工具：包含 CLAHE 強化與支援口罩模式的人臉仿射變換對齊。
    """
    def __init__(self):
        # 初始化 CLAHE 強化工具
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 定義 InsightFace/ArcFace 標準 112x112 空間中，雙眼應在的座標基準
        self.standard_size = (112, 112)
        self.eye_std_l = [38.2946, 51.6963]
        self.eye_std_r = [73.5318, 51.5014]
        
        # 標準 5 點座標 (用於未戴口罩時)
        self.src_landmarks_5pt = np.array([
            [38.2946, 51.6963], # 左眼
            [73.5318, 51.5014], # 右眼
            [56.0252, 71.7366], # 鼻子
            [41.5493, 92.3655], # 左嘴角
            [70.7299, 92.2041]  # 右嘴角
        ], dtype=np.float32)

    def enhance_face(self, face_img):
        """強化影像紋理與對比度"""
        if face_img is None or face_img.size == 0:
            return face_img
        
        # 轉到 LAB 空間強化亮度 L 頻道
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
        
        # 輕微銳化卷積核
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(enhanced_bgr, -1, kernel)

    def align_face(self, frame, landmarks_5pt, is_masked=False):
        """
        執行仿射變換：將歪斜的人臉轉正。
        :param landmarks_5pt: [左眼, 右眼, 鼻子, 左嘴角, 右嘴角] 的 (x, y) 座標
        :param is_masked: 是否戴口罩 (影響對齊策略)
        """
        dst_landmarks = np.array(landmarks_5pt, dtype=np.float32)

        if not is_masked:
            # --- 沒戴口罩：使用標準 5 點變換 (最精準) ---
            tform, _ = cv2.estimateAffinePartial2D(dst_landmarks, self.src_landmarks_5pt)
            if tform is not None:
                return cv2.warpAffine(frame, tform, self.standard_size, borderValue=0)
        
        # --- 戴口罩或 5 點變換失敗：使用「雙眼基準」模式 ---
        eye_l, eye_r = dst_landmarks[0], dst_landmarks[1]
        
        # 1. 計算旋轉角度
        dy = eye_r[1] - eye_l[1]
        dx = eye_r[0] - eye_l[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 2. 計算縮放比例 (以標準眼距為基準)
        dist = np.sqrt(dx**2 + dy**2)
        std_dist = self.eye_std_r[0] - self.eye_std_l[0]
        scale = std_dist / dist
        
        # 3. 取得旋轉與縮放矩陣 (繞雙眼中心點轉)
        eye_center = (float((eye_l[0] + eye_r[0]) / 2), float((eye_l[1] + eye_r[1]) / 2))
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        
        # 4. 修正平移量，將眼睛對齊至標準位置
        t_x = (self.eye_std_l[0] + self.eye_std_r[0]) / 2 - eye_center[0]
        t_y = self.eye_std_l[1] - eye_center[1]
        M[0, 2] += t_x
        M[1, 2] += t_y
        
        return cv2.warpAffine(frame, M, self.standard_size, borderValue=0)

    def get_brightness(self, img):
        """計算平均亮度"""
        return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))