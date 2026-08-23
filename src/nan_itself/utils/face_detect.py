import sqlite3
import pickle
from pathlib import Path
from loguru import logger

import cv2
import numpy as np
import time

import face_recognition


class FaceRecognizer:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        logger.info("FaceRecognizer initialized")
        self.conn = sqlite3.connect(self.db_path)
        self.camera = None

    def detect(self, timeout:float = 3.0):
        """
        打开摄像头，直到检测到一张人脸。

        - 已存在：返回已有 label
        - 不存在：创建新 label，并保存人脸图像和 encoding
        - 无需用户按键
        - 检测完成后自动释放摄像头
        """
        try:
            # -----------------------------
            # 1. 初始化数据库
            # -----------------------------
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL UNIQUE,
                    image BLOB NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

            # -----------------------------
            # 2. 从数据库加载人脸数据
            # -----------------------------
            rows = self.conn.execute("""
                SELECT id, label, encoding
                FROM faces
                ORDER BY id
            """).fetchall()

            known_labels = []
            known_encodings = []

            for row_id, label, encoding_blob in rows:
                try:
                    encoding = np.frombuffer(
                        pickle.loads(encoding_blob),
                        dtype=np.float64
                    )

                    if encoding.shape == (128,):
                        known_labels.append(label)
                        known_encodings.append(encoding)

                except Exception:
                    # 单条数据损坏时，不影响其他人脸
                    continue

            # -----------------------------
            # 3. 打开摄像头
            # -----------------------------
            t0 = time.time()
            self.camera = cv2.VideoCapture(0)
            while True:
                if time.time() - t0 > timeout:
                    return None
                ok, frame = self.camera.read()

                # BGR -> RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # -----------------------------
                # 4. 检测人脸
                # -----------------------------
                locations = face_recognition.face_locations(rgb)

                if not locations:
                    continue

                # 只处理第一张人脸
                location = locations[0]

                encodings = face_recognition.face_encodings(
                    rgb,
                    [location]
                )

                if not encodings:
                    continue

                current_encoding = encodings[0]

                # -----------------------------
                # 5. 与数据库人脸进行比对
                # -----------------------------
                matched_label = None

                if known_encodings:
                    distances = face_recognition.face_distance(
                        known_encodings,
                        current_encoding
                    )

                    best_index = int(np.argmin(distances))
                    best_distance = float(distances[best_index])

                    if best_distance <= 0.48:
                        matched_label = known_labels[best_index]

                # -----------------------------
                # 6. 已存在
                # -----------------------------
                if matched_label is not None:
                    return matched_label

                # -----------------------------
                # 7. 新建人脸
                # -----------------------------
                next_id = self.conn.execute(
                    """
                    SELECT COALESCE(MAX(id), 0) + 1
                    FROM faces
                    """
                ).fetchone()[0]

                new_label = f"person_{next_id:06d}"

                top, right, bottom, left = location

                h, w = frame.shape[:2]

                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)
                left = max(0, left)

                face_image = frame[
                    top:bottom,
                    left:right
                ]

                if face_image.size == 0:
                    continue

                # -----------------------------
                # 8. 保存人脸图像
                # -----------------------------
                success, image_buffer = cv2.imencode(
                    ".jpg",
                    face_image
                )

                if not success:
                    continue

                image_blob = image_buffer.tobytes()

                # -----------------------------
                # 9. 保存 encoding
                # -----------------------------
                encoding_blob = pickle.dumps(
                    current_encoding.astype(np.float64),
                    protocol=pickle.HIGHEST_PROTOCOL
                )

                self.conn.execute(
                    """
                    INSERT INTO faces
                    (label, image, encoding)
                    VALUES (?, ?, ?)
                    """,
                    (
                        new_label,
                        image_blob,
                        encoding_blob
                    )
                )

                self.conn.commit()

                return new_label

        finally:
            self.camera.release()
            self.conn.close()

    def close(self):
        if self.camera:
            self.camera.release()
        self.conn.close()
        logger.info("FaceRecognizer closed")
