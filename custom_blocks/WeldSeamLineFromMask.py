from studio.custom_block import *
import cv2
import numpy as np
import json
from ultralytics import YOLO

# ---- MODEL PATH ----
MODEL_PATH = r"C:\Users\Rumeysa SAKIN\AugeLab Studio\models\best.pt"


class WeldSeamLineFromMask(Block):
    op_code = "WeldSeamLineFromMask"
    title = "Weld Seam Line Detector"
    tooltip = "Detects weld seam line from segmentation mask."

    def init(self):
        self.width = 350

        self.input_sockets = [
            SocketTypes.ImageAny("Input Image")
        ]

        self.output_sockets = [
            SocketTypes.ImageAny("Annotated Image"),
            SocketTypes.String("Coordinates")
        ]

        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            self.logError(f"Model yüklenemedi: {e}")
            self.model = None

    def run(self):
        def skeletonize(binary_img):
            img = binary_img.copy()
            size = img.size
            skel = np.zeros(img.shape, np.uint8)

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            done = False

            while not done:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()

                zeros = size - cv2.countNonZero(img)
                if zeros == size:
                    done = True

            return skel

        if self.model is None:
            self.logError("Model yüklenmemiş!")
            return

        img = self.input["Input Image"].data
        img = np.asarray(img)

        # YOLO
        try:
            results = self.model(img, verbose=False)
        except Exception as e:
            self.logError(f"Tahmin hatası: {e}")
            self.output["Annotated Image"].data = img
            self.output["Coordinates"].data = ""
            return

        res = results[0]

        if res.masks is None:
            self.output["Annotated Image"].data = img
            self.output["Coordinates"].data = ""
            return

        try:
            masks = res.masks.data.cpu().numpy()
        except Exception as e:
            self.logError(f"Maskeleri okurken hata: {e}")
            self.output["Annotated Image"].data = img
            self.output["Coordinates"].data = ""
            return

        best_mask = None
        best_area = 0
        for m in masks:
            m_bin = (m > 0.5).astype(np.uint8)
            area = m_bin.sum()
            if area > best_area:
                best_area = area
                best_mask = m_bin

        if best_mask is None or best_area == 0:
            self.output["Annotated Image"].data = img
            self.output["Coordinates"].data = ""
            return

        mask = best_mask  # 0/1

        binary = (mask * 255).astype(np.uint8)
        skel = skeletonize(binary)

        ys, xs = np.where(skel > 0)
        if len(xs) < 2:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                self.output["Annotated Image"].data = img
                self.output["Coordinates"].data = ""
                return
            cnt = max(contours, key=cv2.contourArea)
            pts = cnt.reshape(-1, 2).astype(np.float32)
        else:
            pts = np.stack([xs, ys], axis=1).astype(np.float32)

        if len(pts) < 2:
            self.output["Annotated Image"].data = img
            self.output["Coordinates"].data = ""
            return

        annotated = img.copy()

        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            direction = np.array([vx, vy]).reshape(2)
            origin = np.array([x0, y0]).reshape(2)

            projections = np.dot(pts - origin, direction)
            t_min, t_max = projections.min(), projections.max()

            p1 = origin + t_min * direction
            p2 = origin + t_max * direction

            x1, y1 = map(int, p1)
            x2, y2 = map(int, p2)

            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            coords = json.dumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            self.output["Coordinates"].data = coords

        except Exception as e:
            self.logError(f"Çizgi hesaplama hatası: {e}")
            self.output["Coordinates"].data = ""

        self.output["Annotated Image"].data = annotated

add_block(WeldSeamLineFromMask.op_code, WeldSeamLineFromMask)
