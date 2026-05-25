"""
ThyroNet Image Predictor
Pipeline: YOLO detects nodule → crop → DenseNet + MobileNet ensemble classify
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import tensorflow as tf


class EnsembleImagePredictor:
    def __init__(self,
                 yolo_path='yolo_model.pt',
                 densenet_path='densenet_model.h5',
                 mobilenet_path='mobilenet_model.h5'):

        print("Loading YOLO nodule detection model...")
        self.yolo = YOLO(yolo_path)

        print("Loading DenseNet classification model...")
        self.densenet = tf.keras.models.load_model(densenet_path)

        print("Loading MobileNet classification model...")
        self.mobilenet = tf.keras.models.load_model(mobilenet_path)

        print("All image models loaded: YOLO + DenseNet + MobileNet")

    def predict(self, image_input):
        try:
            # Load image
            if isinstance(image_input, str):
                img_pil = Image.open(image_input).convert('RGB')
            else:
                img_pil = image_input.convert('RGB')

            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if image is None:
                return {'success': False, 'error': 'Could not read image'}

            # Validate: reject color photos, charts, screenshots, documents
            img_array = np.array(img_pil)
            r = img_array[:,:,0].astype(float)
            g = img_array[:,:,1].astype(float)
            b = img_array[:,:,2].astype(float)
            avg_color_diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3
            gray_ch = np.mean(img_array, axis=2)
            white_ratio = np.sum(gray_ch > 220) / gray_ch.size
            dark_ratio = np.sum(gray_ch < 30) / gray_ch.size
            std_brightness = np.std(gray_ch)

            # Reject clearly colored images
            if avg_color_diff > 15:
                return {'success': False, 'error': '❌ Invalid image. Please upload a grayscale thyroid ultrasound image. Color photos are not accepted.'}
            # Reject images with too much white (documents, charts, screenshots)
            if white_ratio > 0.35:
                return {'success': False, 'error': '❌ Invalid image. This looks like a document or screenshot. Please upload a real thyroid ultrasound.'}
            # Reject blank images
            if std_brightness < 10:
                return {'success': False, 'error': '❌ Image appears blank. Please upload a real thyroid ultrasound.'}
            # Ultrasounds have significant dark regions (acoustic shadow, background)
            if dark_ratio < 0.05:
                return {'success': False, 'error': '❌ Invalid image. Thyroid ultrasounds typically have dark regions. Please upload a real ultrasound.'}

            orig_image = image.copy()
            orig_h, orig_w = image.shape[:2]

            # Remove black border before YOLO detection
            # Convert to grayscale and find non-black region
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                # Only crop if the content region is at least 50% of image
                if bw * bh > orig_w * orig_h * 0.3:
                    image_for_yolo = image[by:by+bh, bx:bx+bw]
                    border_offset = (bx, by)
                else:
                    image_for_yolo = image
                    border_offset = (0, 0)
            else:
                image_for_yolo = image
                border_offset = (0, 0)

            # Step 1: YOLO nodule detection on border-cropped image
            results = self.yolo(image_for_yolo, verbose=False)[0]

            if len(results.boxes) == 0:
                return {
                    'success': False,
                    'error': '❌ No thyroid nodule detected. Please upload a clear thyroid ultrasound image showing a nodule.'
                }

            using_fallback = False

            # Best detection - adjust coords back to original image space
            boxes = results.boxes
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
            # Add border offset back
            x1 += border_offset[0]; y1 += border_offset[1]
            x2 += border_offset[0]; y2 += border_offset[1]
            detection_conf = float(boxes.conf[best_idx])

            # Step 2: Crop nodule region
            pad = 10
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(orig_w, x2 + pad)
            y2c = min(orig_h, y2 + pad)
            crop = image[y1c:y2c, x1c:x2c]

            if crop.size == 0:
                return {'success': False, 'error': 'Could not crop nodule region'}

            # Step 3: Preprocess with CLAHE for better generalization
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            crop_eq = clahe.apply(crop_gray)
            crop_rgb = cv2.cvtColor(crop_eq, cv2.COLOR_GRAY2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224)) / 255.0
            crop_input = crop_resized.reshape(1, 224, 224, 3)

            # Step 4: Classify with both models
            pred_dense = float(self.densenet.predict(crop_input, verbose=0)[0][0])
            pred_mob   = float(self.mobilenet.predict(crop_input, verbose=0)[0][0])

            # Step 5: Ensemble average
            final_score = (pred_dense + pred_mob) / 2
            predicted_class = 'Malignant' if final_score > 0.51 else 'Benign'

            # Confidence scaled from threshold outward (always 50-99%)
            if predicted_class == 'Malignant':
                confidence = round(50 + (final_score - 0.51) / (1.0 - 0.51) * 49, 2)
            else:
                confidence = round(50 + (0.51 - final_score) / 0.51 * 49, 2)
            confidence = float(min(99.0, max(50.0, confidence)))

            # Step 6: Draw detection box
            segmentation_image = self._draw_detection(
                orig_image, x1, y1, x2, y2, predicted_class, confidence
            )

            return {
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    'Benign': round((1 - final_score) * 100, 2),
                    'Malignant': round(final_score * 100, 2)
                },
                'model_used': 'YOLO Detection + DenseNet/MobileNet Ensemble',
                'detection_confidence': round(detection_conf * 100, 2),
                'individual_predictions': {
                    'DenseNet': {
                        'prediction': 'Malignant' if pred_dense > 0.51 else 'Benign',
                        'confidence': round(pred_dense * 100 if pred_dense > 0.51 else (1-pred_dense)*100, 2),
                        'weight': 0.5
                    },
                    'MobileNet': {
                        'prediction': 'Malignant' if pred_mob > 0.51 else 'Benign',
                        'confidence': round(pred_mob * 100 if pred_mob > 0.51 else (1-pred_mob)*100, 2),
                        'weight': 0.5
                    }
                },
                'segmentation_image': segmentation_image
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _draw_detection(self, image, x1, y1, x2, y2, prediction, confidence):
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        color = (255, 50, 50) if prediction == 'Malignant' else (50, 220, 50)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label = f"{prediction} {confidence:.1f}%"
        font_scale = max(0.5, image.shape[1] / 700)
        thickness = max(1, int(image.shape[1] / 400))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(th + 10, y1 - 8)
        cv2.rectangle(overlay, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(overlay, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        result_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"
