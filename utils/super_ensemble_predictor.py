"""
Super Ensemble Predictor for ThyroNet
3 YOLO detectors + 8 classifiers — all trained on TN5000, no data leakage
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
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as tv
import timm


class SuperEnsemblePredictor:
    """
    Super Ensemble:
      Detection  — YOLOv8m + YOLOv8l + YOLOv9c (best confidence wins)
      Classification — DenseNet121, ResNet50, ResNet152, MobileNetV3,
                       EfficientNetB4, VGG16, InceptionV3, ConvNeXt-Tiny
    """

    def __init__(self, models_dir='models_retrained'):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SuperEnsemblePredictor — device: {self.device}")

        # ── YOLO models ───────────────────────────────────────────────
        self.yolo_models = {}
        for fname in ['yolo_v8m_thyroid.pt', 'yolo_v8l_thyroid.pt', 'yolo_v9c_thyroid.pt']:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                try:
                    self.yolo_models[fname] = YOLO(path)
                    print(f"  YOLO loaded: {fname}")
                except Exception as e:
                    print(f"  YOLO failed: {fname} — {e}")

        if not self.yolo_models:
            raise ValueError("No YOLO models loaded from " + models_dir)

        # ── Classifier definitions ────────────────────────────────────
        # (name, filename, loader_fn, weight)
        self.classifier_configs = [
            ('DenseNet121',    'densenet121_thyroid.pt',   self._load_densenet121,   1.2),
            ('ResNet50',       'resnet50_thyroid.pt',      self._load_resnet50,      1.0),
            ('ResNet152',      'resnet152_thyroid.pt',     self._load_resnet152,     1.1),
            ('MobileNetV3',    'mobilenetv3_thyroid.pt',   self._load_mobilenet,     0.9),
            ('EfficientNetB4', 'efficientnetb4_thyroid.pt',self._load_efficientnetb4,1.2),
            ('VGG16',          'vgg16_thyroid.pt',         self._load_vgg16,         1.0),
            ('InceptionV3',    'inceptionv3_thyroid.pt',   self._load_inceptionv3,   1.0),
            ('ConvNeXt',       'convnext_thyroid.pt',      self._load_convnext,      1.1),
        ]

        self.classifiers = {}
        total_weight = 0.0
        for name, fname, loader, weight in self.classifier_configs:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                try:
                    model = loader(path)
                    if model is not None:
                        self.classifiers[name] = {'model': model, 'weight': weight}
                        total_weight += weight
                        print(f"  Classifier loaded: {name}")
                except Exception as e:
                    print(f"  Classifier failed: {name} — {e}")

        if not self.classifiers:
            raise ValueError("No classifiers loaded from " + models_dir)

        # Normalise weights
        for name in self.classifiers:
            self.classifiers[name]['weight'] /= total_weight

        # ── Transforms ────────────────────────────────────────────────
        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])
        self.transform_299 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

        print(f"Ready — {len(self.yolo_models)} YOLO + {len(self.classifiers)} classifiers")

    # ── Model loaders ─────────────────────────────────────────────────
    def _load_densenet121(self, path):
        m = tv.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_resnet50(self, path):
        m = tv.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_resnet152(self, path):
        m = tv.resnet152(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_mobilenet(self, path):
        m = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_efficientnetb4(self, path):
        m = tv.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_vgg16(self, path):
        m = tv.vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, 2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    def _load_inceptionv3(self, path):
        m = tv.inception_v3(weights=None, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.AuxLogits.fc = nn.Linear(768, 2)
        m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        m = m.to(self.device)
        m.eval()
        m.aux_logits = False   # disable aux output at inference
        return m

    def _load_convnext(self, path):
        m = timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.to(self.device).eval()

    # ── Image validation ──────────────────────────────────────────────
    def _validate_image(self, img_pil):
        arr = np.array(img_pil)
        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)
        color_diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3
        gray = np.mean(arr, axis=2)
        white_ratio = np.sum(gray > 220) / gray.size
        dark_ratio  = np.sum(gray < 30)  / gray.size
        std_bright  = np.std(gray)

        if color_diff > 30:
            return False, '❌ Please upload a grayscale thyroid ultrasound image.'
        if white_ratio > 0.60:
            return False, '❌ This looks like a document or screenshot.'
        if std_bright < 5:
            return False, '❌ Image appears blank.'
        return True, None

    # ── YOLO ensemble detection ───────────────────────────────────────
    def _detect_nodule(self, image_bgr):
        best_box, best_conf = None, 0.0
        for yolo in self.yolo_models.values():
            results = yolo(image_bgr, verbose=False)[0]
            if len(results.boxes) > 0:
                idx  = int(results.boxes.conf.argmax())
                conf = float(results.boxes.conf[idx])
                if conf > best_conf:
                    best_conf = conf
                    best_box  = tuple(map(int, results.boxes.xyxy[idx]))
        return best_box, best_conf

    # ── Main predict ──────────────────────────────────────────────────
    def predict(self, image_input):
        try:
            img_pil = (Image.open(image_input) if isinstance(image_input, str)
                       else image_input).convert('RGB')

            valid, err = self._validate_image(img_pil)
            if not valid:
                return {'success': False, 'error': err}

            image    = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            orig     = image.copy()
            h, w     = image.shape[:2]

            # Remove black border
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            border_offset = (0, 0)
            image_for_yolo = image
            if contours:
                largest = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                if bw * bh > w * h * 0.3:
                    image_for_yolo = image[by:by+bh, bx:bx+bw]
                    border_offset  = (bx, by)

            box, det_conf = self._detect_nodule(image_for_yolo)
            if box is None:
                return {'success': False,
                        'error': '❌ No thyroid nodule detected. Please upload a clear ultrasound showing a nodule.'}

            x1, y1, x2, y2 = box
            x1 += border_offset[0]; y1 += border_offset[1]
            x2 += border_offset[0]; y2 += border_offset[1]

            pad = 10
            crop = image[max(0,y1-pad):min(h,y2+pad),
                         max(0,x1-pad):min(w,x2+pad)]
            if crop.size == 0:
                return {'success': False, 'error': 'Could not crop nodule region.'}

            # CLAHE preprocessing
            gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            eq     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_c)
            rgb_c  = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            pil_c  = Image.fromarray(rgb_c)

            t224 = self.transform_224(pil_c).unsqueeze(0).to(self.device)
            t299 = self.transform_299(pil_c).unsqueeze(0).to(self.device)

            # Classify
            individual = {}
            weighted_sum = 0.0

            with torch.no_grad():
                for name, info in self.classifiers.items():
                    model  = info['model']
                    weight = info['weight']
                    tensor = t299 if name == 'InceptionV3' else t224
                    out    = model(tensor)
                    if isinstance(out, tuple):
                        out = out[0]
                    prob_mal = float(torch.softmax(out, 1)[0][1])
                    weighted_sum += prob_mal * weight
                    pred = 'Malignant' if prob_mal > 0.5 else 'Benign'
                    conf = prob_mal if prob_mal > 0.5 else (1 - prob_mal)
                    individual[name] = {
                        'prediction':    pred,
                        'confidence':    round(conf * 100, 2),
                        'malignant_prob':round(prob_mal * 100, 2),
                        'benign_prob':   round((1 - prob_mal) * 100, 2),
                        'weight':        round(weight * 100, 2),
                    }

            final_score     = weighted_sum
            predicted_class = 'Malignant' if final_score > 0.5 else 'Benign'

            if predicted_class == 'Malignant':
                confidence = round(50 + (final_score - 0.5) / 0.5 * 49, 2)
            else:
                confidence = round(50 + (0.5 - final_score) / 0.5 * 49, 2)
            confidence = float(min(99.0, max(50.0, confidence)))

            seg_img = self._draw_detection(orig, x1, y1, x2, y2,
                                            predicted_class, confidence)

            preds_list = [v['prediction'] for v in individual.values()]
            agreement  = preds_list.count(predicted_class) / len(preds_list) * 100

            return {
                'success':            True,
                'prediction':         predicted_class,
                'confidence':         confidence,
                'probabilities': {
                    'Benign':    round((1 - final_score) * 100, 2),
                    'Malignant': round(final_score * 100, 2),
                },
                'model_used':         f'Super Ensemble ({len(self.yolo_models)} YOLO + {len(self.classifiers)} Classifiers)',
                'detection_confidence': round(det_conf * 100, 2),
                'individual_predictions': individual,
                'segmentation_image': seg_img,
                'ensemble_details': {
                    'num_yolo_models':  len(self.yolo_models),
                    'num_classifiers':  len(self.classifiers),
                    'model_agreement':  round(agreement, 2),
                    'weighted_ensemble': True,
                },
            }

        except Exception as e:
            import traceback
            return {'success': False,
                    'error': f'Prediction error: {str(e)}\n{traceback.format_exc()}'}

    def _draw_detection(self, image, x1, y1, x2, y2, prediction, confidence):
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        color   = (255, 50, 50) if prediction == 'Malignant' else (50, 220, 50)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label      = f"{prediction} {confidence:.1f}%"
        font_scale = max(0.5, image.shape[1] / 700)
        thickness  = max(1, int(image.shape[1] / 400))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        label_y = max(th + 10, y1 - 8)
        cv2.rectangle(overlay, (x1, label_y-th-6), (x1+tw+6, label_y+4), color, -1)
        cv2.putText(overlay, label, (x1+3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        buf.seek(0)
        return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
