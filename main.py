import argparse
import csv
import cv2
import torch
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
from ultralytics import RTDETR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from collections import Counter

# Constants
DETECTION_MODEL_PATH = "weights/best.pt"
OCR_MODEL = "microsoft/trocr-small-printed"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3 

class LicensePlateSystemV3:
    def __init__(self, detection_model_path=DETECTION_MODEL_PATH, ocr_model_name=OCR_MODEL):
        print(f"Loading custom RF-DETR model from: {detection_model_path}...")
        self.det_model = RTDETR(detection_model_path)
        
        print(f"Loading OCR model: {ocr_model_name}...")
        self.ocr_processor = TrOCRProcessor.from_pretrained(ocr_model_name)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_model.to(self.device)
        print(f"Models loaded. OCR running on {self.device}")
        
        # Tracking & Voting
        self.track_id_counter = 0
        self.tracks = {} # id -> {"box": [], "history": [], "logged": False, "first_seen": "", "last_seen_idx": 0}
        self.log_file = "traversed_vehicles_v3.csv"
        
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Plate Number", "First Seen", "Last Seen", "Confidence"])

    def get_iou(self, boxA, boxB):
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA, areaB = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]), (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0

    def process_frame(self, frame, frame_idx):
        results = self.det_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        current_dets = [{"box": b.xyxy[0].tolist(), "score": b.conf.item()} for b in results.boxes]

        matched_ids = set()
        new_tracks = {}
        now_ts = datetime.now().strftime("%H:%M:%S")
        
        for det in current_dets:
            best_iou, best_id = 0, None
            for tid, track in self.tracks.items():
                if tid in matched_ids: continue
                iou = self.get_iou(det["box"], track["box"])
                if iou > IOU_THRESHOLD and iou > best_iou:
                    best_iou, best_id = iou, tid
            
            if best_id is not None:
                matched_ids.add(best_id)
                track = self.tracks[best_id]
                track.update({"box": det["box"], "score": det["score"], "last_seen_idx": frame_idx, "last_seen_ts": now_ts})
                new_tracks[best_id] = track
            else:
                self.track_id_counter += 1
                new_tracks[self.track_id_counter] = {
                    "box": det["box"], "score": det["score"], "history": [], 
                    "logged": False, "first_seen": now_ts, "last_seen_ts": now_ts, "last_seen_idx": frame_idx
                }


        for tid, track in self.tracks.items():
            if tid not in matched_ids:
                if (frame_idx - track["last_seen_idx"]) < 15:
                    new_tracks[tid] = track
                elif not track.get("finalized", False) and len(track["history"]) >= 5:
                    self.finalize_log(tid, track)
                    track["finalized"] = True
        
        self.tracks = new_tracks

        for tid, track in self.tracks.items():
            if track["last_seen_idx"] == frame_idx and (frame_idx % 5 == 0):
                text = self.recognize_plate(frame, track["box"])
                clean = "".join(e for e in text if e.isalnum()).upper()
                if len(clean) >= 4: track["history"].append(clean)
            
            if track["history"]:
                track["best_text"] = Counter(track["history"]).most_common(1)[0][0]
        return self.tracks

    def finalize_log(self, tid, track):
        plate = track.get("best_text", "UNKNOWN")
        noise_words = ["TAXI", "FINAINRU", "SYSTEM", "TOTAL", "GOOD", "SERVICE", "FINANCIAL"]
        if plate in noise_words or len(plate) < 4:
            return

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([tid, plate, track["first_seen"], track["last_seen_ts"], f"{track['score']:.2f}"])
        print(f"  [FINAL LOG] ID {tid}: {plate} (Traversed: {track['first_seen']} -> {track['last_seen_ts']})")

    def recognize_plate(self, image, box):
        xmin, ymin, xmax, ymax = map(int, box)
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image.shape[1], xmax), min(image.shape[0], ymax)
        if xmax <= xmin or ymax <= ymin: return ""
        
        crop_img = image[ymin:ymax, xmin:xmax]
        crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        pixel_values = self.ocr_processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.ocr_model.generate(pixel_values)
        return self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="stable_result.mp4")
    args = parser.parse_args()

    system = LicensePlateSystemV3()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input}")
        return
        
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing for Stability: {args.input}")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        tracks = system.process_frame(frame, frame_idx)
        
        for tid, track in tracks.items():
            if track["last_seen_idx"] == frame_idx:
                b = list(map(int, track["box"]))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                best_text = track.get('best_text', 'Detecting...')
                label = f"ID {tid}: {best_text}"
                cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0: print(f"  Frame {frame_idx}...")

    cap.release()
    out.release()
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
