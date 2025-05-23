import cv2
import argparse
from ultralytics import YOLO
import torch

def load_source(source):
    # Check if webcam
    if source == "0":
        return True, cv2.VideoCapture(0)
    # Check if image
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    if source.split('.')[-1].lower() in img_formats:
        return False, cv2.imread(source)
    # Otherwise assume video
    return True, cv2.VideoCapture(source)

def draw_boxes(image, results, names, thickness=2):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/mkc.mp4", help="Path to video/image or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    args = parser.parse_args()

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model
    model = YOLO(args.model)
    model.to(device)

    # Load source
    is_video, src = load_source(args.source)

    if not is_video:
        # If image
        results = model(src, conf=args.conf)
        names = model.names
        img = draw_boxes(src, results, names, args.thickness)
        cv2.imshow("Detection", img)
        cv2.waitKey(0)
    else:
        # If webcam or video
        cap = src
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf)
            names = model.names
            img = draw_boxes(frame, results, names, args.thickness)

            cv2.imshow("Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    cv2.destroyAllWindows()
