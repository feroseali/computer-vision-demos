import cv2
from ultralytics import YOLO
from collections import defaultdict

def __main__():
    # Load Yolo11 pretrained model trained with CoCo dataset
    model = YOLO("yolo11l.pt")      # Load Yolo11 pretrained model trained with CoCo dataset
    class_names = model.names

    video_path = "test_videos/Road-traffic-720p.mp4"
    cap = cv2.VideoCapture(video_path)

    # tracking video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # run yolo tracking on the frame, using BoT-SORT algorithm
        results = model.track(frame, persist=True)

        # print("results >>> ", results)
        if results[0].boxes.data is not None:
            # get the detected bboxes, their class indices, track ids and confidence scores
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()
            
            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[class_idx]
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                
        # draw the results on the frame
        cv2.imshow("Yolo Object Tracking & Counting", frame)

        # exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()