import cv2
from ultralytics import YOLO
from collections import defaultdict

def __main__():
    # Load Yolo11 pretrained model trained with CoCo dataset
    model = YOLO("yolo11l.pt")      # Load Yolo11 pretrained model trained with CoCo dataset
    class_names = model.names

    video_path = "test_videos/Road-traffic-720p.mp4"
    cap = cv2.VideoCapture(video_path)

    line_y_red = 430 # red line position
    line_x1 = 670
    line_x2 = 1130

    # To count the number of objects of each class
    class_counts = defaultdict(int) 

    # To store the ids of objects that have crossed the red line
    crossed_ids = set() 
    prev_positions = {}

    # tracking video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # run yolo tracking on the frame, using BoT-SORT algorithm
        results = model.track(frame, persist=True, classes=[1,2,3,5,6,7])


        # print("results >>> ", results)
        if results[0].boxes.data is not None:
            # get the detected bboxes, their class indices, track ids and confidence scores
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()

            # draw incoming red zone
            cv2.line(frame, (line_x1, line_y_red), (line_x2, line_y_red), (0, 0, 255), 2)

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)

                # to find the center of x and y values of each object
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                class_name = class_names[class_idx]



                cv2.circle(frame, (x_center, y_center), 1, (0,0,255), -1)
                cv2.putText(frame, f"{class_name} ({track_id})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

                # check if the object has crossed the red line
                if track_id in prev_positions:
                    prev_y = prev_positions[track_id]
                    # Check for crossing from top to bottom:
                    # Previous y was less than line_y (above) AND current y is greater or equal (below)
                    # Also check if within x bounds of the line
                    if prev_y < line_y_red and y_center >= line_y_red:
                        if line_x1 < x_center < line_x2: # Check X bounds
                            if track_id not in crossed_ids:
                                crossed_ids.add(track_id)
                                class_counts[class_name] += 1
                
                # Update previous position
                prev_positions[track_id] = y_center
                
            # display the counts on the frame
            y_offset = 30    # space between each class count label
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y_offset += 30
        
        # draw the results on the frame
        cv2.imshow("Yolo Object Tracking & Counting", frame)

        # exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    __main__()