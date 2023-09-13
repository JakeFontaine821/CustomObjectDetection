import cv2
import os
from ultralytics import YOLO

video_name = 'Muck_Wolves'

muck_video = cv2.VideoCapture('MuckTestVideos/{}.mp4'.format(video_name))
   
size = (int(muck_video.get(3)), int(muck_video.get(4)))

results_path = 'DetectionResults/{}_result.mp4'.format(video_name)
results_video = cv2.VideoWriter(results_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

YOLO_model = YOLO('runs/detect/train4/weights/best.pt')

while True:
    success, img = muck_video.read()

    results = YOLO_model(img)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        cv2.rectangle(img, (int(x1), int(y1), int(x2), int(y2)), (255, 0, 0), 4)
        cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
        
    results_video.write(img)

    if not success:
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

muck_video.release()
results_video.release()
cv2.destroyAllWindows()
print("DONE")