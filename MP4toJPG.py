import cv2
import os

video = cv2.VideoCapture('MP4/Muck_02.mp4')

video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print(str(video_length) + " Frames")
frame_number = 0 

jpg_folder = 'JPG'
if not os.path.exists(jpg_folder):
    os.makedirs(jpg_folder)

while True:
    success, img = video.read()

    cv2.imshow("Muck Video", img)

    if frame_number % 10 == 0:
        print('Frame {:05d} / {}'.format(frame_number, video_length))
        name = '{}/File2_Frame{:05d}.jpg'.format(jpg_folder, frame_number)
        cv2.imwrite(name, img)

    frame_number += 1    

    if not success:
        print("Failed")
        break

    key = cv2.waitKey(1)
    if key==ord('q'):
        print("Quit")
        break

cv2.destroyAllWindows