import cv2
import os

video_path = 'input_videos/input_video.mp4'
output_folder = 'frames2_gt'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Čuva svaki drugi frame (možeš promeniti u %1 da sačuva sve)
    if count % 2 == 0:
        cv2.imwrite(f'{output_folder}/frame_{frame_id:04d}.jpg', frame)
        frame_id += 1

    count += 1

cap.release()
print(f"Snimljeno je {frame_id} frameova u folder '{output_folder}'")


