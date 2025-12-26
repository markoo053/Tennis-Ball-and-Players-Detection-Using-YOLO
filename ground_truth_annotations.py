import os
import pandas as pd
import cv2

folder = 'frames2_gt'
data = []
classes = {0: 'player', 1: 'ball'}  # klasa 0 = player, 1 = ball

for file in os.listdir(folder):
    if file.endswith('.txt'):
        img_path = os.path.join(folder, file.replace('.txt', '.jpg'))
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        frame_id = int(file.replace('frame_', '').replace('.txt', ''))
        with open(os.path.join(folder, file)) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                cls_name = classes[int(cls)]
                # pretvori iz relativnih u piksel vrednosti
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                data.append([frame_id, cls_name, x1, y1, x2, y2])

df = pd.DataFrame(data, columns=['frame_id','class_name','x1','y1','x2','y2'])
df.to_csv('ground_truth_annotations.csv', index=False)

print(" Ground truth fajl sačuvan kao 'ground_truth_annotations.csv'")


