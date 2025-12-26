import pickle
import pandas as pd
import numpy as np
import csv

# === UČITAVANJE ===
with open('tracker_stubs/player_detections_filtered.pkl', 'rb') as f:
    player_preds = pickle.load(f)

with open('tracker_stubs/ball_detections_interpolated.pkl', 'rb') as f:
    ball_preds = pickle.load(f)


gt = pd.read_csv('ground_truth_annotations_adjusted.csv')

# === FUNKCIJA ZA IoU ===
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxA_area + boxB_area - inter + 1e-6)

# === EVALUACIJA ===
def evaluate(preds, gt_df, class_name, iou_thresh=0.3):
    tp, fp, fn = 0, 0, 0

    gt_class = gt_df[gt_df['class_name'] == class_name].copy()
    gt_frame_ids = gt_class['frame_id'].unique()

    for gt_frame_id in gt_frame_ids:
        if gt_frame_id >= len(preds):
            continue 

        frame_preds = preds[gt_frame_id]
        gt_boxes = gt_class[gt_class['frame_id'] == gt_frame_id][['x1','y1','x2','y2']].values
        pred_boxes = list(frame_preds.values())
        
        matched = set()
        for p in pred_boxes:
            best_iou, best_idx = 0, -1
            for idx, g in enumerate(gt_boxes):
                current_iou = iou(p, g)
                if current_iou > best_iou:
                    best_iou, best_idx = current_iou, idx
            if best_iou >= iou_thresh:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall

# === IZRAČUNAVANJE ===
precision_p, recall_p = evaluate(player_preds, gt, 'player')
precision_b, recall_b = evaluate(ball_preds, gt, 'ball')

# === ISPIS I ČUVANJE ===
print(f"Player detections: Precision={precision_p:.3f}, Recall={recall_p:.3f}")
print(f"Ball detections: Precision={precision_b:.3f}, Recall={recall_b:.3f}")

with open('metrics_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model','Object','Precision','Recall'])
    writer.writerow(['YOLOv9c','Player',precision_p,recall_p])
    writer.writerow(['YOLOv9c','Ball',precision_b,recall_b])

print("Rezultati su sačuvani u 'metrics_results.csv'")


