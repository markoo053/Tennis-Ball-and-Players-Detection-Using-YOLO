import pandas as pd

gt = pd.read_csv('ground_truth_annotations.csv')
gt['frame_id'] = gt['frame_id'] * 2
gt.to_csv('ground_truth_annotations_adjusted.csv', index=False)

print(" Ground truth je ažuriran — frame_id sada odgovara detekcijama.")

