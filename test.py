import pickle, cv2

# Ucitaj pkl fajl
with open('tracker_stubs/ball_detections.pkl','rb') as f:
    detections = pickle.load(f)

# Pogledaj neki bbox
print("Primer bounding boxa:", detections)
