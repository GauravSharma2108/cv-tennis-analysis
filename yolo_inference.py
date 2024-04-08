from ultralytics import YOLO

model = YOLO('yolov8x') # load the yolo model

run_type = 'video'

if __name__ == '__main__':
    if run_type=='image':
        # when save=True is used, the image will be saved in the same directory run/predict/predicted_image
        result = model.predict("Media/image.png", save=True) # predict the image and save the image
        print(result)
        print("="*30)
        for box in result[0].boxes:
            print(box)
    
    elif run_type=='video':
        result = model.predict("Media/input_video.mp4", save=True) # predict the image and save the image
        print(result)
        print("="*30)
        for box in result[0].boxes:
            print(box)

    else:
        pass

# --- sample output for images ---
# boxes: ultralytics.engine.results.Boxes object
# keypoints: None
# masks: None
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# obb: None
# orig_img: array([[[ 24,  23,  38],
#         [ 24,  23,  38],
#         [ 24,  23,  38],
#         ...,
#         [  0,   0,   0]]], dtype=uint8)
# orig_shape: (1204, 2471)
# path: '/Users/psykick/Documents/learning/github/cv-tennis-analysis/Media/image.png'
# probs: None
# save_dir: '/Users/psykick/Documents/learning/github/cv-tennis-analysis/runs/detect/predict2'
# speed: {'preprocess': 3.258943557739258, 'inference': 351.4680862426758, 'postprocess': 540.686845779419}]



# --- sample box output for images ---

# ultralytics.engine.results.Boxes object with attributes:
# cls: tensor([0.])
# conf: tensor([0.9018])
# data: tensor([[7.4372e+02, 9.0695e+02, 9.3568e+02, 1.1587e+03, 9.0184e-01, 0.0000e+00]])
# id: None
# is_track: False
# orig_shape: (1204, 2471)
# shape: torch.Size([1, 6])
# xywh: tensor([[ 839.7046, 1032.8372,  191.9606,  251.7808]])
# xywhn: tensor([[0.3398, 0.8578, 0.0777, 0.2091]])
# xyxy: tensor([[ 743.7242,  906.9467,  935.6849, 1158.7275]])
# xyxyn: tensor([[0.3010, 0.7533, 0.3787, 0.9624]])