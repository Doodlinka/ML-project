import os
PROJDIR = os.path.abspath(os.path.dirname(__file__))
MODEL = os.path.join(PROJDIR, 'yolov8n-seg.pt')
CONFNAME = os.path.join(PROJDIR, "conf_nano_v1_15k.yaml")
DATASET = os.path.join(PROJDIR, "train_15000_split.txt")
PARAMS = {
    "data": CONFNAME,
    "device": "cpu",
    "name": 'nano_v1_15k',
    "epochs": 5,
    "imgsz": 640,
    "batch": 16,
}
