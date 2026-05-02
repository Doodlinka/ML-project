import os
AMWAUG = 15
PROJDIR = os.path.abspath(os.path.dirname(__file__))
MODEL = os.path.join(PROJDIR, 'yolov8n-seg.pt')
CONFNAME = os.path.join(PROJDIR, f"conf_nano_v1_{AMWAUG}k.yaml")
DATASET = os.path.join(PROJDIR, f"train_{AMWAUG}000_split.txt")

PARAMS = {
    "data": CONFNAME,
    "device": 0,
    "amp": True,
    # disable per-pixel cause it's done offline
    "hsv_h": 0.0,      
    "hsv_s": 0.0,      
    "hsv_v": 0.0,    

    "name": f'nano_v3_{AMWAUG}k',        

    "imgsz": 640,
    "batch": 0.8,

    "epochs": 10,
    "patience": 5,
    "close_mosaic": 2,

    # "cls": 1.5,
    "copy_paste": 0.3,
    "optimizer": "AdamW",
}