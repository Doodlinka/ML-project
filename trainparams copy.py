import os
AMWAUG = 15
PROJDIR = os.path.abspath(os.path.dirname(__file__))
MODEL = os.path.join(PROJDIR, 'yolov8n-seg.pt')
CONFNAME = os.path.join(PROJDIR, f"conf_nano_v1_{AMWAUG}k.yaml")
DATASET = os.path.join(PROJDIR, f"train_{AMWAUG}000_split.txt")

PARAMS = {
    "data": CONFNAME,
    # disable per-pixel cause it's done offline
    "hsv_h": 0.0,      
    "hsv_s": 0.0,      
    "hsv_v": 0.0,            

    "device": 0,
    
    "name": f'nano_v2_{AMWAUG}k',
    "cls": 2,
    "epochs": 10,
    "imgsz": 640,
    "batch": 12,
    "workers": 5,
}