MODEL = 'yolov8n-seg.pt'
CONFNAME = "conf_nano_v1_15k.yaml"
DATASET = "train_15000_split.txt"
PARAMS = {
    "device": "cpu",
    "name": 'nano_v1_15k',
    "epochs": 5,
    "imgsz": 640,
    "batch": 16,
    "save_period": None,
}
