from ultralytics import YOLO
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    model = YOLO('./runs/segment/ForestSeg/fixed_bushes_small_v1_20k2/weights/best.pt', 'segment')
    # results = model.val(data='./tmpconfig.yaml', split='test', augment=True)
    results = model.val(data='./tmpconfig.yaml', split='test')
    print(results.results_dict) 
    print(f"Precision (Box): {results.box.p}")
    print(f"Recall (Box):    {results.box.r}")
    print(f"mAP50 (Box):     {results.box.map50}")
    print(f"mAP50-95 (Box):  {results.box.map}")
    print(f"Precision (Mask): {results.seg.p}")
    print(f"Recall (Mask):    {results.seg.r}")
    print(f"mAP50 (Mask):     {results.seg.map50}")
    print(f"mAP50-95 (Mask):  {results.seg.map}")
    # model.export(format='engine', device=0)
    
# no TTA
# Precision (Box): [    0.65359     0.31094]
# Recall (Box):    [    0.60553     0.12952]
# mAP50 (Box):     0.3607012942407141
# mAP50-95 (Box):  0.19393424210070176
# Precision (Mask): [    0.62183     0.31144]
# Recall (Mask):    [    0.53534      0.1152]
# mAP50 (Mask):     0.3160510646926501
# mAP50-95 (Mask):  0.1360566252793251

# TTA
# Precision (Box): [    0.65359     0.31094]
# Recall (Box):    [    0.60553     0.12952]
# mAP50 (Box):     0.3607012942407141
# mAP50-95 (Box):  0.19393424210070176
# Precision (Mask): [    0.62183     0.31144]
# Recall (Mask):    [    0.53534      0.1152]
# mAP50 (Mask):     0.3160510646926501
# mAP50-95 (Mask):  0.1360566252793251