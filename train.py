import os, time
from ultralytics import YOLO
from trainparams import *


# why can't yolo take an in-memory dict? the mind boggles
def makeconfig(inpath, outpath, trainfile):
    with open(inpath) as f:
        config = f.readlines()
    config[0] = f"path: {os.path.abspath(os.path.dirname(__file__))}\n"
    config[1] = f"train: {trainfile}\n"
    with open(outpath, "w") as f:
        for line in config: f.write(line)

if __name__ == "__main__":
    start = time.time()
    os.chdir(PROJDIR)
    model = YOLO(MODEL)
    makeconfig("dataset.yaml", CONFNAME, DATASET)
    results = model.train(
        project='ForestSeg',
        **PARAMS
    )
    print(f"\n\n{time.time() - start} s\n\n")