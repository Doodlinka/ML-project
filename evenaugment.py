from augment import *
from pathlib import Path
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    with open('train_split.txt', 'r') as f:
        baseimgs = [Path(line.strip()).resolve().as_posix() for line in f if line.strip()]
    if not baseimgs:
        print("err: split file not found")
        exit(1)
    create_several_augs(baseimgs, baseimgs, iter_count=6)