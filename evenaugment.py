from augment import *
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    with open('train_split.txt', 'r') as f:
        baseimgs = [line.strip() for line in f if line.strip()]
    if not baseimgs:
        print("err: split file not found")
        exit(1)
    create_several_augs(baseimgs, baseimgs, aug_count=5000, iter_count=8)