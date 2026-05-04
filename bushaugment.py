from augment import *

def get_imgs_to_aug(baseimgs):
    targetimgs = []
    ones = 0
    total = 0
    uniqueimgs = 0
    for img_path in baseimgs:
        lbl_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                curtotal = 0
                cur1s = 0
                for line in f:
                    curtotal += 1
                    if line.startswith('1'): cur1s += 1
                weight = int(10 * ((cur1s / curtotal))) #  ** 2))
                if weight: uniqueimgs += 1
                for _ in range(weight): 
                    total += curtotal
                    ones += cur1s
                    targetimgs.append(img_path)
    print(len(targetimgs), uniqueimgs, ones, total)
    return targetimgs
    
    
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    with open('train_split.txt', 'r') as f:
        baseimgs = [Path(line.strip()).resolve().as_posix() for line in f if line.strip()]
    if not baseimgs:
        print("err: split file not found")
        exit(1)
    targetimgs = get_imgs_to_aug(baseimgs)
    create_several_augs(baseimgs, targetimgs, iter_count=4)