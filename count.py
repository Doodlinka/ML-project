import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

forests = 0
bushes = 0

with open('train_20000_split.txt') as f:
    paths = [line.strip().replace("images", "labels").replace("png", "txt") for line in f if line.strip()]

for path in paths:
    try:
        with open(path) as file:
            lines = file.readlines()
    except FileNotFoundError: continue
    for i in range(len(lines)-1, -1, -1):
        # just in case a wrong class has more than 1 digit
        cls = lines[i].split(maxsplit=1)[0]
        if cls == '0': forests += 1
        elif cls == '1': bushes += 1

# 208171 9614
print(forests, bushes)
class_names = ['Forests', "Bushes"]
plt.bar(class_names, [forests, bushes], color=['red', 'blue'])
plt.ylabel('Instance Count')
plt.title('Class Distribution')
plt.show()