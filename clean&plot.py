import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

forests = 0
bushes = 0

def clean(path):
    global forests, bushes
    mustclean = False
    with open(path) as file:
        lines = file.readlines()
    for i in range(len(lines)-1, -1, -1):
        # just in case a wrong class has more than 1 digit
        cls = lines[i].split(maxsplit=1)[0]
        if cls == '0': forests += 1
        elif cls == '1': bushes += 1
        else:
            lines.pop(i)
            mustclean = True
    if mustclean:
        with open(path, 'w') as file:
            file.writelines(lines)

for root, _, files in os.walk("./labels"):
    for fname in files:
        clean(os.path.join(root, fname))

# 208171 9614
print(forests, bushes)
class_names = ['Forests', "Bushes"]
plt.bar(class_names, [forests, bushes], color=['red', 'blue'])
plt.ylabel('Instance Count')
plt.title('Class Distribution')
plt.show()