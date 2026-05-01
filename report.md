# the thing

## 1. justify size choices

Testing on nano, final on small because I'm alone and starting late. TODO: justify augmentation amount, describe models??

## 2. look at the dataset

208171 forests, 9614 bushes. Gonna need to help the bushes out a bit, they're, like, 1 out of 23 things.
![208171 forests, 9614 bushes](image.png)

## 3. augment

Many images look very similar save for being rotated and flipped, and, to my knowledge, YOLO does a lot of geometric augmentation, so I'm only going to mess with colors. This also saves me from having to recalculate the labels. I don't need to bring the class amounts exactly up to par because YOLO also (apparently) uses Focal Loss, but I'll prioritise duplicating bushes to help out a little.