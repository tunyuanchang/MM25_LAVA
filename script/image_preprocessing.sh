# script for image preprocessing
#!/bin/bash

cd ../preprocessing
echo training set
python crop_and_resize.py --s 1 --d image_train
python crop_and_resize.py --s 5 --d image_train
python crop_and_resize.py --s 6 --d image_train
python crop_and_resize.py --s 7 --d image_train
python crop_and_resize.py --s 8 --d image_train

echo testing set
python crop_and_resize.py --s 9 --d image_test

echo done