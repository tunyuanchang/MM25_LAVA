# script for image preprocessing
#!/bin/bash

cd ../preprocessing
python crop_and_resize.py --s 1 --d train
python crop_and_resize.py --s 5 --d train
python crop_and_resize.py --s 6 --d train
python crop_and_resize.py --s 7 --d train
python crop_and_resize.py --s 8 --d train

python crop_and_resize.py --s 9 --d test