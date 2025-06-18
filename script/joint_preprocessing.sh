# script for joint preprocessing
#!/bin/bash

cd ../preprocessing
python joint_comp.py --f joint_train --t True
python joint_comp.py --f joint_test --t False