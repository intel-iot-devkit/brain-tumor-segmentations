#!/bin/bash

# Copyright (c) 2018 Intel Corporation.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#Install the dependencies
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install numpy jupyter h5py
sudo apt-get install python3-matplotlib
sudo apt-get install python3-tk

BASE_DIR=`pwd`

#Optimize the model
cd /opt/intel/openvino/deployment_tools/model_optimizer/

python3.5 mo_tf.py --input_model $BASE_DIR/resources/saved_model_frozen.pb  --output_dir $BASE_DIR/resources/output/IR_models/FP32/ --input_shape=[1,144,144,4] --data_type FP32 --model_name saved_model
python3.5 mo_tf.py --input_model $BASE_DIR/resources/saved_model_frozen.pb  --output_dir $BASE_DIR/resources/output/IR_models/FP16/ --input_shape=[1,144,144,4] --data_type FP16 --model_name saved_model
