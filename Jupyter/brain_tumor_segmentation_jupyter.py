#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
import numpy as np
import h5py
import time 
from inference import Network
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

TARGET_DEVICE = 'CPU'
model_xml = ''
CPU_EXTENSION = ''
png_directory = ''
number_iter = 5
rows_per_image = 4
data_file = ''
stats = False

def env_parser():
    """
    Parse the environment variables
    """

    global png_directory, data_file, model_xml, TARGET_DEVICE, CPU_EXTENSION, stats, rows_per_image, number_iter 

    if 'DEVICE' in os.environ:
        TARGET_DEVICE = os.environ['DEVICE'] 
    if 'MODEL' in os.environ:
        model_xml = os.environ['MODEL']
    if 'CPU_EXTENSION' in os.environ:
        CPU_EXTENSION = os.environ['CPU_EXTENSION']
    if 'RESULTS' in os.environ:
        png_directory = os.environ['RESULTS']
    if 'NUMBER_ITER' in os.environ:
        number_iter = os.environ['NUMBER_ITER']
    if 'ROWS_PER_IMAGE' in os.environ:
        rows_per_image = os.environ['ROWS_PER_IMAGE']
    if 'DATA_FILE' in os.environ:
        data_file = os.environ['DATA_FILE']
    if 'STATS' in os.environ:
        stats = os.environ['STATS']


def check_args():
    """
    Checks the Model and Data File
    """

    global model_xml, data_file
    if model_xml == '':
        print("You need to specify the path to the .xml file")
        print("Export the path using 'MODEL' environmental variable  ")
        sys.exit(1)
    if data_file == '':
        print("You need to specify the path to the .h5 file")
        print("Export the path using 'DATA_FILE' environmental variable  ")
        sys.exit(1)    

def print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob):
    """
    Prints layer by layer inference times.
    Good for profiling which ops are most costly in your model.
    """

    # Start sync inference
    print("Starting inference ({} iterations)".format(number_iter))
    infer_time = []

    for i in range(args.number_iter):
        input_data_transposed_1=input_data[0:batch_size].transpose(0,3,1,2)
        t0 = time.time()
        res = exec_net.infer(inputs={input_blob: input_data_transposed_1[:,:n_channels]})
        infer_time.append((time.time() - t0) * 1000)


    average_inference = np.average(np.asarray(infer_time))
    print("Average running time of one batch: {:.5f} ms".format(average_inference))
    print("Images per second = {:.3f}".format(batch_size * 1000.0 / average_inference))

    perf_counts = exec_net.requests[0].get_perf_counts()
    log.info("Performance counters:")
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format("name",
                                                         "layer_type",
                                                         "exec_type",
                                                         "status",
                                                         "real_time, us"))
    for layer, stats in perf_counts.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                             stats["layer_type"],
                                                             stats["exec_type"],
                                                             stats["status"],
                                                             stats["real_time"]))

def load_data():
    """
    Modify this to load your data and labels
    """

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    df = h5py.File(data_fn, "r")
    imgs_validation = df["imgs_validation"]
    msks_validation = df["msks_validation"]
    img_indicies = range(len(imgs_validation))

    """
    OpenVINO uses channels first tensors (NCHW).
    TensorFlow usually does channels last (NHWC).
    So we need to transpose the axes.
    """
    input_data = imgs_validation
    msks_data = msks_validation
    return input_data, msks_data, img_indicies


def calc_dice(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def plotDiceScore(img_no,img,msk,pred_mask,plot_result, time):
    dice_score = calc_dice(pred_mask, msk)

    if plot_result:
        plt.figure(figsize=(15, 15))
        plt.suptitle("Time for prediction: {} ms".format(time), x=0.1, y=0.70,  fontsize=20, va="bottom")
        plt.subplot(1, 3, 1)
        plt.imshow(img[0,0,:,:], cmap="bone", origin="lower")
        plt.axis("off")
        plt.title("MRI Input", fontsize=20)
        plt.subplot(1, 3, 2)
        plt.imshow(msk[0,0, :, :], origin="lower")
        plt.axis("off")
        plt.title("Ground truth", fontsize=20)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[0,0, :, :], origin="lower")
        plt.axis("off")
        plt.title("Prediction\nDice = {:.4f}".format(dice_score), fontsize=20)

        plt.tight_layout()

        png_name = os.path.join(png_directory, "pred{}.png".format(img_no))
        plt.savefig(png_name, bbox_inches="tight", pad_inches=0)


env_parser()
check_args()

# Create output directory for images
if not os.path.exists(png_directory):
    os.makedirs(png_directory)

data_fn = data_file
if not os.path.exists(data_fn):
    print("Wrong input path or File not exists")
    sys.exit(1)


log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

Net = Network()
log.info("Loading U-Net model to the plugin")
[batch_size, n_channels, height, width], exec_net, input_blob, out_blob = Net.load_model(model_xml, TARGET_DEVICE, 1, 1, 0, CPU_EXTENSION)[1:5]

# Load data
input_data, label_data, img_indicies = load_data()

if stats:
    # Print the latency and throughput for inference
    print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob)

val_id = 1
infer_time = 0
process_time_start = time.time()

for idx in img_indicies:

    input_data_transposed=input_data[idx:(idx+batch_size)].transpose(0,3,1,2)
    start_time = time.time()
    res = exec_net.infer(inputs={input_blob:input_data_transposed[:,:n_channels]})
    # Save the predictions to array
    predictions = res[out_blob]
    time_elapsed = time.time()-start_time
    infer_time += time_elapsed
    plotDiceScore(idx,input_data_transposed,label_data[[idx]].transpose(0,3,1,2),predictions,True, round(time_elapsed*1000))
    val_id += 1

total_time = time.time() - process_time_start
with open(os.path.join(png_directory, 'stats.txt'), 'w') as f:
                f.write(str(round(infer_time, 4))+'\n')
                f.write(str(val_id)+'\n')
                f.write("Frames processed per second = {}".format(round(val_id/infer_time)))

print("\nThe results are stored in '{}' directory".format(png_directory))

Net.clean()