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

"""
This module just reads parameters from the command line.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", help="the path to the data", required=True)
parser.add_argument("-r", "--results_directory", default="inference_examples",
                    help="the folder to save results")
parser.add_argument("--keras_api", help="use keras instead of tf.keras",
                    action="store_true",default=True)

# OpenVINO arguments
parser.add_argument("-number_iter", "--number_iter",
                        help="Number of iterations", default=5, type=int)
parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. "
                             "Absolute path to a shared library with "
                             "the kernels impl.", type=str)
parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)", default="CPU",
                        type=str)
parser.add_argument("-rows_per_image", "--rows_per_image",
                        help="Number of rows per plot (when -plot = True)",
                        default=4, type=int)
parser.add_argument("-stats", "--stats", help="Plot the runtime statistics",
                        default=False, action="store_true")

parser.add_argument("-m", "--model", help="Path to the .xml file", required=True)


args = parser.parse_args()