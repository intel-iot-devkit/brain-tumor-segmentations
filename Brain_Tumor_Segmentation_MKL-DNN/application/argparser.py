#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
This module just reads parameters from the command line.
"""

import argparse
import settings    # Use the custom settings.py file for default parameters
import os

#parser = argparse.ArgumentParser(
#    description="Trains 2D U-Net model (Keras/TF) on BraTS dataset.",
#    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser()
parser.add_argument("--data_filename", default=settings.DATA_FILENAME,
                    help="the HDF5 data filename")
parser.add_argument("-r", "--results_directory", default="../results/",
                    help="the folder to save results")
parser.add_argument("--inference_filename", default=settings.INFERENCE_FILENAME,
                    help="the Keras inference model filename")

args = parser.parse_args()

