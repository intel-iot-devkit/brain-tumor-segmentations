: '
  Copyright (c) 2018 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
'

# Update the apt repository
sudo apt-get update
env1=stocktf
env2=mkltf

# Create conda environment for tensorflow
ENVS=$(conda env list | awk '{print $env1}' )
if [[ $ENVS = *"$env1"* ]]; then
   echo "$env1 environment exists"
else 
   conda config --set auto_activate_base false
   conda env create -f "stocktf.yml"
fi

# Create conda environment for MKL
ENVS=$(conda env list | awk '{print $env2}' )
if [[ $ENVS = *"$env2"* ]]; then
   echo "$env2 environment exists"
else 
   conda config --set auto_activate_base false
   conda env create -f "mkltf.yml"
fi

