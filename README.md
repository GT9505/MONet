# MONet
Using Multi-label Classification to Improve Object Detection

MONet is modified based on [py-R-FCN-priv](https://github.com/soeaver/py-RFCN-priv), thanks for soeaver's job.


### Disclaimer

The official R-FCN code (written in MATLAB) is available [here](https://github.com/daijifeng001/R-FCN).

The official R-FCN code (written in PYTHON) is available [here](https://github.com/YuwenXiong/py-R-FCN).

 
### Installation

1. Clone the MONet repository
    ```Shell
    git clone https://github.com/GT9505/MONet
    ```
    We'll call the directory that you cloned MONet into `MONET_ROOT`

2. Build the Cython modules
    ```Shell
    cd $MONET_ROOT/lib
    make
    ```
    
3. Build Caffe and pycaffe
    ```Shell
    cd $MONET_ROOT/caffe
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    
    # cp Makefile.config.example Makefile.config
    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j && make pycaffe -j
   ```    
   
   **Note:** Caffe *must* be built with support for Python layers!
    ```make
    # In your Makefile.config, make sure to have this line uncommented
    WITH_PYTHON_LAYER := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    # NCCL (https://github.com/NVIDIA/nccl) is necessary for multi-GPU training with python layer
    USE_NCCL := 1
    ```
   
### Preparation for Training & Testing

Please follow the official [py-R-FCN](https://github.com/YuwenXiong/py-R-FCN) code to preparation training set and testing set

Please download backbone network ResNet-101 in [here](https://drive.google.com/open?id=1Uh3vxUf445nWoorejeEWFzkGVb2ZEE2W)

### Start training
The usage is same as [py-R-FCN-priv](https://github.com/soeaver/py-RFCN-priv)

	cd $MONet_ROOT
	./experments/scripts/monet_end2end_ohem_multi_gpu.sh 0 pascal_voc

### Results on PASCAL VOC 2007

Using the default hyperparameters and iterations, you can achieve a mAP around 83.0%. [The model with 83.0% mAP](https://drive.google.com/open?id=1ucVg7o964DRZF_idDpVYuOagrIpJMmsM) 
	
### License

MONet is released under the MIT License (refer to the LICENSE file for details).

