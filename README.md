# Learning a Motion Model for Multiple Object Tracking

This repository provides the implementation for my master's thesis, where a motion model for MOT has been developed. The motion model aims at maximizing the IoU between object bounding boxes of consecutive frames, thus facilitating the performance of a _Tracktor_ tracker [Bergmann et al., 2019]. As features, the model exploits both the previous trajectory of an object and correlations between the convolutional feature maps of adjacent frames.

Our model significantly increases Tracktor performance, achieving scores comparable to much slower baseline methods in the case of non-occluded objects. Further, we combine our model and a motion compensation algorithm into a hybrid method which -- at the time of publishing -- is able to outperform the best baseline on the test set, while reaching far superior performance at low frame rates.

## Installation

1. Follow the instructions of how to install the Tracktor framework, as provided at https://github.com/phil-bergmann/tracking_wo_bnw#installation. However, instead of their repository, clone this one during step 1. The object detector weights used for training the provided motion model can be downloaded [here](https://drive.google.com/file/d/1PfUlHiwaxZG3aKmC8geV34j-7Se8cbWH/view?usp=sharing), in case of changes in the upstream repository.

2. Compile the feature correlation layer [Fischer et al., 2015] that is provided alongside our code:

      ```bash
      cd src/tracktor/motion/correlation
      pip install .
      ```
   
    You can change the `nvcc_args` defined in `setup.py` to match your GPU architecture.
    
3. Download the final correlation model weights from [here](https://drive.google.com/file/d/1P8YW-KIjq9BRuy42x5rzfKlFlKjWCqi3/view?usp=sharing) and move the file into the `output/motion` directory.

4. Download the PCA parameters for feature dimensionality reduction from [here](https://drive.google.com/file/d/1Hi_oLnCIWqs9a9jkLtl5unt_RJfEwb5V/view?usp=sharing) and move the file into the `output/pca` directory.

## Evaluate Tracktor with the motion model

Tracktor configuration is done through the `experiments/cfgs/tracktor.yaml`  file. The correlation model can be configured by changing the `experiments/cfgs/correlation_model.yaml` file. By default, the hybrid model with `CMCCVA-reID` inactive strategy and an inactive patience of 80 frames is evaluated.

2. Run Tracktor by executing:

  ```bash
  python experiments/scripts/test_tracktor.py with correlation_model
  ```

3. The results are logged into the corresponding `output` directory.

For reproducibility, the train and test set results on the MOT17 dataset are supplied in the following:

```
********************* MOT17 TRAIN Results *********************
IDF1  IDP  IDR| Rcll  Prcn|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP
69.0 88.7 56.4| 63.1  99.2| 1638 553  718  367|  1639 124167  552  1342|  62.5  89.6

********************* MOT17 TEST Results *********************
IDF1  IDP  IDR| Rcll  Prcn|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP
60.2 80.3 48.1| 58.3  97.4| 2355 499 1027  829|  8839 235129  1333 3715|  56.5  78.9
```

## Training

The training configuration can be found in `experiments/cfgs/train_motion_im.yaml`. With our code, we supply a training framework which takes care of validating, logging, checkpointing and automatic resuming of training runs. Start training by running

```bash
python experiments/scripts/train_motion_im.py with correlation_model train.name=<training_run_name>
```

Checkpoints and logs will be saved into the `output/motion/<training_run_name>` directory.