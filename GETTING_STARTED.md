# Getting Started
This document briefly describes how to install and use the code.

## Environment
We conducted experiments in the following environment:
 - Linux
 - Python 3
 - TITAN X GPUs with CuDNN
 - FFmpeg

Similar environments (e.g. with OSX, Python 2) might work with small modification, but not tested.


## Datasets

HMDB-51 and UCF-101 can be downloaded with the following scripts.
```bash
cd data
./get_hmdb51_data.sh
./get_ucf101_data.sh
```
The following calls reencode the downloaded videos into a mpeg4 format that has the same GOP structure as described in paper, and resize them to 340x256. (We used [an environment](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/environment.yml) that uses FFmpeg 3.1.3. )
```bash
./reencode.sh hmdb51/videos/ hmdb51/mpeg4_videos/
./reencode.sh ucf101/UCF-101/ ucf101/mpeg4_videos/
```

## Data loader

This data loader that directly takes a compressed video and returns compressed representation (I-frame, motion vectors, or residual) as a numpy array.
In our experiments, it's fast enough so that it doesn't delay GPU training with 8 CPU workers.

#### Supported video format
Currently we only support mpeg4 raw videos. Other codecs, e.g. H.264, coming soon. The mpeg4 raw videos can be obtained using FFmpeg:

`ffmpeg -i input.mp4 -c:v  -c:v mpeg4 -f rawvideo output.mp4`

#### Install
 - Download FFmpeg (`git clone https://github.com/FFmpeg/FFmpeg.git`).
 - Go to FFmpeg home,  and `git checkout 74c6a6d3735f79671b177a0e0c6f2db696c2a6d2`.
 - `make clean`
 - `./configure --prefix=${FFMPEG_INSTALL_PATH} --enable-pic --disable-yasm --enable-shared`
 - `make`
 - `make install`
 - If needed, add `${FFMPEG_INSTALL_PATH}/lib/` to `$LD_LIBRARY_PATH`.
 - Go to `data_loader` folder.
 - Modify `setup.py` to use your FFmpeg path (`${FFMPEG_INSTALL_PATH}`).
 - `./install.sh`

#### Usage
(Please feel free to ignore this part if you just want to use CoViAR
but not use this data loader independently. )  
The data loader has two functions: `load` for loading a representation and `get_num_frames` for
counting the number of frames in a video.

The following call returns one frame (specified by `frame_index=0,1,...`) of one GOP
(specified by `gop_index=0,1,...`).
```python
from coviar import load
load([input], [gop_index], [frame_index], [representation_type], [accumulate])
```
 - input: path to video (.mp4).
 - representation_type: `0`, `1`, or `2`. `0` for I-frames, `1` for motion vectors, `2` for residuals.
 - accumulate: `True` or `False`. `True` returns the accumulated representation. `False` returns the original compressed representations. (See paper for details. )

For example, 
```
load(input.mp4, 3, 8, 1, True)
```
returns the accumulated motion vectors of the 9th frame of the 4th GOP.

## Training

For example, we used the following commands to train on HMDB-51.
```bash
# I-frame model.
python train.py --lr 0.0003 --batch-size 40 --arch resnet152 \
 	--data-name hmdb51 --representation iframe \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split1_train.txt \
 	--test-list data/datalists/hmdb51_split1_test.txt \
 	--model-prefix hmdb51_iframe_model \
 	--lr-steps 55 110 165  --epochs 220 \
 	--gpus 0 1

# Motion vector model.
python train.py --lr 0.005 --batch-size 80 --arch resnet18 \
 	--data-name hmdb51 --representation mv \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split1_train.txt \
 	--test-list data/datalists/hmdb51_split1_test.txt \
 	--model-prefix hmdb51_mv_model \
 	--lr-steps 120 200 280  --epochs 360 \
 	--gpus 0

# Residual model.
python train.py --lr 0.001 --batch-size 80 --arch resnet18 \
 	--data-name hmdb51 --representation residual \
 	--data-root data/hmdb51/mpeg4_videos \
 	--train-list data/datalists/hmdb51_split1_train.txt \
 	--test-list data/datalists/hmdb51_split1_test.txt \
 	--model-prefix hmdb51_residual_model \
 	--lr-steps 120 180 240  --epochs 300 \
 	--gpus 0

```
and for UCF-101, 
```bash
# I-frame model.
python train.py --lr 0.0003 --batch-size 80 --arch resnet152 \
 	--data-name ucf101 --representation iframe \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_iframe_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0 1 2 3

# Motion vector model.
python train.py --lr 0.01 --batch-size 80 --arch resnet18 \
 	--data-name ucf101 --representation mv \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_mv_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0

# Residual model.
python train.py --lr 0.005 --batch-size 80 --arch resnet18 \
 	--data-name ucf101 --representation residual \
 	--data-root data/ucf101/mpeg4_videos \
 	--train-list data/datalists/ucf101_split1_train.txt \
 	--test-list data/datalists/ucf101_split1_test.txt \
 	--model-prefix ucf101_residual_model \
 	--lr-steps 150 270 390  --epochs 510 \
 	--gpus 0

```
These commands train on split-1 of both datasets.
Please modify arguments `--train-list` and `--test-list`
accordingly for training/testing on a different split. 

The hyperparameters here are slightly different from those used in the original paper, 
because the pre-trained weights are different. 
The original paper uses ResNet (pre-activation) pre-trained by MXNet, here we use ResNet (non-pre-activation) pre-trained by PyTorch. They offer similar results.

## Testing

Given a trained model,
```bash
python test.py --gpus 0 \
	--arch [architecture] \
	--data-name [data_name] --representation [representation] \
	--data-root [data root] \
	--test-list [test list] \
	--weights [path to model].pth.tar \
	--save-scores [output score filename]

```
performs full evaluation on the test set, and stores the results in `[output score filename]`.

## Combining models to get final results
After getting the evaluation results for each decoupled model using `test.py`,
we use `combine.py` to combine the results and calculate
the final accuracy.
```bash
python combine.py --iframe ${iframe_score_file} \
	--mv ${mv_score_file} \
	--res ${residual_score_file}
```
