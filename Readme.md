# Code for HongKong-3D datasets Semantic Segmentation

---


<center>  
    <img src ="/pic/pointsWithColor.png"
    width="45%">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src ="pic/labeledPoints.png"
    width="45%"> <br>
</center>

<center> points with color
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
labeled points</center>

---

## Requirements

**The code is tested on the following environment**

- Python 3.9
- Pytorch 1.12.1
- laspy 2.2.2
- open3d 0.15.0
- matplotlib 3.5.1
- numpy 1.24.2
- pathlib 1.0.1
- CUDA 11.3
- cudnn 8.2.1

## Dataset

- Original dataset: HongKong-3D(.las/.laz format)
- Processed dataset: HongKong-3D(.ply format)

## Usage

### Data Preprocessing

- Using `/utils/data_prepare_hongkong.py` to preprocess the original dataset.
- Using `/utils/generate_augmented_pc.py` to generate augmented point clouds.

### Training

- Using `/main_HongKong.py` to train the model. You can try like following (more detail in the code):

>python main_HongKong.py --name NAME --log_dir LOG_DIR --max_epoch EPOCH --gpu GPU_ID --vla_split VAL_SPLIT

### Testing

- Using `/test_HongKong.py` to test the model. You can try like following (more detail in the code):

>python test_HongKong.py --checkpoint_path CHECKPOINT_PATH --name NAME --log_dir LOG_DIR

### Pretrained Model
Trained models are provided for use, for details see [Pretrained Model](/train_output/HongKong_Augm_All_Track_B_5/checkpoint.tar)

### config

- Using `/helper_tool.py` to set the parameters(including bath_size. etc).

## Results
<center>  
    <br><br>
    <img src ="/pic/HKResult.png"
    width="80%"><br>
</center>


## Reference
- Out work are based on the [RandLA-Net](https://github.com/QingyongHu/RandLA-Net)

## Concact Us
Please contact us if you have any questions during use or if you have any other needs.
- Email: yangzhou97@126.com
