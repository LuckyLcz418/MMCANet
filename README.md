# Scribble-Supervised RGB-T Salient Object Detection with Segment Anything Model-Guided Pseudo-Label Generation
![](https://github.com/LuckyLcz418/FileInsert/blob/main/Snipaste_2024-03-08_16-26-48.jpg)

## Framework
![](https://github.com/LuckyLcz418/FileInsert/blob/main/newFramework.jpg)

## Download
- Scribble datasets:
    Google Drive:
    Baidu:
- Pseudo labels:
    Google Drive: https://drive.google.com/file/d/1pnbZAFPQIXMX1mtF6oFyJtmmQZ57efM9/view?usp=drive_link
    Baidu: https://pan.baidu.com/s/1jyy-A45xt9N0KeVur7FwZA  提取码：1234
- Test datasets: 
    Google Drive:
    Baidu:
- SAM model parameters: 
    Google Drive: https://drive.google.com/file/d/1JrXxFSXiqJtB4iN2ExTabgocaK6HJy9D/view?usp=drive_link
    Baidu: https://pan.baidu.com/s/1DBCb2VBUMcdIbJL6G1Sk9A  提取码：1234
- Backbone parameters: 
    Google Drive: https://drive.google.com/file/d/1rXda5bje95ork-QLacVA258FgBw2M5kZ/view?usp=drive_link
    Baidu: https://pan.baidu.com/s/1AwBka4VW6HH2XjWlzpLrmw  提取码：1234
- Pre-trained parameters:
    Google Drive: https://drive.google.com/file/d/1cCeL4G_eo_GApRYVokheogfsew6VQOGy/view?usp=drive_link
    Baidu: https://pan.baidu.com/s/1shjx13fpJzDCELz0-hb_5g  提取码：1234
- Prediction results: 
    Google Drive: https://drive.google.com/file/d/1ZFcP4VnA-KvSF6TGUHOZ-2Qg4-wYI9r3/view?usp=drive_link
    Baidu: https://pan.baidu.com/s/1utF3fppMpfT45-QWCo6tOA  提取码：1234

## Usage

### Prepare
1. Please use conda to install torch (1.12.1) and torchvision (0.13.1).
2. Install other packages: pip install -r requirements.txt.

### Train Model
1. Download the scribble datasets and the SAM model parameters and generate the pseudo labels. We have provided the pseudo labels.
2. Download the backbone parameters and set your path of all datasets or backbone parameters correctly. During the training process, we use the scribbel datasets and the pseudo labels to supervise the model training.
3. Run ./train.py
### Test Model
1. Download the pre-trained parameters(we have provided it, also, you can get it through the above training process) and test datasets.
2. Run ./test.py

### Evalution Code
code: https://github.com/lartpang/PySODEvalToolkit

## Comprison Experiments with SOTA methods
![](https://github.com/LuckyLcz418/FileInsert/blob/main/experiment1.jpg)
![](https://github.com/LuckyLcz418/FileInsert/blob/main/experiment2.jpg)
![](https://github.com/LuckyLcz418/FileInsert/blob/main/visualization.jpg)
