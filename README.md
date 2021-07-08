# Self-Supervised Learning for MRI Reconstruction with a Parallel Network Training Framework. (MICCAI 2021, official code)
### Abstract
Image reconstruction from undersampled k-space data plays an important role in accelerating the acquisition of MR data, and a lot of deep learning-based methods have been exploited recently. Despite the achieved inspiring results, the optimization of these methods commonly relies on the fully-sampled reference data, which are time-consuming and difficult to collect. To address this issue, we propose a novel self-supervised learning method. Specifically, during model optimization, two subsets are constructed by randomly selecting part of k-space data from the undersampled data and then fed into two parallel reconstruction networks to perform information recovery. Two reconstruction losses are defined on all the scanned data points to enhance the network’s capability of recovering the frequency information. Meanwhile, to constrain the learned unscanned data points of the network, a difference loss is designed to enforce consistency between the two parallel networks. In this way, the reconstruction model can be properly trained with only the undersampled data. During the model evaluation, the undersampled data are treated as the inputs and either of the two trained networks is expected to reconstruct the high-quality results. The proposed method is flexible and can be employed in any existing deep learning-based method. The effectiveness of the method is evaluated on an open brain MRI dataset. Experimental results demonstrate that the proposed self-supervised method can achieve competitive reconstruction performance compared to the corresponding supervised learning method at high acceleration rates (4 and 8).
![network_train](/figs/network_train.png)
<p align="center">
(a) Training phase
</p>

![network_test](/figs/network_test.png)
<p align="center">
(b) Test phase
</p>

<p align="center">
The pipeline of our proposed framework for self-supervised MRI reconstruction.
</p>

## How to use
This project is conducted on an Ubuntu 18.04 LTS (64-bit) operating system utilizing two NVIDIA RTX 2080 Ti GPUs (each with a memory of 11GB). The following we will explain how to use this code to achieve self-supervised MRI reconstruction.

### Clone repository
```bash
git clone https://github.com/chenhu96/Self-Supervised-MRI-Reconstruction.git
```

### Download dataset
Download the [IXI Dataset T1 images](http://brain-development.org/ixi-dataset/) and divide them into three disjoint parts: training, validation, and test. Change the path in **main.py** and **testdemo.py** correctly.

### Install dependencies
This code is tested with Python3.7. We use [Anaconda](https://www.anaconda.com/) to manage the Python environment. Run the following scripts to create a clean environment and install library dependencies used in this project:
```bash
conda create -n SSLMRI python=3.7
conda activate SSLMRI
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Training phase
Enter the path of the project and run the following scripts to train the parallel network.
```bash
python main.py -m='train' -trsr=0.03 -vsr=0.01
```
or
```bash
python main.py -m='train' -trsr=0.03 -vsr=0.01 -pt=true
```

### Test phase
Enter the path of the project and run the following scripts to test the saved model.
```bash
python main.py -m='test' -tesr=0.01
```
or
```bash
python testdemo.py -tesr=0.01
```
Noting that the reconstructed images of the two networks are roughly the same since they are basically recovering the same thing. Thus, in the test phase, the obtained undersampled data is fed to either of the two trained networks to generate the high-quality results.

## Saved information
The trained models at different acceleration rates (4 and 8) and the corresponding loss curves in the training phase are saved in the folder named **saved_infor**. You can run the following script to show the loss curve in tensorboard.
```bash
tensorboard --logdir dir/loss_curve
```

## Results
![results](/figs/results.png)
<p align="center">
Example reconstruction results of the different methods at two acceleration rates (4  and 8) along with their corresponding error maps.
</p>

It can be observed that our method recovers more details structural information compared to U-Net-256 and SSDU. Moreover, our self-supervised learning method generates very competitive results which are close to those generated by the corresponding supervised learning method that utilizes the exact same network architecture.

## Citation
If you find this code is helpful for your research or work, please cite the following paper.

`Self-Supervised Learning for MRI Reconstruction with a Parallel Network Training Framework. (MICCAI 2021)`

## Acknowledgments
[1]. [IXI Dataset](http://brain-development.org/ixi-dataset/)

[2]. Zhang, J., Ghanem, B.: ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)