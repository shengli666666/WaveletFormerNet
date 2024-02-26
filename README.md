# WaveletFormer: A Transformer-based Wavelet Network for Real-world Non-homogeneous and Dense Fog Removal

>__Abstract:__ Although deep convolutional neural networks have achieved remarkable success in removing synthetic fog, it is essential to be able to process images taken in complex foggy conditions, such as dense or non-homogeneous fog, in the real world. However, the haze distribution in the real world is complex, and downsampling can lead to color distortion or loss of detail in the output results as the resolution of a feature map or image resolution decreases. Moreover, the over-stacking of convolutional blocks might increase the model complexity. In addition to the challenges of obtaining sufficient training data, overfitting can also arise in deep learning techniques for foggy image processing, which can limit the generalization abilities of the model, posing challenges for its practical applications in real-world scenarios.
>
>Considering these issues, this paper proposes a Transformer-based wavelet network (WaveletFormerNet) for real-world foggy image recovery.
We embed the discrete wavelet transform into the Vision Transformer by proposing the WaveletFormer and IWaveletFormer blocks, aiming to alleviate texture detail loss and color distortion in the image due to downsampling. We introduce parallel convolution in the Transformer block, which allows for the capture of multi-frequency information in a lightweight mechanism. Such a structure reduces computational expenses and improves the effectiveness of the network. Additionally, we have implemented a feature aggregation module (FAM) to maintain image resolution and enhance the feature extraction capacity of our model, further contributing to its impressive performance in real-world foggy image recovery tasks.
>
>Through extensive experiments on real-world fog datasets, we have demonstrated that our WaveletFormerNet achieves superior performance compared to state-of-the-art methods, as shown through quantitative and qualitative evaluations of minor model complexity.
>
>Additionally, our satisfactory results on real-world dust removal and application tests showcase the superior generalization ability and improved performance of WaveletFormerNet in computer vision-related applications compared to existing state-of-the-art methods, further confirming our proposed approach's effectiveness and robustness.

>__Highlights:__ We propose the WaveletFormer and IWaveletFormer blocks to alleviate texture detail loss and maintain image resolution, combining the advantages of Vision Transformer and wavelet transform. The parallel convolution in the WaveletFormer and IWaveletFormer blocks captures the multi-frequency information in the lightweight mechanism, decreasing the network computational expenses.
>
>The feature aggregation module is proposed to capture the long-range multi-frequency information in the lightweight mechanism and further enhance the feature extraction capability of WaveletFormerNet.
>
>We present WaveletFormerNet, an end-to-end wavelet reconstruction network guided by frequency information to tackle image dehazing problems under complex, hazy conditions in the real world. To validate the effectiveness of WaveletFormerNet, we conducted extensive experiments on both synthetic and real-world datasets. The results demonstrate that our method yields competitive de-fogging performance in comparison to state-of-the-art methods.

* [Network Architecture]()
* [Dependencies and Installation]()
* [Datasets Preparation]()
    * [Synthetic Fog Dataset]()
    * [Real-world Fog Datasets]()
    * [Real world Dust Dataset]()
* [Quick Run]()
* [Quantitative comparisons]()
    * [Comparison of referenced evaluation on real-world dehazing]()
    * [Comparison of non-referenced evaluation on real-world dehazing]()
    * [Comparison on synthetic dataset dehazing]()
    * [Parameters and Runtime Analysis]()
* [Qualitative Comparisons]()
    *[Comparisons on the on the natural hazy images and RTTS dataset]()
* [Generality Analysis for WaveletFormerNet]()
* [Application Test]()
* [Ablation Experiments]()

## Network Architecture

As we have uploaded a lot of pictures, the page may refresh slowly. Thank you very much for your patience!

### WaveletFormerNet Network

<center>
    <img src='network/allnetwork.png'>
</center>

(Ⅰ) is the overall structure of the proposed WaveletFormerNet network; WaveletFormerNet is a Swin Transformer block-based wavelet network with encoding and decoding structure for real-world fog removal. (Ⅱ) and (Ⅲ) is the structure of our WaveletFormer block and feature aggregation module (FAM). The WaveletFormer block and the IWaveletFormer block have the same structure: they utilize DWT and IDWT to substitute downsampling and upsampling, respectively.

### WaveletFormer Block and Feature Aggregation Module

|<img src="network/wformer.png">|<img src="network/fam.png">|
|:-:|:-:|

## Dependencies and Installation

* python3.9
* PyTorch>=1.0.1
* NVIDIA GPU+CUDA>=11.4
* numpy
* matplotlib
* pytorch_wavelets

### Datasets Preparation

#### Synthetic Fog Dataset:

* Dataset: RESIDE; Paper: [RESIDE: A Benchmark for Single Image Dehazing](https://arxiv.org/pdf/1712.04143.pdf)

#### Real-world Fog Datasets:

* Dataset: I-HAZE; Paper: [I-HAZE: a dehazing benchmark with real hazy and haze-free indoor images](https://arxiv.org/pdf/1804.05091.pdf)
* Dataset: O-HAZE; Paper: [O-HAZE: a dehazing benchmark with real hazy and haze-free outdoor images](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Ancuti_O-HAZE_A_Dehazing_CVPR_2018_paper.pdf)
* Dataset: DENSE; Paper: [Dense-Haze: a benchmark for image dehazing with dense-haze and haze-free images](https://arxiv.org/pdf/1904.02904.pdf)
* Dataset: NHHAZE; Paper: [NTIRE 2020 NonHomogeneous Dehazing Challenge (2020)](https://competitions.codalab.org/competitions/22236)
* Dataset: Natural hazy images dataset; Paper: [Dehazing Using Color-Lines](https://www.cs.huji.ac.il/w~raananf/projects/dehaze_cl/results/#stadium)
* Dataset: RTTS dataset: [We have placed this dataset in the link](https://drive.google.com/file/d/1v_yQ69qgAjdD99GgN_MFv3CfIOQcnkex/view?usp=drive_link)

<details>
<summary> FILE STRUCTURE (click to expand) </summary>

```
    DWT-ViT
    |-- README.md
    |-- datasets
        |-- RESIDE
            |-- ITS
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- OTS 
                |-- hazy
                    |-- *.jpg
                |-- clear
                    |-- *.jpg
            |-- SOTS
                |-- indoor
                    |-- hazy
                        |-- *.png
                    |-- clear
                        |-- *.png
                |-- outdoor
                    |-- hazy
                        |-- *.jpg
                    |-- clear
                        |-- *.png
        |--RealWorld
            |-- dense
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- nhhaze
                |-- hazy
                    |-- *.png
                |-- clear
                    |-- *.png
            |-- ihaze
                |-- hazy
                    |-- *.jpg
                |-- clear
                    |-- *.jpg
            |-- ohaze
                |-- hazy
                    |-- *.jpg
                |-- clear
                    |-- *.jpg
```
</details>

### Real world Dust Dataset:

* Dataset: RB-Dust; Paper: [RB-Dust – A Reference-based Dataset for Vision-based Dust Removal](https://arxiv.org/pdf/2306.07244.pdf)(www.agriscapes-dataset.com)

## Quick Run

Before you start your experiment, please follow the [instructions](https://www.cnblogs.com/wanghui-garcia/p/12526298.html) to install pytorch-wavelets.

Trained_models will be available.

For your quick run, we have selected a small number of hazy images in the `./datasets/hazy` folder.

*Put  models in the `./trained_models/` folder.*

*Trained_models are available at [google drive](https://drive.google.com/drive/folders/1whYayNFJsXgKgErAMeBPSsOfishFkpw_?usp=drive_link) 

To test the pre-trained models of Dense-Haze, Ihaze, Ohaze, and NHHaze on your own images, run:
```shell
python test.py --task Task_Name --input_dir path_to_images
```
Here is an example to perform NH-Haze:

```shell
python test.py --task nhhaze --input_dir datasets/hazy
```
*Check out the test results in the `./dwtformer_nhhaze` folder.*

We have put more experimental results  in the `./results/` folder. 

* `overhead.py` is a file where you can test network model's parameters (Params), multiply-accumulate operations (MACs), and inference time.

To train the model on your way, run:

```shell
python main.py 
```
*Go to `./models/data_utils.py` and `./models/option.py` to adjust the hyperparameters as well as the training details.

## Quantitative comparisons

We have uploaded the MATLAB versions of the three non-referenced indicators (**`FADE.m`**, **`NIQE.m`**, **`Entropy.m`**) used in our papaer at `./metrics/` folder.

A higher Entropy score indicates that the image presents more detail, a lower NIQE score indicates better image quality, while a lower FADE score indicates better visibility.

### Comparison of referenced evaluation on real-world dehazing

<center>
    <img src='results/result1.png'>
</center>

### Comparison of non-referenced evaluation on real-world dehazing

<center>
    <img src='results/result2.png'>
</center>

### Comparison on synthetic dataset dehazing

<center>
    <img src='results/result3.png'>
</center>

### Parameters and Runtime Analysis

<center>
    <img src='results/overhead.png'>
</center>

We conduct parameters (# Param), floating-point operations (# FLOPs), and inference time as the main metrics of computational efficiency on RGB image with a resolution of 256 × 256 between WaveletFormerNet and SOTA methods.

## Qualitative Comparisons

<center>
<img src="results\ihaze.png">
<img src="results\ohaze.png">
<img src="results\nhhaze.png">
<img src="results\densehaze.png">
<img src="results\hecheng.png">
</center>

## Comparisons on the on the natural hazy images and RTTS dataset

<center>
    <img src='results/result4.png'>
</center>

<center>
    <img src='results/hazyrtts.png'>
</center>
    
More dehazed results are available in the `./results/` folder.

## Generality Analysis for WaveletFormerNet
<center>
    <img src='results/dust.png'>
</center>

WaveletFormerNet also achieves satisfactory results on the agricultural landscape dust dataset (RB-Dust dataset) compared to SOTA methods in extension experiments, demonstrating the promising robustness and better generalization ability of WaveletFormerNet.

## Application Test

We perform a feature point matching test to evaluate the performance of WaveletFormerNet.

<center>
<img src="results/sift.png">
</center>

## Ablation Experiments

### Quantitative and Visualization comparison of ablation experiments

|<img src="result/ablation1.png">|<img src="result/ablation2.png">|
|:-:|:-:|

## Citation

If our work can bring you help, we will be very excited! If you have other questions, please contact the Authors (email: zhangshengli_win@163.com)

We look forward to your valuable suggestions,your comments will make our work better!




  
 
