# CART for few-shot medical image segmentation

![image](https://github.com/zmcheng9/CART/blob/main/overview.png)

### Abstract
Although image segmentation has been one of the critical tasks in computer-assisted medical research, the dense annotation of medical images requires strict accuracy and consumes significant human, material, and financial resources. As a result, learning a high-performing model from limited medical image data becomes an urgent requirement and a challenging problem. Existing Few-Shot Medical Image Segmentation (FSMIS) approaches tend to segment query images using a single prototype learned only from the support set, which often leads to a serious bias problem. To address this challenging issue, we propose a method called Cascaded Altering Refinement Transformer (CART) to iteratively calibrate the prototypes with both support images and query images. This method focuses on capturing the commonality between foregrounds in support and query images using the Alterating Refinement Transformer (ART) module, which include two Multi-head Cross Attention (MCA) modules, and the cascaded ART modules are employed to refine the class prototypes, resulting in that more representative prototypes are learned. This process ultimately leads to more accurate segmentation predictions for the query images. Besides, to preserve more valid information in previous iterations and achieve better performance, we propose a new inference method that accumulates the predicted segmentation map in each iteration by applying the Rounding-Up strategy. Extensive experiments on three publicly available medical image datasets are conducted, the results demonstrate that the proposed model can outperform all the state-of-the-art methods, and detailed analysis also validates the reasonableness of our design.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 
