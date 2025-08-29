# CGCNet
This is the official code repository for "CGCNet: Road Extraction from Remote Sensing Image  with Compact Global Context-aware"（TGRS 2025）. 
SW-XJU road dataset link: 链接: https://pan.baidu.com/s/10d_NkyyL3yC3ObDw9ov11A. Please email liupengxju@163.com to get the password. Paper url: https://ieeexplore.ieee.org/abstract/document/11097289.

## Abstract
In recent years, methods leveraging global context modeling for continuous road extraction from remote sensing images have garnered significant attention. These methods can effectively address the issues of road discontinuity and missed detection caused by complex background interference, but they generally suffer from high computational complexity. Therefore, we propose CGCNet, a simple yet effective road extraction network based on global context modeling. CGCNet adopts an encoder–decoder architecture with a compact global context-aware block (CGCB) embedded in the center part to capture long-range dependencies among road segments. This block effectively enhances the model’s global modeling capability and significantly reduces computational complexity using a compact representation of the embedded Gaussian nonlocal block (NLB). Furthermore, we introduce SW-XJU, the first remote sensing road dataset constructed explicitly for the unique landscapes of Western China. Extensive experiments show that CGCNet achieves a superior balance between efficiency and accuracy compared with the state-of-the-art methods, and the extracted roads exhibit stronger connectivity. The source code and dataset are available at https://github.com/LPeng625/CGCNet.

## 1.Training
DeepGlobe
`python main_CGCNet_Road.py`

Massachusetts
`python main_CGCNet_Mas.py`

## 2.Evaluation
DeepGlobe
`python test_CGCNet_Road.py`

Massachusetts
`python test_CGCNet_Mas.py`

## 3.Acknowledgments
We thank the authors of [RCFSNet](https://github.com/CVer-Yang/RCFSNet) for open-source code.

