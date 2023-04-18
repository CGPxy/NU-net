#  NU-net: Rethinking the unpretentious U-net for medical ultrasound image segmentation

## 1. Datasets：
### Breast ultrasound dataset:
(1)[BUSI:](https://doi.org/10.1016/j.dib.2019.104863) W. Al-Dhabyani., Dataset of breast ultrasound images, Data Br. 28 (2020) 104863.  
(2)[Dataset B:](https://doi.org/10.1016/j.artmed.2020.101880) M. H. Yap et al., Breast ultrasound region of interest detection and lesion localisation, Artif. Intell. Med., vol. 107, no. August 2019, p. 101880, 2020.  
(3)[STU:](https://doi.org/10.1371/journal.pone.0221535) Z. Zhuang, N. Li, A. N. Joseph Raj, V. G. V Mahesh, and S. Qiu, “An RDAU-NET model for lesion segmentation in breast ultrasound images,” PLoS One, vol. 14, no. 8, p. e0221535, 2019.  

## 2. Development environment:
The development environment is TensorFlow 2.6.0, Python 3.6 and two NVIDIA RTX 3090 GPU. More environment variables are requested in Requirements.
	
## 3. Network hyperparameters:
The epoch size and batch size are set to 50 and 12, respectively. We utilize the Adam optimizer to train our network and its hyperparameters are set to the default values (the learning rate is 0.001, the momentum is 0.99, the epsilon is 1e-07, and the weight decay is None).

## 4. Reproduce
#### Step 1: Perform data augmentation
     python Data_augement.py   --Namepath='Name of your original image'   --Imagepath='path of your original train image'   --Labelpath= 'path of your original train label'    --Saveimagepath='path to save your train imag'   --Savelabelpath= 'path to save your train label' 

#### Step 2: Network training
     python Train.py   --filepath='path of your train data'   --save_dir='path to output model'
     
#### Step 3: Network test
     python Predict.py  --model='path of your trained model' --Imagepath='path of your test image'   --Saveimagepath='path to save your predict mask'     
     
#### Step 4: Evaluation of quantitative indicators
     python compare_function.py     --Imagepath='path of your predic mask'   --Labelpath= 'path of your ground-truth mask'  --Savepath= 'path to save your results' 
     
## 5. Experimental results:

![1662684955131](https://user-images.githubusercontent.com/52651150/189250438-bd4178e4-b4cd-4909-b09c-51d4338dc011.png)

