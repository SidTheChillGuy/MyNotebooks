# MyNotebooks
This repo contains all the machine learning notebooks which I have created, to learn, apply, and experiment with current ML technologies. The results are repeatable. 

## Contents
### [Breast Cancer with Rapids](https://github.com/SidTheChillGuy/MYNotebooks/blob/main/RAPIDS_implementation.ipynb)
Implementation of Rapids, an alternative to scikit-learn which enables NVIDIA CUDA gpu acceleration and faster inferencing. The models trained in the notebook performed better and faster than scikit-learn counterparts.


### [Employee Attrition](https://github.com/SidTheChillGuy/MyNotebooks/blob/main/ML_attrition_gpu_accel.ipynb)
A scikit-learn + Rapids implementation.

### [Brain Cancer detection using CNN models](https://github.com/SidTheChillGuy/MyNotebooks/blob/main/Brain_cancer-multi-model-finetuning-0-98-acc.ipynb)
Brain Cancer detection using images. The Functional Model is made up of three pretrained ML models, utilizing transfer learning. Unlike ensemble models where 3 independent models predict outputs which are then processed, this implementation combines the N models in a single meta model. 
- Improved Memory usage in localised testing
- Faster training time
- Quick convergence
- Very high accuracy
Can still be improved

### [Lung Disease Detection using audio samples](https://github.com/SidTheChillGuy/MyNotebooks/blob/main/Lung_Diseases_Code_audio.ipynb)
Detection of lungs diseases using the audio samples and recurrent networks

### [BirdLens](https://github.com/SidTheChillGuy/BirdLens)
CNN model finetuning along with ensembling. High accuracy

### [Plant Leaf Disease Detection](https://github.com/SidTheChillGuy/PlantLeafDiseaseDetection)
A CNN model which is trained to detect diseases of plants based on their leaf images.
