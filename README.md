## Cross-Correlation Attack: Adversarial Attack for Visual Object Tracking
The code will be all open source in the future and is currently being sorted out. . .
## Abstract

This code are main based on ([CSA](https://github.com/MasterBin-IIAU/CSA/)), ([Pysot](https://github.com/STVIR/pysot)), and ([Pytracking](https://github.com/visionml/pytracking)).

## Installation
#### Clone the repository
```
git clone https://github.com/Rongmiq/CCA.git
cd <Project_name>
```
#### Create Environment
```
conda create -n CCA python=3.7
source activate CCA
conda install pytorch=1.0.0 torchvision cuda100 -c pytorch
pip install -r requirements.txt
conda install pillow=6.1
```

#### Prepare the training set (optional), from CSA.
1. Download the training set of GOT-10K.   
2. Then change 'got10k_path' and 'save_path' in Unified_GOT10K_process.py to yours.    
3. Finally, run the following script.   
(it takes a long time. After running it, you can do the next steps :)   
```
python Unified_GOT10K_process.py
```
#### Download pretrained models
1. SiamRPN++([Model_Zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md))   
Download **siamrpn_r50_l234_dwxcorr** and **siamrpn_r50_l234_dwxcorr_otb**  and rename them to **model.pth**.  
Put them under pysot/experiments/<MODEL_NAME>  
We use **siamrpn_r50_l234_dwxcorr_otb** test OTB2015 and **siamrpn_r50_l234_dwxcorr** test VOT2018, UAV123, and LaSOT.  

2. SiamCAR([Model_Zoo](https://github.com/ohhhyeahhh/SiamCAR))     
Download **general_modelr** and **LaSOT_model** and rename them to **model.pth** and **LaSOT_model.pth**, respectively.  
Put them under pysot/experiments/siamcar_r50.   
We use **LaSOT_model.pth** test LaSOT and **model.pth** test OTB2015, VOT2018, and UAV123.  

3. SiamBAN([Model_Zoo](https://github.com/hqucv/siamban/blob/master/MODEL_ZOO.md))  
Download **siamban_r50_l234** and **siamban_r50_l234_otb** and rename them to **model.pth** and **model_vot.pth**, respectively.   
Put them under pysot/experiments/siamban_r50.   
We use **model.pth** test OTB2015 and **model_vot.pth** test VOT2018, UAV123, and LaSOT.  

4. SiamGAT([Model Zoo](https://github.com/ohhhyeahhh/SiamGAT))  
Download models trained for OTB2015 and LaSOT and rename them to **model.pth**.  
Put them under pysot/experiments/Siamgat_googlenet and pysot/experiments/Siamgat_googlenet_lasot, respectively.  


6. Perturbation Generators  
Download checkpoints you need, then put them under checkpoints/<MODEL_NAME>/   
([Google Drive](https://drive.google.com/open?id=117GuYBQpj8Sq4yUNj7MRdyNciTCkpzXL),  
[Baidu](https://pan.baidu.com/s/1rlpzCWczWf6Hw5YnnQThOw)[Extraction code: 98rb])  


#### Set some paths
**Step1**: Add pix2pix and pysot to environment variables   
```
sudo gedit ~/.bashrc
# add the following two lines to the end
export PYTHONPATH=<CSA_PATH>:$PYTHONPATH
export PYTHONPATH=<CSA_PATH>/pysot:$PYTHONPATH
export PYTHONPATH=<CSA_PATH>/pix2pix:$PYTHONPATH
# close the file
source ~/.bashrc
```
**step2**: Set another paths
1. Gather testing datasets     
create a folder outside the project folder as <DATASET_ROOT>  
then put soft links for OTB100, VOT2018 and LaSOT into it   
2. Set 'project_path_' and 'dataset_root_'
Open common_path.py, go to the end     
project_path_ = <CSA_PATH>  
dataset_root_ = <DATASET_ROOT>
train_set_path_ = <TRAIN_SET_PATH>
## Training (Optional)
**Train a generator you need.**
**Option1: Attacking template branch**  

```
python train.py
```

Train a generator for attacking search regions (**Only Cooling**)  
```
python train0.py # See visualization in http://localhost:8096/
```
**Option2: Change Settings**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If you want to train other models (like the generator for attacking the template), 
you can change the **lines 23 and 24** in pix2pix/options/**base_option0.py** (or base_option1.py). 
In specific, modify the default values to **'G_template_L2_500'** (or 'G_template_L2_500_regress'). 
Then run ```python train0.py``` or ```python train1.py```  
**Option3: Train Your Own Models**  
**Step1**: Create a new python file under pix2pix/models.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can copy a file that belongs to this folder, then develop based on it. 
Note that the class name must match the filename.   
**Step2**: Change default values and train (Do as instructions in Option2)
## Testing
open ```common_path.py```, choose the dataset and siamese model to use.  
open ```GAN_utils_xx.py```, choose the generator model to use.  
```cd pysot/tools```  
run experiments about attcking **search regions**  
```
python run_search_adv0.py # or run_search_adv1.py
```
run experiments about attacking **the template**  
```
python run_template_adv0.py # or run_template_adv1.py
```
run experiments about attacking **both search regions and the template**
```
python run_template_search_adv0.py # or run_template_search_adv1.py
```
