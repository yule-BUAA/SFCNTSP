# Predicting Temporal Sets with Simplified Fully Connected Networks

The description of "Predicting Temporal Sets with Simplified Fully Connected Networks" at AAAI 2023 is [available here](https://ojs.aaai.org/index.php/AAAI/article/view/25609). 

### Original data:
The original data could be downloaded from [here](https://drive.google.com/file/d/1f2Eexc9vwRYYrrvLzuL4zBnWwWs6EHhI/view?usp=sharing). 
You can download the data and then put the data files in the ```./original_data``` folder.


### To run the code:
  1. run ```./preprocess_data/preprocess_data_{dataset_name}.py``` to preprocess the original data, 
     where ```dataset_name``` could be JingDong, DC, TaoBao and TMS. 
     We also provide the preprocessed datasets at [here](https://drive.google.com/file/d/1ytAM41VwfiOnfAQ-EQw42sf3MLm3TeAJ/view?usp=sharing), 
     which should be put in the ```./dataset``` folder.
     
  2. run ```./train/train_SFCNTSP.py``` to train the model and get the results on different datasets according to the configuration in ```./utils/config.json```.


## Environments:
- [PyTorch 1.8.1](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)


## Hyperparameter settings:
Hyperparameters can be found in ```./utils/config.json``` file, and you can adjust them when training the model on different datasets.

| Hyperparameters  |  JingDong | DC  |  TaoBao | TMS |
| -------    | ------- | -------  | -------  | -------  |
| learning rate  | 0.001  | 0.001  | 0.001  |  0.001   |
| dropout rate | 0.2  | 0.1  | 0.05  |  0.1   |
| embedding channels  | 64  | 64  | 32  |  64   |
| alpha  | 1.0  | 1.0  | 1.0 |  1.0  |
| beta  | 0.1  | 0.1  | 0.1  |  0.1   |


## Citation
Please consider citing our paper when using the codes or datasets.

```
@inproceedings{yu2023predicting,
  title={Predicting Temporal Sets with Simplified Fully Connected Networks},
  author={Yu, Le and Liu, Zihang and Zhu, Tongyu and Sun, Leilei and Du, Bowen and Lv, Weifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={4},
  pages={4835--4844},
  year={2023}
}
```
