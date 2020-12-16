## BDANN
**IJCNN 2020 BDANN: BERT-Based Domain Adaptation Neural Network for Multi-Modal Fake News Detection**

## Environment
Pytorch 1.4.0 
Python 3.8

## Dataset

**Twitter**: “verifying multimedia use” task by MediaEval Benchmarking Initiative for Multimedia Evaluation 

**Weibo**: download from  https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing

## Training
For Twitter dataset:
```
python BDANN_twitter.py
```
For Weibo dataset:
```
python BDANN_weibo.py
```

## Removed Posts from Weibo
https://github.com/xiaolan98/RemovedPostsFromWeibo

## Citation
```
@inproceedings{zhang2020bdann,
  title={BDANN: BERT-Based Domain Adaptation Neural Network for Multi-Modal Fake News Detection},
  author={Zhang, Tong and Wang, Di and Chen, Huanhuan and Zeng, Zhiwei and Guo, Wei and Miao, Chunyan and Cui, Lizhen},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2020},
  organization={IEEE}
}
```