# RETFound-vs-Traditional-Models

This repository shows the code we used in order to train and evaluate 3 traditional computer vision models so we can compare their performances to that of fine-tuned RETFound.

The dataset we used for this project could be found on: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

To finetune RETFound please check the original author's repo on: https://github.com/rmaphoh/RETFound_MAE

In order to be able to run the code properly, please make sure to divide the data in the following format  
```
data/
  train/ 
    class A
    class B
    ...
  val/ 
    class A
    class B
    ...
  test/
    class A
    class B
    ...
```
