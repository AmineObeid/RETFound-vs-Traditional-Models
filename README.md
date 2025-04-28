# RETFound-vs-Traditional-Models

This repository describes how we did the comparison of traditional models to fine-tuned RETFound model for a multi-classification task.

The dataset could be found on: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

To finetune RETFound please check the original author's repo on: https://github.com/rmaphoh/RETFound_MAE

In order to be able to run the code properly, please make sure to divide the data in the format of 
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
