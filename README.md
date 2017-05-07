# Introduction


# Dependencies
- numpy v1.12
- sklearn

# Dataset
The MNIST database of handwritten digits  
Reference : http://yann.lecun.com/exdb/mnist/

# Results


# Usage
```
usage: main.py [-h] [--train_X TRAIN_X] [--train_T TRAIN_T] [--test_X TEST_X]
               [--test_T TEST_T] [--load LOAD] [--save SAVE] [--pre {pca,lda}]
               [--deg DEG] [--task {validate,train,plot}] [--model {rf}]
               [--param_rf PARAM_RF] [--verbose VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  --train_X TRAIN_X     training data X
  --train_T TRAIN_T     training data T
  --test_X TEST_X       testing data X
  --test_T TEST_T       testing data T
  --load LOAD           model load from
  --save SAVE           model save to
  --pre {pca,lda}       preprocess type
  --deg DEG             degree for preprocess
  --task {validate,train,plot}
                        task type
  --model {rf}          model type
  --param_rf PARAM_RF   parameter for rf
  --verbose VERBOSE     log on/off

```

## To train the model for example
```
./train.sh "{list of number of decision tree}" "{list of minimum number of samples per leaf nodes}" "list of fraction of samples"

e.g.
./train.sh "100 1000" "10 5" "0.25 0.5"

```

## To evaluate the model for example

