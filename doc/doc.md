# INTRODUCTION TO DEEP LEARNING
    - Amine ITJI (p2018984)


# Part 1: Perceptron

## Indicate and explain the size of each tensor of the provided file perceptron_pytorch.py

tensors: w (weight), b (bias), x (input), y (output), t (target labels), grad (gradient).


## Weight tensor

```python
w = torch.empty((data_train.shape[1],label_train.shape[1]),dtype=torch.float)
```
    - data_train.shape[1] = 784 (=28*28)
    - label_train.shape[1] = 10 (output classes: 0-9, example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [784,10]

## Bias tensor 

```python
b = torch.empty((1,label_train.shape[1]),dtype=torch.float)
```
    - 1
    - label_train.shape[1] = 10

    Size => [1,10]


## Input batch tensor 

```python
x = data_train[indices[i:i+batch_size]]
```
    - batch_size = 5
    - nb_data_train = 784

    Size => [5,784]

## Output tensor 

```python
y = torch.mm(x,w)+b
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]

## Target labels tensor 

```python
t = label_train[indices[i:i+batch_size]]
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]

## Gradient tensor 

```python
grad = (t-y)
```

    - batch_size = 5
    - output classes size = 10 (example: [0,0,0,0,0,1,0,0,0,0] for digit 5)

    Size => [5,10]


# Part 2: Shallow network
# Part 3: Deep network
# Part 4: CNN
