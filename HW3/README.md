# Run part 1:
## Part 1.1 
```
cd /HW3/code/part1/
python part1.py --net convnet  [--use_GPU True/False] 
```

## Part 1.2
```
cd /HW3/code/part1/
python part1.py --net mobilenet  [--use_GPU True/False] 
```


## Part 1.3
```
cd /HW3/code/part1/
python part1.py --net resnet  [--use_GPU True/False] 
```

# Part 2 and Part 3 use the same code 

# Run part 2: 
## Part 2.1 
```
cd /HW3/code/part2/
python part2.py --loss_type cls  [--use_GPU ] 
```
## Part 2.2 
```
cd /HW3/code/part2/
python part2.py --loss_type total  [--use_GPU ] 
```


# Run part 3:

## Part 3.1 
```
cd /HW3/code/part3/
python part3.py --loss_type total --visual True  [--use_GPU ] 
```

## Part 3.2
### A simple way to train the network: use all losses at once.  
```
python part3.py [--use_GPU true/false] --resume false --loss_type simple --max_epoches 20 --lr 1e-3
```

### Alternative training: train with proposal loss first 
```
python part3.py --use_GPU true --resume false --loss_type proposal --max_epoches 20 --lr 1e-3 
```
### Train with object classification loss 
```
python part3.py --use_GPU false --resume true --loss_type object --max_epoches 20 --lr 1e-3

```
### Fine-tune the network 
```
python part3.py --use_GPU false --resume true --loss_type total --max_epoches 20 --lr 1e-4
```

## Part 3.3
### Mobile base net: 
```
python part3.py [--use_GPU true/false] --resume false --loss_type simple --max_epoches 20 --lr 1e-3 --net fasterrcnnmobilenet
```

### Residual base net: 
```
python part3.py [--use_GPU true/false] --resume false --loss_type simple --max_epoches 20 --lr 1e-3 --net fasterrcnnresnet
```
