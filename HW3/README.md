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
python part2.py --loss_type total --visual True  [--use_GPU ] 
```

## Part 3.2

### Train with proposal loss first 
```
python part3.py --use_GPU true --resume true --loss_type proposal --max_epoches 20 --lr 1e-3 
```
### Train with object classification loss 
```
python part3.py --use_GPU false --resume true --loss_type object --max_epoches 40 --lr 1e-3

```
### Fine-tune the network 
```
python part3.py --use_GPU false --resume true --loss_type total --max_epoches 20 --lr 1e-4

```
### Alternative way to train the network: use all losses at once.  
```
python part3.py --use_GPU true --resume false --loss_type simple --max_epoches 20 --lr 1e-3
```

