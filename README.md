# Robust Labeling and Invariance Modeling for Unsupervised Cross-Resolution Person Re-Identification

## Environment
Python >= 3.6

PyTorch >= 1.1

## Installation
```python
git clone https://github.com/zqpang/RLIM
```

```python
cd RLIM
```

## Train
```python
python train.py --dataset mlr_cuhk --iters 400 --cuda 0,1 --batch-size 32 --cr
```
"--cr" represents using cluster refinement.
