# CN-sudoku
a class project for EI339 Artificial Intelligence, SJTU 

## Description

This project implements the identification and solving of sudoku puzzles in an image, with digits written in Chinese or Arabic characters.

## Usage

### Creating Anaconda environment

This command creates an Anaconda environment called "EI339".

```shell
conda env create -f EI339.yml
```

### Training the network

*Notes: due to the copyrights policy I may not publicate the EI339-CN dataset on github. Please contact me if you have further interest in deploying this repo.*

To train the network:

```shell
python train_digit_classifier.py --epoch [num_epoch] --net [name_net] --version [number] --debug [true for n > 0] 
```

For example:

```shell
python train_digit_classifier.py --epoch 30 --net lenet --version 075 --debug 1 
```

### Solving the puzzles in the image

```shell
python solve_sudoku_puzzle.py --model [path_model] --image [path_image] --debug [true for n > 0]
```

For example:

```shell
python solve_sudoku_puzzle.py --model lenet_epoch=30_ver=072.h5 --image test_problems/1-3.jpg --debug 1
```
