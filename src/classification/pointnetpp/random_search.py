# training python script is in pointnetpp/train_semseg.py
# we need to find hyperparameters for the model
# random search for hyperparameters

"""
AVALABLE HYPERPARAMETERS:
--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
--epoch', default=64, type=int, help='Epoch to run [default: 32]')
--learning_rate', default=1e-3, type=float, help='Initial learning rate [default: 0.001]')
--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
--log_dir', type=str, default=None, help='Log path [default: None]')
--decay_rate', type=float, default=0.001, help='weight decay [default: 1e-4]')
--npoint', type=int, default=4096, help='Point Number [default: 4096]')
--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
"""

# The ones we will be changing are:
# batch_size (between 1 and 32)
# learning_rate (between 1e-5 and 1e-1), choose on log scale
# optimizer (Adam or SGD)
# decay_rate (between 1e-5 and 1e-1), choose on log scale
# npoint (between 256 and 16384), choose on log scale
# step_size (between 1 and 10)
# lr_decay (between 0.5 and 0.9)
from random import randint, uniform
import os
import tqdm
import subprocess


def random_hyperparameters():
    batch_size = 1  # randint(1, 32)
    learning_rate = 10 ** uniform(-5, -1)
    decay_rate = 10 ** uniform(-5, -1)
    npoint = 2 ** randint(8, 14)
    step_size = randint(1, 10)
    lr_decay = uniform(0.5, 0.9)
    return batch_size, learning_rate, decay_rate, npoint, step_size, lr_decay


log_dir = "random_search"
num_trials = 2000
results = []
errors = []
# make a file to store the results: "random_search_results.txt"

with open("random_search_results.txt", "w") as f:
    for i in tqdm.tqdm(range(num_trials)):
        batch_size, learning_rate,  decay_rate, npoint, step_size, lr_decay = random_hyperparameters()
        # print("batch_size: {}, learning_rate: {}, optimizer: {}, decay_rate: {}, npoint: {}, step_size: {}, lr_decay: {}".format(batch_size, learning_rate, optimizer, decay_rate, npoint, step_size, lr_decay))
        wq = "python train_semseg.py --batch_size {} --learning_rate {} --decay_rate {} --npoint {} --step_size {} --lr_decay {} --log_dir {}".format(
            batch_size, learning_rate, decay_rate, npoint, step_size, lr_decay, log_dir)
        # run and wait for process to finish
        result = subprocess.run(wq, shell=True, capture_output=True)
        results.append(result.stdout)
        errors.append(result.stderr)
        f.write(str(result.stdout))
        f.write(str(result.stderr))
