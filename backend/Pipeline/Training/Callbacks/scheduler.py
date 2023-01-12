""" Functions for schedulers """

import sys
import numpy as np
import tensorflow as tf

def hardcoded_scheduler_adam(epoch: int, lr: float) -> float:
    if epoch < 8:
        return 0.0001
    
    if epoch < 15:
        return 0.00001

    if epoch < 23:
        return 0.000001

def hardcoded_scheduler_sgd(epoch: int, lr: float) -> float:
    if epoch < 10:
        return 0.005
    
    if epoch < 15:
        return 0.001

    if epoch < 23:
        return 0.0001

    return 0.00001