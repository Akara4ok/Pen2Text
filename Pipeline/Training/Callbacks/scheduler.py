import sys
import numpy as np
import tensorflow as tf

def scheduler(epoch, lr):
    if epoch < 10:
        return 0.0001
    
    if epoch < 15:
        return 0.00001

    if epoch < 20:
        return 0.000001