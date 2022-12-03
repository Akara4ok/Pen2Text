""" Described different types """
from collections import namedtuple

Sample = namedtuple('Sample', 'text, file_path')
Batch = namedtuple('Batch', 'imgs, texts')