# -*- coding: UTF-8 -*-
# import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np


from data_processing.fastq_to_npz import fastq_to_npz
from data_processing import shuffle_npz

if __name__ == '__main__':
    fastq_to_npz('../data','../output_npz')
    shuffle_npz.main(os.path.join('..','output_npz'),os.path.join('..','output_npz'),64)

if __name__ == '__main__':
    from data_processing import npz_to_model_yaml
    npz_to_model_yaml.npz_to_yaml(os.path.join('..','output_npz'),os.path.join('..','model','model_config.yaml'))

