"""Detector for Kakariki using pre-trained tensorflow model."""
import os
import numpy as np

from kokako.detectors.tfgraph import TFGraphUser
from kokako.score import Detector

# for the GRUBlockCell op
import tensorflow.contrib.rnn


class KakarikiRNN(Detector, TFGraphUser):
    code = 'kakariki'
    description = 'Load a trained neural net for Kakariki detection'
    version = '1.0.0'
