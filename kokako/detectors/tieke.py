"""Detector for Tieke using pre-trained tensorflow model."""
import os
import numpy as np

from kokako.detectors.tfgraph import TFGraphUser
from kokako.score import Detector

# make sure extra ops that we need for the RNN get imported
import tensorflow.contrib.rnn


class TiekeRNN(Detector, TFGraphUser):
    code = 'tieke'
    description = 'Load a trained neural net for Tieke detection.'
    version = '0.0.1'

    def __init__(self, detector_path=None, prediction_block_size=0):
        """Loads a Tieke detector.

        Args:
            detector_path (Optional[str]): path to the Tieke detector. If left
                unspecified looks for a file `./models/tieke.pb` relative to
                where this file is.

            prediction_block_size (Optional[int]): how many consecutive
                windows of audio we average over. For the Tieke model, this can
                probably be relatively small, because the window size has
                to be quite large.

        Raises:
            NotFoundError: if we can't find the file.
        """
        if not detector_path:
            detector_path = os.path.join(
                os.path.dirname(__file__), 'models', 'tieke.pb')

        super(TiekeRNN, self).__init__(detector_path, num_cores=None,
                                       trace=True)

        self._audio_chunk_size = 76800  # aka 3.2s at 24kHz
        self._audio_framerate = 24000  # audio sample rate we trained on
        self._audio_hop_size = self._audio_chunk_size // 2  # amount of overlap
        self._prediction_block = prediction_block_size

    def score(self, audio):
        """score some audio"""
        audio_data = audio.audio
        # scale it between -1, 1 (it comes in as signed 16 bit ints)
        audio_data = audio_data.astype(np.float32) / (2**15)

        if audio.framerate != self._audio_framerate:
            raise ValueError(
                'Audio framerate is wrong (expected {}, found {})'.format(
                    self._audio_framerate, audio.framerate))

        result = self.average_graph_outputs(audio_data,
                                            self._audio_chunk_size,
                                            self._audio_hop_size,
                                            self._prediction_block)
        return result
