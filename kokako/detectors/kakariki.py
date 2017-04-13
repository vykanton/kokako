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

    def __init__(self, detector_path=None, prediction_block_size=2):
        """Loads the Kakariki detector.

        Args:
            detector_path (Optional[str]): path to the saved model.
                Looks in `./models/kakariki.pb` relative to this file if
                left unspecified.

            prediction_block_size (Optional[int]): how consecutive windows of
                audio we average over when getting the predictions.

        Raise:
            NotFoundError: if the saved graph is not where we expect.
        """
        if not detector_path:
            detector_path = os.path.join(
                os.path.dirname(__file__), 'models', 'kakariki.pb')
        super(KakarikiRNN, self).__init__(detector_path, num_cores=None,
                                          trace=False, debug=False)

        self._audio_chunk_size = 76800
        self._audio_framerate = 24000
        self._audio_hop_size = self._audio_chunk_size // 1
        self._prediction_block = prediction_block_size

    def score(self, audio):
        """score the audio"""
        audio_data = audio.audio
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
