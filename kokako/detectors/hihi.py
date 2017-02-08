"""A detector for Hihi. This is just a specific version of
TFGraphDetector with some hardcoded parameters to point to a pre-trained
Hihi detector."""
import os

from kokako.detectors.tfgraph import TFGraphDetector


class HihiCNN(TFGraphDetector):
    code = 'hihi'
    description = 'Loads a trained convolutional neural net for Hihi detection'
    version = '0.0.1'

    def __init__(self, detector_path=None):
        """Loads a hihi detector.

        Args:
            detector_path (Optional[str]): path to the hihi detector. If not
                specified, looks for a file ./models/hihi.pb relative to the
                directory of this file.

        Raises:
            NotFoundError: if we can't find the file.
        """
        if not detector_path:
            detector_path = os.path.join(
                os.path.dirname(__file__), 'model', 'hihi.pb')

        super(HihiDetector, self).__init__(detector_path)

    def score(self, audio):
        """score some audio using the tensorflow graph"""
        pass
