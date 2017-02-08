"""A base for detectors implemented using a saved tensorflow graph."""
import numpy as np
import tensorflow as tf


class TFGraphUser(object):
    """Base class for objects which load and run a tensorflow graph.
    The standard implementation does not run any initialisers, we assume the
    graph has been frozen into a single GraphDef with all variables folded
    into constants.
    """

    @staticmethod
    def chunk_audio(audio, chunk_size, hop_size=None):
        """Generator which yields chunks of samples. Expects audio to be a
        numpy array, not a kokako.score.Audio. We also expect the first axis
        to be time. WILL skip any data at the end if the audio doesn't break
        evenly into chunks.

        Args:
            audio (ndarray): the numpy array full of audio data.
            chunk_size (int): what size chunks to divide the audio into
                (in samples).
            hop_size (Optional[int]): how far to step along (in samples) before
                grabbing a new chunk. This controls how much the chunks
                overlap. If not specified, it is set to `chunk_size` to
                retrieve chunks with no overlap.
        """
        if not hop_size:
            hop_size = chunk_size
        num_samples = audio.shape[0]
        for i in range(0, num_samples, hop_size):
            yield audio[i:i+chunk_size, ...]

    def __init__(self, graphdef_path, input_name=None, output_name=None):
        """Initialise the basic graph handling. Tries to be as
        self-contained as possible (ie. avoid a lot of tensorflows default
        global data structures). What this does is the following:
        - sets up an empty graph
        - attempts to import the file at `graphdef_path` as if it is a
          serialised (binary) GraphDef protobuf.
        - starts a session using the graph.

        Args:
            graphdef_path (str): path to a serialised (binary only right now)
                GraphDef protobuf.
            input_name (Optional[str]): name of the input node. If not
                specified we try to use "input:0"
            output_name (Optional[str]): name of the output node. If not
                specified we try to use "output:0"
        """
        self._input_node_name = input_name or 'input:0'
        self._output_node_name = output_name or 'output:0'
        self._graphdef_path = graphdef_path
        # see if we can load the graphdef proto
        with open(self._graphdef_path, 'rb') as fp:
            self._graphdef = tf.GraphDef()
            self._graphdef.ParseFromString(fp.read())

        self._graph = tf.Graph()
        with self._graph.as_default():
            return_elements = [self._input_node_name, self._output_node_name]
            self._input_node, self._output_node = tf.import_graph_def(
                graph_def, return_elements=return_elements)

        self.session = tf.Session(graph=self._graph)

    def average_graph_outputs(self, audio, chunk_size, hop_size=None):
        """Averages the outputs of the graph over chunks of audio. Expects the
        audio to be in whateber format it is that the graph expects.

        Args:
            audio (ndarray): numpy array, we assume the first dimension is
                time, but that is the only assumption.
            chunk_size (int): the size of the chunks we break `audio` into.
            hop_size (Optional[int]): the number of samples we skip before
                breaking off a new chunk. This can be used to extract
                overlapping patches. If not specified, is set to `chunk_size`,
                resulting in non-overlapping paches.

        Returns:
            scalar: the average
        """
        pass
