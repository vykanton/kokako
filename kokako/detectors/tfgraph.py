"""A base for detectors implemented using a saved tensorflow graph."""
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tfdbg


# TODO: this is hacky and fragile
def _np_rfft(data):
    """Does rfft with numpy"""

    def _np_spectrum(data):
        return np.fft.rfft(data).astype(np.complex64)

    spectrum = tf.py_func(_np_spectrum, [data], [tf.complex64], stateful=False)
    # figure out what the output shape should be so we can specify it directly
    # (otherwise tf can't deal with py_func)
    out_shape = data.get_shape().as_list()
    out_shape[-1] = out_shape[-1] // 2 + 1
    return tf.reshape(spectrum[0], out_shape)


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
            chunk = audio[i:i+chunk_size, ...]
            if chunk.shape[0] == chunk_size:
                yield audio[i:i+chunk_size, ...]
            # if not, we should just fall off the end

    def __init__(self, graphdef_path, input_name=None, output_name=None,
                 num_cores=None, trace=False, debug=False):
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
            num_cores (Optional[int]): how many cores to instruct tensorflow to
                use. Default (None) will let tensorflow behave as it sees fit,
                any other number will be passed in as the max for both intra-op
                parallelism _and_ inter-op parallelism.
            trace (Optional[bool]): whether to generate traces to profile
                per-op performance.
            debug (Optional[bool]): whether to drop into the tensorflow debug
                console on tf.Session.run() calls. Default is not to.
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
                self._graphdef, return_elements=return_elements)

        self._run_metadata = tf.RunMetadata()

        if trace:
            self._run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
        else:
            self._run_options = None
        self._trace = trace

        if num_cores:
            conf = tf.ConfigProto(
                intra_op_parallelism_threads=num_cores,
                inter_op_parallelism_threads=num_cores)
        else:
            conf = None
        self._session = tf.Session(graph=self._graph,
                                   config=conf)
        if debug:
            self._session = tfdbg.LocalCLIDebugWrapperSession(self._session)
            self._session.add_tensor_filter('has_inf_or_nan',
                                            tfdbg.has_inf_or_nan)

        # TODO: hacky and fragile
        # at this stage we need to use numpy for ffts on the cpu, so we have to
        # wrap it up. For this to work with a saved graph, we have to register
        # the wrapper with tensorflows python function registry, which does not
        # happen unless we make an op
        _ = _np_rfft(tf.random_normal([123]))

    # this is slightly sub-optimal, but we have to do our best to clean up
    def __del__(self):
        if self._session:
            self._session.close()

    @property
    def output_shape(self):
        return self._output_node.get_shape()

    @property
    def input_shape(self):
        return self._input_node.get_shape()

    def _run_graph(self, input_value):
        """Run the graph once on a given input.

        Args:
            input_value (ndarray): the data to feed in.

        Returns:
            ndarray: the result returned by running the output node.
        """
        result = self._session.run(self._output_node,
                                   {self._input_node: input_value},
                                   options=self._run_options,
                                   run_metadata=self._run_metadata)
        if self._trace:
            if not tf.gfile.Exists('timeline.json'):
                with open('timeline.json', 'w') as fp:
                    tl = timeline.Timeline(self._run_metadata.step_stats)
                    fp.write(tl.generate_chrome_trace_format())
            # turn off tracing now, otherwise the remaining runs are wildly
            # slow
            self._run_options = None
            self._run_metadata = None
        return result

    def collect_graph_outputs(self, audio, chunk_size, hop_size=None):
        """Collects the outputs of the graph over chunks of audio. Expects the
        audio to be in whatever format it is that the graph expects.

        Args:
            audio (ndarray): numpy array, we assume the first dimension is
                time, but that is the only assumption.
            chunk_size (int): the size of the chunks we break `audio` into.
            hop_size (Optional[int]): the number of samples we skip before
                breaking off a new chunk. This can be used to extract
                overlapping patches. If not specified, is set to `chunk_size`,
                resulting in non-overlapping paches.

        Returns:
            list: the result of the graph at each location.
        """
        results = [self._run_graph(chunk)
                   for chunk in self.chunk_audio(audio, chunk_size, hop_size)]
        return results

    def _max_block_average(self, outputs, block_size):
        """Look at the average of block_size consecutive chunks in a row and
        return the maximum. Note that block_size=1 will reduce to just taking
        the maximum. If block size is 0 or None, we just average the whole
        thing."""
        if not block_size:
            return np.mean(outputs)
        # go through the chunks one at a time and collect the average
        best = 0
        block = deque()
        # this is not especially efficient/clever
        for chunk in outputs:
            block.append(chunk)
            if len(block) == block_size:
                block_val = np.mean(block)
                if block_val > best:
                    best = block_val
                block.popleft()
        # may be a smaller chunk at the end
        if len(block):
            final_val = np.mean(block)
            if final_val > best:
                best = final_val
        return best

    def average_graph_outputs(self, audio, chunk_size, hop_size=None,
                              block_size=2):
        """Aggregates the outputs of the graph over chunks of audio. Expects the
        audio to be in whateber format it is that the graph expects. Ultimately
        returns the maximum value across the specified multiple of chunks.

        Args:
            audio (ndarray): numpy array, we assume the first dimension is
                time, but that is the only assumption.
            chunk_size (int): the size of the chunks we break `audio` into.
            hop_size (Optional[int]): the number of samples we skip before
                breaking off a new chunk. This can be used to extract
                overlapping patches. If not specified, is set to `chunk_size`,
                resulting in non-overlapping paches.
            block_size (Optional[int]): how many chunks we average over to get
                single prediction.

        Returns:
            scalar: the average
        """
        chunk_outputs = self.collect_graph_outputs(audio, chunk_size, hop_size)
        chunk_outputs = np.array(chunk_outputs)
        return self._max_block_average(chunk_outputs, block_size)
