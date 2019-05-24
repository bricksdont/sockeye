# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for training
"""
import logging
import os
import pickle
import random
import shutil
import time
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np
from math import sqrt

from . import checkpoint_decoder
from . import constants as C
from . import data_io
from . import loss
from . import lr_scheduler
from . import model
from . import utils
from . import vocab
from .optimizers import BatchState, CheckpointState, SockeyeOptimizer, OptimizerConfig
import multiprocessing
import sockeye.multiprocessing_utils as mp_utils

logger = logging.getLogger(__name__)


class ReconstructionModel(model.SockeyeModel):
    """
    TrainingModel is a SockeyeModel that fully unrolls over source and target sequences.

    :param config: Configuration object holding details about the model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param output_dir: Directory where this model is stored.
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param bucketing: If True bucketing will be used, if False the computation graph will always be
            unrolled to the full length.
    :param gradient_compression_params: Optional dictionary of gradient compression parameters.
    :param fixed_param_names: Optional list of params to fix during training (i.e. their values will not be trained).
    """

    def __init__(self,
                 config: model.ModelConfig,
                 context: List[mx.context.Context],
                 output_dir: str,
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 default_bucket_key: Tuple[int, int],
                 bucketing: bool,
                 gradient_compression_params: Optional[Dict[str, Any]] = None,
                 fixed_param_names: Optional[List[str]] = None,
                 r_lambda: Optional[int] = 1) -> None:
        super().__init__(config)
        self.context = context
        self.output_dir = output_dir
        self.fixed_param_names = fixed_param_names
        self._bucketing = bucketing
        self._gradient_compression_params = gradient_compression_params
        self._r_lambda = r_lambda
        self._initialize(provide_data, provide_label, default_bucket_key)
        self._monitor = None  # type: Optional[mx.monitor.Monitor]

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
                                    axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source_words)
        source_labels = mx.sym.reshape(data=source_words, shape=(-1,), name="source_label")
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        self.model_loss = loss.get_loss(self.config.config_loss)
        self.reconstruction_loss = loss.get_loss(self.config.config_loss)

        data_names = [C.SOURCE_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len)
            reconstructed_sequence = self.reconstructor.decode_sequence(target_decoded, target_embed_length, target_embed_seq_len,
                                                          source_embed, source_embed_length, source_embed_seq_len)

            # target_decoded: (batch_size * target_seq_len, decoder_depth)
            target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))
            
            reconstructed_sequence = mx.sym.reshape(data=reconstructed_sequence, shape=(-3, 0))
            
            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(target_decoded)
            reconstructed_logits = self.reconstruction_output_layer(reconstructed_sequence)

            loss_output = self.model_loss.get_loss(logits, labels)
            loss_reconstruction_output = self.reconstruction_loss.get_loss(reconstructed_logits, source_labels, reconstruction=True)
            
            loss_output = loss_output + self._r_lambda * loss_reconstruction_output

            return mx.sym.Group(loss_output), data_names, label_names

        if self.config.lhuc:
            arguments = sym_gen(default_bucket_key)[0].list_arguments()
            fixed_param_names = [a for a in arguments if not a.endswith(C.LHUC_NAME)]
        else:
            fixed_param_names = self.fixed_param_names

        if self._bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context,
                                                 compression_params=self._gradient_compression_params,
                                                 fixed_param_names=fixed_param_names)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.config_data.max_seq_len_source, self.config.config_data.max_seq_len_target)
            symbol, _, __ = sym_gen(default_bucket_key)
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context,
                                        compression_params=self._gradient_compression_params,
                                        fixed_param_names=fixed_param_names)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=True,
                         force_rebind=True,
                         grad_req='write')

        self.module.symbol.save(os.path.join(self.output_dir, C.SYMBOL_NAME))

        self.save_version(self.output_dir)
        self.save_config(self.output_dir)

    def run_forward_backward(self, batch: mx.io.DataBatch, metric: mx.metric.EvalMetric):
        """
        Runs forward/backward pass and updates training metric(s).
        """
        self.module.forward_backward(batch)
        self.module.update_metric(metric, batch.label)

    def update(self):
        """
        Updates parameters of the module.
        """
        self.module.update()

    def get_gradients(self) -> Dict[str, List[mx.nd.NDArray]]:
        """
        Returns a mapping of parameters names to gradient arrays. Parameter names are prefixed with the device.
        """
        # We may have None if not all parameters are optimized
        return {"dev_%d_%s" % (i, name): exe.grad_arrays[j] for i, exe in enumerate(self.executors) for j, name in
                enumerate(self.executor_group.arg_names)
                if name in self.executor_group.param_names and self.executors[0].grad_arrays[j] is not None}

    def get_global_gradient_norm(self) -> float:
        """
        Returns global gradient norm.
        """
        # average norm across executors:
        exec_norms = [global_norm([arr for arr in exe.grad_arrays if arr is not None]) for exe in self.executors]
        norm_val = sum(exec_norms) / float(len(exec_norms))
        norm_val *= self.optimizer.rescale_grad
        return norm_val

    def rescale_gradients(self, scale: float):
        """
        Rescales gradient arrays of executors by scale.
        """
        for exe in self.executors:
            for arr in exe.grad_arrays:
                if arr is None:
                    continue
                arr *= scale

    def prepare_batch(self, batch: mx.io.DataBatch):
        """
        Pre-fetches the next mini-batch.

        :param batch: The mini-batch to prepare.
        """
        self.module.prepare(batch)

    def evaluate(self, eval_iter: data_io.BaseParallelSampleIter, eval_metric: mx.metric.EvalMetric):
        """
        Resets and recomputes evaluation metric on given data iterator.
        """
        for eval_batch in eval_iter:
            self.module.forward(eval_batch, is_train=False)
            self.module.update_metric(eval_metric, eval_batch.label)

    @property
    def current_module(self) -> mx.module.Module:
        # As the BucketingModule does not expose all methods of the underlying Module we need to directly access
        # the currently active module, when we use bucketing.
        return self.module._curr_module if self._bucketing else self.module

    @property
    def executor_group(self):
        return self.current_module._exec_group

    @property
    def executors(self):
        return self.executor_group.execs

    @property
    def loss(self):
        return self.model_loss

    @property
    def optimizer(self) -> Union[mx.optimizer.Optimizer, SockeyeOptimizer]:
        """
        Returns the optimizer of the underlying module.
        """
        # TODO: Push update to MXNet to expose the optimizer (Module should have a get_optimizer method)
        return self.current_module._optimizer

    def initialize_optimizer(self, config: OptimizerConfig):
        """
        Initializes the optimizer of the underlying module with an optimizer config.
        """
        self.module.init_optimizer(kvstore=config.kvstore,
                                   optimizer=config.name,
                                   optimizer_params=config.params,
                                   force_init=True)  # force init for training resumption use case

    def save_optimizer_states(self, fname: str):
        """
        Saves optimizer states to a file.

        :param fname: File name to save optimizer states to.
        """
        self.current_module.save_optimizer_states(fname)

    def load_optimizer_states(self, fname: str):
        """
        Loads optimizer states from file.

        :param fname: File name to load optimizer states from.
        """
        self.current_module.load_optimizer_states(fname)

    def initialize_parameters(self, initializer: mx.init.Initializer, allow_missing_params: bool):
        """
        Initializes the parameters of the underlying module.

        :param initializer: Parameter initializer.
        :param allow_missing_params: Whether to allow missing parameters.
        """
        self.module.init_params(initializer=initializer,
                                arg_params=self.params,
                                aux_params=self.aux_params,
                                allow_missing=allow_missing_params,
                                force_init=False)

    def log_parameters(self):
        """
        Logs information about model parameters.
        """
        arg_params, aux_params = self.module.get_params()
        total_parameters = 0
        info = []  # type: List[str]
        for name, array in sorted(arg_params.items()):
            info.append("%s: %s" % (name, array.shape))
            total_parameters += reduce(lambda x, y: x * y, array.shape)
        logger.info("Model parameters: %s", ", ".join(info))
        if self.fixed_param_names:
            logger.info("Fixed model parameters: %s", ", ".join(self.fixed_param_names))
        logger.info("Total # of parameters: %d", total_parameters)

    def save_params_to_file(self, fname: str):
        """
        Synchronizes parameters across devices, saves the parameters to disk, and updates self.params
        and self.aux_params.

        :param fname: Filename to write parameters to.
        """
        arg_params, aux_params = self.module.get_params()
        self.module.set_params(arg_params, aux_params)
        self.params = arg_params
        self.aux_params = aux_params
        super().save_params_to_file(fname)

    def load_params_from_file(self, fname: str, allow_missing_params: bool = False):
        """
        Loads parameters from a file and sets the parameters of the underlying module and this model instance.

        :param fname: File name to load parameters from.
        :param allow_missing_params: If set, the given parameters are allowed to be a subset of the Module parameters.
        """
        super().load_params_from_file(fname)  # sets self.params & self.aux_params
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=allow_missing_params)

    def install_monitor(self, monitor_pattern: str, monitor_stat_func_name: str):
        """
        Installs an MXNet monitor onto the underlying module.

        :param monitor_pattern: Pattern string.
        :param monitor_stat_func_name: Name of monitor statistics function.
        """
        self._monitor = mx.monitor.Monitor(interval=C.MEASURE_SPEED_EVERY,
                                           stat_func=C.MONITOR_STAT_FUNCS.get(monitor_stat_func_name),
                                           pattern=monitor_pattern,
                                           sort=True)
        self.module.install_monitor(self._monitor)
        logger.info("Installed MXNet monitor; pattern='%s'; statistics_func='%s'",
                    monitor_pattern, monitor_stat_func_name)

    @property
    def monitor(self) -> Optional[mx.monitor.Monitor]:
        return self._monitor


def global_norm(ndarrays: List[mx.nd.NDArray]) -> float:
    # accumulate in a list, as asscalar is blocking and this way we can run the norm calculation in parallel.
    norms = [mx.nd.square(mx.nd.norm(arr)) for arr in ndarrays if arr is not None]
    return sqrt(sum(norm.asscalar() for norm in norms))

