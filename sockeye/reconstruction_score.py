"""
CLI to rescore an nbest list of translations with reconstruction loss.
"""

import argparse
import json
import os
from contextlib import ExitStack
from typing import List, Optional, Tuple

from . import arguments
from . import constants as C
from . import log
from . import utils
from . import vocab
from . import data_io
from . import model
from . import reconstruction_scoring
from .output_handler import get_output_handler

logger = log.setup_main_logger(__name__, console=True, file_logging=False)


def create_data_iter(args: argparse.Namespace):
    
    sources = [args.source] + args.source_factors
    sources = [str(os.path.abspath(source)) for source in sources]
    
    source_vocabs = vocab.load_source_vocabs(args.model) 
    target_vocab = vocab.load_target_vocab(args.model)
    buckets = []
    
    source_sentences = []
    target_sentences = []
    
    bucketing=not args.no_bucketing
    
    source_files = [utils.smart_open(args.source) for source in sources]
    
    with utils.smart_open(args.hypotheses) as hypotheses:
        for i, (source_line, hypothesis_line) in enumerate(zip(source_files[0], hypotheses)):
            
            hypotheses = json.loads(hypothesis_line.rstrip())
            utils.check_condition('translations' in hypotheses,
                              "Reranking requires nbest JSON input with 'translations' key present.")
            
            batch_size = len(hypotheses['translations'])
            sources_list = [source_line.rstrip()] * batch_size
            
            source_sentences.extend(sources_list)
            target_sentences.extend(hypotheses['translations'])
            
    source_sequence_readers = [data_io.ReconstructionSequenceReader(source_sentences,
                                                                    source_vocabs[0], add_eos=True)]
    target_sequence_reader = data_io.ReconstructionSequenceReader(target_sentences, target_vocab, add_bos=True)
    
    max_source = max([source_sentence.split() for source_sentence in source_sentences], key=len)
    max_seq_len_source = len(max_source) +1 # <eos>
    max_target = max([target_sentence.split() for target_sentence in target_sentences], key=len)
    max_seq_len_target = len(max_target) +1 # <eos>
    
    if len(sources) > 1:
        factors_sequence_readers = [SequenceReader(source, vocab, add_eos=True) for source,
                                    vocab in zip(sources[1:], source_vocabs[1:])]
        source_sequence_readers.extend(factors_sequence_readers)
        
    length_statistics = data_io.calculate_length_statistics(source_sequence_readers,
                                                            target_sequence_reader,
                                                            max_seq_len_source,
                                                            max_seq_len_target)

    logger.info("%d sequences of maximum length (%d, %d) in source and hypotheses.",
                length_statistics.num_sents, max_seq_len_source, max_seq_len_target)
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)",
                length_statistics.length_ratio_mean,
                length_statistics.length_ratio_std)
        
    buckets = data_io.define_parallel_buckets(max_seq_len_source, 
                                              max_seq_len_target, 
                                              args.bucket_width,
                                              length_statistics.length_ratio_mean) if bucketing else [(max_seq_len_source, max_seq_len_target)]
    
    data_statistics = data_io.get_data_statistics(source_sequence_readers,
                                                  target_sequence_reader,
                                                  buckets,
                                                  length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                                  source_vocabs,
                                                  target_vocab)
    
    bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words=False,
                                                   batch_num_devices=1,
                                                   data_target_average_len=data_statistics.average_len_target_per_bucket)
    
    data_loader = data_io.RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)
    
    parallel_data = data_loader.load(source_sequence_readers, 
                                     target_sequence_reader,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, args.fill_up)
    
    data_iter = data_io.ParallelSampleIter(data=parallel_data,
                                           buckets=buckets,
                                           batch_size=batch_size,
                                           bucket_batch_sizes=bucket_batch_sizes,
                                           num_factors=len(sources),
                                           permute=False)
            
            
    return data_iter, source_vocabs, target_vocab          
            

def score(args: argparse.Namespace):
    utils.log_basic_info(args)
    with ExitStack() as exit_stack:
        context = utils.determine_context(device_ids=args.device_ids,
                                          use_cpu=args.use_cpu,
                                          disable_device_locking=args.disable_device_locking,
                                          lock_dir=args.lock_dir,
                                          exit_stack=exit_stack)

        logger.info("Scoring Device(s): %s", ", ".join(str(c) for c in context))

        args.no_bucketing = True
        args.fill_up = 'zeros'
        logger.info(args)

        data_iter, source_vocabs, target_vocab = create_data_iter(args)
        model_config = model.SockeyeModel.load_config(os.path.join(args.model, C.CONFIG_NAME))
        
        if args.checkpoint is None:
            params_fname = os.path.join(args.model, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(args.model, C.PARAMS_NAME % args.checkpoint)

        reconstruction_scoring_model = reconstruction_scoring.ReconstructionScoringModel(config=model_config,
                                                                                         params_fname=params_fname,
                                                                                         context=context,
                                                                                         provide_data=data_iter.provide_data,
                                                                                         provide_label=data_iter.provide_label,
                                                                                         default_bucket_key=data_iter.default_bucket_key,
                                                                                         score_type=args.score_type,
                                                                                         bucketing=False,
                                                                                         softmax_temperature=args.softmax_temperature)
        
        scorer = reconstruction_scoring.Scorer(reconstruction_scoring_model,
                                               source_vocabs, 
                                               target_vocab,
                                               r_lambda=args.reconstruction_lambda)
        scorer.score(data_iter,
                     get_output_handler(output_type=args.output_type,
                                        output_fname=args.output),
                     get_output_handler(output_type=C.OUTPUT_HANDLER_NBEST_NEMATUS_FORMAT,
                                        output_fname=args.nbest_nematus))
    
            
            
def main():
    """
    Commandline interface to rescore nbest lists with reconstruction loss.
    """
    log.log_sockeye_version(logger)

    params = argparse.ArgumentParser(description="Rerank nbest lists of translations with reconstruction loss.")
    arguments.add_reconstruction_score_args(params)
    args = params.parse_args()
    score(args)
    
    


if __name__ == "__main__":
    main()
