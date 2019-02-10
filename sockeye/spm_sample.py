import os
from typing import Dict, Iterable, List
import logging

import mxnet as mx
import sentencepiece as spm

from sockeye.log import setup_main_logger, log_sockeye_version
from . import constants as C
from . import utils
from . import vocab
from . import data_io

logger = logging.getLogger(__name__)

class SentencepieceSampler:
    def __init__(self,
                 spm_alpha: float,
                 spm_nbest_size: int,
                 spm_model: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 output_folder: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 original_sources: List[str],
                 original_target: str,
                 bucketing: bool,
                 bucket_width: int,
                 batch_size: int,
                 batch_by_words: bool,
                 batch_num_devices: int,
                 fill_up: str,
                 permute: bool):
        
        self.spm_alpha=spm_alpha
        self.spm_nbest_size=spm_nbest_size
        self.sp = spm.SentencePieceProcessor()
        logger.info("Loading sentencepiece language model for segmentation from %s" % spm_model)
        self.sp.Load(spm_model)
        self.source_vocabs = source_vocabs
        self.source_vocab = source_vocabs[0]
        self.source_vocab_inv= vocab.reverse_vocab(self.source_vocab)
        self.strip_ids_s = {self.source_vocab[C.EOS_SYMBOL], C.PAD_ID}
        self.target_vocab = target_vocab
        self.target_vocab_inv= vocab.reverse_vocab(self.target_vocab)
        self.strip_ids_t = {self.target_vocab[C.EOS_SYMBOL], C.PAD_ID}
        self.output_folder = output_folder
        self.max_seq_len_source=max_seq_len_source
        self.max_seq_len_target=max_seq_len_target
        self.original_sources = original_sources
        self.original_target = original_target
        self.bucketing=bucketing
        self.bucket_width=bucket_width
        self.batch_size= batch_size
        self.batch_by_words =batch_by_words
        self.batch_num_devices= batch_num_devices
        self.fill_up= fill_up
        self.permute=permute
        self._counter=0
        
    # read text files and use spm model to split words    
    def sample(self):
        output_name_s = os.path.join(self.output_folder, C.SOURCE_SPM + str(self._counter))
        output_name_t = os.path.join(self.output_folder, C.TARGET_SPM + str(self._counter))
    
        spm_file_s = open(output_name_s, 'w')
        spm_file_t = open(output_name_t, 'w')
        
        for src_line, trg_line in zip( open(self.original_sources[0], 'r'), open(self.original_target, 'r')):
            encoded_source = self.sp.sample_encode_as_pieces(src_line, self.spm_nbest_size, self.spm_alpha)
            encoded_target = self.sp.sample_encode_as_pieces(trg_line, self.spm_nbest_size, self.spm_alpha)
            spm_file_s.write(C.TOKEN_SEPARATOR.join(w for w in encoded_source) +"\n")
            spm_file_t.write(C.TOKEN_SEPARATOR.join(w for w in encoded_target) +"\n")

        spm_file_s.close()
        spm_file_t.close()
        
        logger.info("Sampled data for epoch %i" % (self._counter))
        self._counter +=1
        return output_name_s, output_name_t
    
    def split_validation(self, validation_sources, validation_target):
        # dev set: best segmentation, no sampling
        output_name_valid_s = os.path.join(self.output_folder, C.SOURCE_SPM_VALIDATION)
        output_name_valid_t = os.path.join(self.output_folder, C.TARGET_SPM_VALIDATION )
        spm_file_valid_s = open(output_name_valid_s, 'w')
        spm_file_valid_t = open(output_name_valid_t, 'w')
        
        for src_line, trg_line in zip( open(validation_sources[0], 'r'), open(validation_target, 'r')):
            encoded_source = self.sp.encode_as_pieces(src_line)
            encoded_target = self.sp.encode_as_pieces(trg_line)
            spm_file_valid_s.write(C.TOKEN_SEPARATOR.join(w for w in encoded_source) +"\n")
            spm_file_valid_t.write(C.TOKEN_SEPARATOR.join(w for w in encoded_target) +"\n")

        spm_file_valid_s.close()
        spm_file_valid_t.close()
        
        logger.info("Segmented dev set with sentencepiece language model (best segmentation)")
        return output_name_valid_s, output_name_valid_t
        
        
    def resample(self):
           new_source, new_target = self.sample()
           new_sources = self.original_sources
           new_sources[0] =new_source
           length_statistics = data_io.analyze_sequence_lengths(new_sources, new_target, 
                                                         self.source_vocabs, self.target_vocab, self.max_seq_len_source, self.max_seq_len_target)
           buckets = data_io.define_parallel_buckets(self.max_seq_len_source, 
                                             self.max_seq_len_target, self.bucket_width,
                                             length_statistics.length_ratio_mean) if self.bucketing else [
                                      (self.max_seq_len_source, self.max_seq_len_target)]
                                             
           sources_sentences, target_sentences = data_io.create_sequence_readers(new_sources, new_target, self.source_vocabs, self.target_vocab)
           data_statistics = data_io.get_data_statistics(sources_sentences, target_sentences, buckets,
                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                          self.source_vocabs, self.target_vocab)
           
           bucket_batch_sizes = data_io.define_bucket_batch_sizes(buckets,
                                                        self.batch_size,
                                                        self.batch_by_words,
                                                        self.batch_num_devices,
                                                        data_statistics.average_len_target_per_bucket)
           data_statistics.log(bucket_batch_sizes)
           
           # Pass 3: Load the data into memory and return the iterator.
           data_loader = data_io.RawParallelDatasetLoader(buckets=buckets,
                                                eos_id=self.target_vocab[C.EOS_SYMBOL],
                                                pad_id=C.PAD_ID)
           
           training_data = data_loader.load(sources_sentences, target_sentences,
                                            data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes, self.fill_up)
           
           train_iter = data_io.ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=self.batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=len(new_sources),
                                    permute=self.permute,
                                    use_spm=True,
                                    sentencepiece_sampler=self)
           return train_iter

    def get_epoch_counter(self):
        return self._counter
    
    def increase_epoch_counter(self):
        self._counter +=1
