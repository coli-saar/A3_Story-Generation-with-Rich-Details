import copy

import time
import torch
import os
import shutil

from allennlp.models.language_model import LanguageModel
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.callbacks import validate, checkpoint, track_metrics, gradient_norm_and_clip
from allennlp.training import checkpointer
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.nn.regularizers import RegularizerApplicator, L2Regularizer
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper


from globals import GLOBAL_CONSTANTS
from dataset_readers import LanguageModelSegmentReader
from utils import RandomSearchMetaOptimizer, PrintColors


_tag = 'ord_lm'
_train_data_path = 'data_seg_train'
_val_data_path = 'data_seg_val'
_device = 4
_n_epochs = 30
_n_hyper_search = 10


class LanguageModelOptimizer(RandomSearchMetaOptimizer):
    def __init__(self, domains, num_combinations, tag):
        parameters = {
            'batch_size': {'domain': domains['batch_size'], 'sample_criterion': '2e', 'type': 'int'},
            'ed_ncoder_size': {'domain': domains['ed_ncoder_size'], 'sample_criterion': '2e', 'type': 'int'},
            'word_embedding_size': {'domain': domains['word_embedding_size'], 'sample_criterion': '2e', 'type': 'int'},
            'lr': {'domain': domains['lr'], 'sample_criterion': '10e', 'type': 'float'},
            'l2': {'domain': domains['l2'], 'sample_criterion': '10e', 'type': 'float'},
            'dropout': {'domain': domains['dropout'], 'sample_criterion': 'u', 'type': 'float'},
            'clip': {'domain': domains['clip'], 'sample_criterion': '10e', 'type': 'float'}
        }
        super().__init__(parameters=parameters,
                         metric_names=['training_loss', 'best_validation_loss', 'best_epoch', 'time_consumed(hrs)'],
                         num_trials=num_combinations,
                         tag=tag)

    def train(self, args_hpo, index):
        """
        trains the model, and return the metrics to the meta optimizer.
        :param args_hpo:
        :param index:
        :return:
        """
        PrintColors.prYellow('\n===== training with: {}'.format(args_hpo))
        PrintColors.prGreen('----- in {} mode -----'.format('train'))
        ''' ============ LOAD DATA ================================================================================ '''
        starting_time = time.time()
        lm_dataset_reader = LanguageModelSegmentReader(global_constants=GLOBAL_CONSTANTS)
        train_data, val_data = (lm_dataset_reader.read(folder) for folder in
                                [_train_data_path, _val_data_path])
        lm_vocabulary = Vocabulary.from_instances(train_data + val_data)
        iterator = BasicIterator(batch_size=args_hpo.batch_size)
        iterator.index_with(lm_vocabulary)
        ''' ============ DEFINE MODEL ============================================================================= '''
        ''' 
        the class params 'pop' its parameters i.e. they disappear after first use. So we instantiate a Params 
        instance for each model defining execution. More than that, they turn dicts into Mutable mappings and 
        destroys the original dict. So here's your copy allennlp. Thanks. (I still love you)
        '''
        token_embedding = Embedding.from_params(vocab=lm_vocabulary,
                                                params=Params(copy.deepcopy(GLOBAL_CONSTANTS.GLOVE_PARAMS_CONFIG)))

        token_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({'tokens': token_embedding})
        ''' define encoder to wrap up an lstm feature extractor '''
        contextualizer: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(input_size=args_hpo.word_embedding_size,
                          hidden_size=args_hpo.ed_ncoder_size,
                          bidirectional=False, batch_first=True))

        model = LanguageModel(vocab=lm_vocabulary,
                              text_field_embedder=token_embedder,
                              contextualizer=contextualizer,
                              dropout=args_hpo.dropout,
                              regularizer=RegularizerApplicator([('l2', L2Regularizer(alpha=args_hpo.l2))]),
                              )\
            .cuda(_device)

        ''' ============ TRAIN ================================================================================ '''
        '''  callbacks  '''
        if index == 0:
            for file in os.listdir(os.path.join(*['.', 'lm_models'])):
                path = os.path.join(*['.', 'lm_models', file])
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        serialization_path = 'models_lm_{}_{}'.format(_tag, index)
        serialization_path_longer = os.path.join(*['.', 'lm_models', serialization_path])
        vocab_path = 'vocab_lm_{}_{}'.format(_tag, index)
        vocab_dir_longer = os.path.join(*['.', 'lm_models', vocab_path])
        if not os.path.exists(serialization_path_longer):
            os.mkdir(serialization_path_longer)
        callbacks = list()
        ''' for validation '''
        callbacks.append(validate.Validate(validation_data=val_data, validation_iterator=iterator))
        ''' for early stopping. it tracks 'loss' returned by model.forward() '''
        callbacks.append(track_metrics.TrackMetrics(patience=3))
        ''' for grad clipping '''
        callbacks.append(gradient_norm_and_clip.GradientNormAndClip(grad_clipping=args_hpo.clip))
        ''' 
            for checkpointing
            TODO: NOTE:serialization path CANNOT exist before training ??
        '''
        model_checkpointer = checkpointer.Checkpointer(serialization_dir=serialization_path_longer,
                                                       num_serialized_models_to_keep=1)
        callbacks.append(checkpoint.Checkpoint(checkpointer=model_checkpointer))
        ''' for sample generations '''

        callback_trainer = CallbackTrainer(
            model=model,
            training_data=train_data,
            iterator=iterator,
            optimizer=torch.optim.Adam(model.parameters(), lr=args_hpo.lr),
            num_epochs=_n_epochs,
            serialization_dir=serialization_path_longer,
            cuda_device=_device,
            callbacks=callbacks
        )

        ''' trainer saves the model, but the vocabulary needs to be saved, too '''
        lm_vocabulary.save_to_files(vocab_dir_longer)

        ''' check the metric names to synchronize with the class '''
        metrics = callback_trainer.train()
        metrics['time_consumed(hrs)'] = round((time.time() - starting_time) / 3600, 4)

        return metrics


if __name__ == '__main__':

    # NOTE: visit globals.py to check addtional configurations before execution ! +++++++++++++++++++++++++++++++++++++
    meta_optimizer = LanguageModelOptimizer(
        domains={'batch_size': [16, 511],
                 'ed_ncoder_size': [128, 2047],
                 'word_embedding_size': [300, 300],
                 'lr': [1e-5, 5e-3],
                 'l2': [1e-4, 5e-2],
                 'dropout': [0.05, 0.8],
                 'clip': [1, 30]},
        num_combinations=_n_hyper_search,
        tag=_tag
    )

    meta_optimizer.search()
