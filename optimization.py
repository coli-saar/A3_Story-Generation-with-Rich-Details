from allennlp.training.callbacks import checkpoint, track_metrics, gradient_norm_and_clip
from allennlp.training import checkpointer
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.data import Vocabulary

import torch
import torch.nn.functional
import os
import time
import enum
import shutil

from dataset_readers import IrregularGenerationReader, AlternatingSeq2seqIterator
from utils import PrintColors, RandomSearchMetaOptimizer, AllenNLPTensorboardLogger
from globals import GLOBAL_CONSTANTS
from flesh_model_s import AlternatingSeq2Seq
import hacked_validate as validate


class FleshGenMetaOptimizer(RandomSearchMetaOptimizer):
    """
    """

    # todo: this shoud be a part of the parent class. update.
    class Mode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'
        AUGMENTED_OPTIMIZATION = 'augment'

    def __init__(self, configurations, model):
        domains = configurations.param_domains
        parameters = {
            'batch_size': {'domain': domains['batch_size'], 'sample_criterion': '2e', 'type': 'int'},
            'ed_ncoder_size': {'domain': domains['ed_ncoder_size'], 'sample_criterion': '2e', 'type': 'int'},
            'lr': {'domain': domains['lr'], 'sample_criterion': '10e', 'type': 'float'},
            'percent_student': {'domain': domains['percent_student'], 'sample_criterion': '10e', 'type': 'float'},
            'l2': {'domain': domains['l2'], 'sample_criterion': '10e', 'type': 'float'},
            'dropout': {'domain': domains['dropout'], 'sample_criterion': 'u', 'type': 'float'},
            'clip': {'domain': domains['clip'], 'sample_criterion': '10e', 'type': 'float'},
            'max_progress_len': {'domain': domains['max_progress_len'], 'sample_criterion': 'u', 'type': 'int'},
            'num_hidden_layeres': {'domain': domains['num_hidden_layeres'], 'sample_criterion': 'u', 'type': 'int'}
        }
        super().__init__(parameters=parameters,
                         metric_names=['training_loss', 'best_validation_loss', 'training_accuracy',
                                       'best_validation_accuracy', 'best_epoch', 'time_consumed(hrs)'],
                         #  'training_irregular_loss', 'training_regular_loss', 'val_irregular_loss',
                         #  'val_regular_loss'
                         num_trials=configurations.num_trials,
                         tag=configurations.tag)
        self.configurations = configurations
        self.model = model

    def train(self, args_hpo, index):
        """
        trains the model, and return the metrics to the meta optimizer.
        :param args_hpo:
        :param index:
        :return:
        """

        PrintColors.prYellow('\n===== training with: {}'.format(args_hpo))
        PrintColors.prGreen('----- in mode {} -----'.format(self.configurations.MODE))
        ''' ============ LOAD DATA ================================================================================ '''
        starting_time = time.time()
        reader = IrregularGenerationReader(global_constants=GLOBAL_CONSTANTS,
                                           merge_irregular_events=self.configurations.MERGE_IRREGULAR_EVENTS)
        ''' .read returns list of instances '''
        train_data, val_data = (reader.read(folder) for folder in
                                [self.configurations.train_data_path,
                                 self.configurations.val_data_path])
        ''' build vocabulary.  '''
        vocabulary = Vocabulary.from_instances(train_data + val_data)
        ''' note that iterator is NOT associated with data. this project does not supports easy batching '''
        iterator = AlternatingSeq2seqIterator(batch_size=args_hpo.batch_size)
        ''' tell iterator how to numericalize input '''
        iterator.index_with(vocabulary)

        ''' ============ DEFINE MODEL ============================================================================= '''
        model = self.model(args_hpo, vocabulary, configurations=self.configurations, dataset_reader=reader)
        # load language model for train-time MMI generation
        aug_mode = self.configurations.MODE == FleshGenMetaOptimizer.Mode.AUGMENTED_OPTIMIZATION
        model.load_language_model(vocab_folder=GLOBAL_CONSTANTS.aug_language_model_vocab_path if aug_mode
                                  else GLOBAL_CONSTANTS.language_model_vocab_path,
                                  model_path=GLOBAL_CONSTANTS.aug_language_model_path if aug_mode
                                  else GLOBAL_CONSTANTS.language_model_path,
                                  aug_mode=aug_mode)
        ''' ============ TRAIN ================================================================================ '''
        '''  callbacks  '''
        if index == 0:
            for file in os.listdir(os.path.join(*['.', 'models'])):
                path = os.path.join(*['.', 'models', file])
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        serialization_path = 'models_{}_{}'.format(self.configurations.tag, index)
        serialization_path_longer = os.path.join(*['.', 'models', serialization_path])
        vocab_path = 'vocab_{}_{}'.format(self.configurations.tag, index)
        vocab_dir_longer = os.path.join(*['.', 'models', vocab_path])
        if not os.path.exists(serialization_path_longer):
            os.mkdir(serialization_path_longer)
        callbacks = list()
        ''' for early stopping. it tracks 'loss' returned by model.forward() '''
        callbacks.append(track_metrics.TrackMetrics(patience=self.configurations.PATIENCE))
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
        sample_generation_callback = AlternatingSeq2Seq.SampleGeneration(
            agenda_s=self.configurations.SAMPLE_AGENDA_S,
            dataset_reader=reader,
            scenario_s=self.configurations.SAMPLE_SCENARIO_S,
            index=index)
        callbacks.append(sample_generation_callback)
        ''' for tensorboard '''
        loss_tracker = None
        if self.configurations.LOG_TO_TENSORBOARD:
            ''' it's here because tensorboard asks for a sample by default '''
            instance = reader.text_to_instance(event_context=['{}_grocery'.format(GLOBAL_CONSTANTS.beginning_event)],
                                               text_context=['{}_grocery'.format(GLOBAL_CONSTANTS.begin_of_story)],
                                               indices=[],
                                               forthcoming_event='irregular_grocery',
                                               agenda=['{}_grocery'.format(GLOBAL_CONSTANTS.beginning_event)] * 2,
                                               e_f_index=3)
            ''' call AllenNLP iterator to numericalize the instance '''
            input_to_model = list(iterator([instance], num_epochs=1))[0]
            tensorboard_logger = AllenNLPTensorboardLogger(
                log_folder='tb_logs_{}/{}'.format(self.configurations.tag, index),
                metric_s=['accuracy', 'loss'],
                input_to_model=input_to_model,
                tag=self.configurations.tag,
                index=index)
            callbacks.append(tensorboard_logger)
            loss_tracker = AlternatingSeq2Seq.TrackRegularAndIrregularLoss(tensorboard_logger.writer)
            callbacks.append(loss_tracker)
        ''' for validation '''
        if self.configurations.LOG_TO_TENSORBOARD:
            callbacks.append(validate.Validate(validation_data=val_data, validation_iterator=iterator,
                                               loss_tracker=loss_tracker))
        else:
            callbacks.append(validate.Validate(validation_data=val_data, validation_iterator=iterator))
        callback_trainer = CallbackTrainer(
            model=model,
            training_data=train_data,
            iterator=iterator,
            optimizer=torch.optim.Adam(model.parameters(), lr=args_hpo.lr),
            num_epochs=self.configurations.MAX_EPOCHS,
            serialization_dir=serialization_path_longer
            if self.configurations.MODE == FleshGenMetaOptimizer.Mode.OPTIMIZATION else None,
            cuda_device=self.configurations.DEVICE,
            callbacks=callbacks
        )

        ''' trainer saves the model, but the vocabulary needs to be saved, too '''
        vocabulary.save_to_files(vocab_dir_longer)

        ''' check the metric names to synchronize with the class '''
        metrics = callback_trainer.train()
        metrics['time_consumed(hrs)'] = round((time.time() - starting_time) / 3600, 4)

        return metrics
