"""
The random hyperparameter search class is updated to use allenNLP.common.params.Params to fit the ecosystem.

Zhai Fangzhou
2020.02.23
"""
import collections
import math
import os


# from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.training.callbacks import Callback, handle_event, Events

import dill
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


'================================ General ==============================='


class PrintColors:
    @staticmethod
    def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
    @staticmethod
    def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
    @staticmethod
    def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
    @staticmethod
    def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
    @staticmethod
    def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
    @staticmethod
    def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
    @staticmethod
    def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
    @staticmethod
    def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))


class Struct(dict):
    """
    extend a dictionary so we can access its keys as attributes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)


class TextTokenizer:
    def __init__(self, text, vocabulary_size=-1):
        """
        :param text: a corpus as a list of tokens
        """
        assert text
        self._encoder = dict()
        if vocabulary_size == -1:
            token_counter = collections.Counter(text).most_common(len(text))
        else:
            token_counter = collections.Counter(text).most_common(vocabulary_size)
        for type_, count in token_counter:
            self._encoder[type_] = len(self._encoder)
        self._decoder = dict(zip(self._encoder.values(), self._encoder.keys()))

    def encode(self, text):
        if type(text) == str:
            return torch.tensor(self._encoder[text])
        else:
            return torch.tensor([self._encoder[text[i]] for i in range(len(text))])

    def decode(self, index):
        if (type(index) == torch.Tensor and len(index.size()) == 0) or\
                (type(index) == np.ndarray and len(index.shape) == 0) or type(index) == int:
            return self._decoder[index]
        else:
            return [self._decoder[index[i]] for i in range(len(index))]

    def append_type(self, new_type):
        if new_type in self._encoder:
            PrintColors.prRed('warning: type {} already exists.'.format(new_type))
        else:
            index = len(self._encoder)
            self._encoder[new_type] = index
            self._decoder[index] = new_type

    def append_type_S(self, new_type_s: list):
        for new_type in new_type_s:
            self.append_type(new_type)

    @property
    def vocabulary_size(self):
        return len(self._encoder)


def last_index_of(l, element):
    return len(l) - l[::-1].index(element) - 1


def to_categorical(y, num_classes):
    """
    projects a number to one-hot encoding
    """
    return np.eye(num_classes, dtype='uint8')[y]


def pad_sequence(sequence, target_length: int, padding_token: str):
    return list(sequence) + [padding_token] * (target_length - len(sequence))


def preserve(f: float, n: int = 2):
    """ preserves n valid digits of float number f"""
    return float(('{:.' + str(n) + 'g}').format(f))


'========================== ALLENNLP Related ============================'


class AllenNLPTensorboardLogger(Callback):
    """
        log metrics, etc. to tensorboard with torch.utils.tensorboard for AllenNLP models.
    """

    def __init__(self, log_folder, metric_s, input_to_model, tag, index,
                 log_embeddings=True,
                 add_graph=False):
        """

        :param log_folder: full path of log folder.
        :param metric_s: metrics to log.
        :param log_embeddings: bool. If True, the callback calls
                trainer.model.prepare_embeddings_for_tensorboard()
            to access weights. The function is supposed to return a list of dict each with
            'mat': V * E
            'metadata': V
            'tag': the namespace of the embeddings, e.g. 'words'.
        """
        self.writer = SummaryWriter(log_folder)
        self.metric_s = metric_s
        self.log_embeddings = log_embeddings
        self.input_to_model = input_to_model
        self.add_graph = add_graph
        self.tag = tag
        self.index = index
        super().__init__()

    @handle_event(Events.TRAINING_START)
    def log_graph(self, trainer):
        ''' fixme: yes this is fishy, it seems input examples are mandatory. '''
        if self.add_graph:
            self.writer.add_graph(trainer.model, input_to_model=self.input_to_model)

    @handle_event(Events.TRAINING_END)
    def log_hparams(self, trainer):
        self.writer.add_hparams(
            hparam_dict=trainer.model.args_hpo.__dict__,
            metric_dict={'hp_val_accuracy': trainer.metrics['best_validation_accuracy'],
                         'hp_val_loss': trainer.metrics['best_validation_loss'],
                         'hp_train_accuracy': trainer.metrics['training_accuracy'],
                         'hp_train_loss': trainer.metrics['training_loss']})

    @handle_event(Events.EPOCH_END)
    def log_metrics(self, trainer):
        if self.metric_s:
            for metric in self.metric_s:
                for phase in ['train', 'val']:
                    self.writer.add_scalar(tag='{}_{}'.format(phase, metric),
                                           scalar_value=getattr(trainer, '{}_metrics'.format(phase))[metric],
                                           global_step=trainer.epoch_number)

    @handle_event(Events.EPOCH_END)
    def log_embedding(self, trainer):
        if self.log_embeddings:
            embedding_data = trainer.model.prepare_embeddings_for_tensorboard()
            for embedder in embedding_data:
                self.writer.add_embedding(mat=embedder['mat'],
                                          metadata=embedder['metadata'],
                                          tag=embedder['tag'],
                                          global_step=trainer.epoch_number)

    @handle_event(Events.TRAINING_END)
    def close_writer(self, trainer):
        self.writer.close()


class RandomSearchMetaOptimizer(Registrable):
    """
    to perform random hyper-parameter search
    this is a class that abuses GPU to squeeze performance of small models
    usage:
        inherit the class and override self.train()
        call self.search to perform the searching and log results
    """

    def __init__(self, parameters: dict, num_trials: int, tag: str, metric_names: list):
        """
        generates parameter combinations for random parameter search, stored in self.combinations as dictionaries
        all sampling parameters are dictionaries formed as dict{hyper_name: value}

        :param parameters:
            the parameters involved in hyper parameter search. a dict of dicts. The first hierarchy of keys are the
            parameter names; the second hierachy should include the following:
            'domain': the range of the parameter. should be a tuple.
                Note: for exponential sampling, the upperbound of the domain does not get sampled.
            'sample_criterion':
                'u': uniform over the domain
                '2e': the parameter's logrithm wrt 2 is sampled uniformly as an INTEGER
                '10e': the parameter's logrithm is sampled uniformly as a FLOAT
            'type':
                'int': floor and returns an integer
                'float': float, direct copy
        :param metric_names:
            the name of the metrics returned by .train() that should be logged. if allennlp is used, the metric names
            are usually expected from trainer.train(), i.e. trainer.metrics.
        :param num_trials:
        :param tag:
        """
        self.hyper_combs = [Struct() for _ in range(num_trials)]
        self.num_trials = num_trials
        self.tag = tag
        self.log_path = "logs_{}.csv".format(self.tag)
        self.combs_path = "hyper_combs_{}".format(self.tag)
        self.parameters = parameters
        self.hyper_names = [hyper for hyper in self.parameters]
        self.metric_names = metric_names

        ''' generate parameters '''
        for i, combination in enumerate(self.hyper_combs):
            for hyper in self.parameters:
                min_value, max_value = self.parameters[hyper]['domain']
                # note if min_value == max_value, the quantity will be returned
                assert min_value <= max_value
                if min_value == max_value:
                    combination[hyper] = min_value
                    continue
                rnd_ready = None
                if self.parameters[hyper]['sample_criterion'] == '2e':
                    assert min_value > 0
                    min_exp, max_exp = np.log2(min_value), np.log2(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(2., np.floor(rnd))
                elif self.parameters[hyper]['sample_criterion'] == '10e':
                    assert min_value > 0
                    min_exp, max_exp = np.log10(min_value), np.log10(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(10., rnd)
                elif self.parameters[hyper]['sample_criterion'] == 'u':
                    rnd_ready = np.random.uniform() * (max_value - min_value) + min_value

                if self.parameters[hyper]['type'] == 'int':
                    combination[hyper] = int(rnd_ready)
                elif self.parameters[hyper]['type'] == 'float':
                    combination[hyper] = rnd_ready

        ''' initialize log if applicable '''
        if not os.path.exists(self.log_path):
            header = 'index' + ',' + ','.join(self.hyper_names) + ',' + ','.join(self.metric_names)
            with open(self.log_path, 'w') as log:
                log.write(header + '\n')

        ''' save combinations '''
        dill.dump(self.hyper_combs, open(self.combs_path, 'wb'))

    def train(self, combination, index):
        """
        execute one round of random hyper-parameter search.
        ! this function should be overrided in derived classes.

        implement execution callbacks, model saving, etc. in this function

        input: hyper parameter combinations
        :param combination:
        :return: metrics for evaluation as a dictionary
        """
        raise NotImplementedError

    def search(self):
        print('======Performing Random Hyper Search for execution {}======'.format(self.tag))
        for execution_idx, hyper_comb in enumerate(self.hyper_combs):
            print('------ Random Hyper Search Round {} ------'.format(execution_idx))
            metrics = self.train(hyper_comb, execution_idx)
            log_line = str(execution_idx) + ',' + \
                ','.join(['{:.3g}'.format(hyper_comb[name]) for name in self.hyper_names]) + ',' + \
                ','.join(['{:.3g}'.format(metrics[name]) for name in self.metric_names])
            with open(self.log_path, 'a') as log_out:
                log_out.write(log_line + '\n')

            # todo: define a test mode parameter!!!!
            # try:
            #     print('------ Random Hyper Search Round {} ------'.format(execution_idx))
            #     metrics = self.train(hyper_comb, execution_idx)
            #     log_line = str(execution_idx) + ',' + \
            #         ','.join(['{:.3g}'.format(hyper_comb[name]) for name in self.hyper_names]) + ',' + \
            #         ','.join(['{:.3g}'.format(metrics[name]) for name in self.metric_names])
            #     with open(self.log_path, 'a') as log_out:
            #         log_out.write(log_line + '\n')
            # except RuntimeError as rte:
            #     PrintColors.prPurple(rte)
            #     continue


class MaskedPositionalEncoding(torch.nn.Module):
    """
    A variant of the original positional encoding that accepts a source_mask.
    Masked elements do not get positional embeddings.
    fixme: it seems that the original module does not support batching. I will inherit this for now.
    """

    def __init__(self, input_dim: int, device, max_len: int = 100) -> None:
        super().__init__()
        # Compute the positional encodings once in log space.
        # FINAL shape: 1 * max_len * input_dim
        positional_encoding = torch.zeros(max_len, input_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).to(device)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        todo: tested for batchsize = 1, but should work otherwise with the correct masks
            to make it compartible for larger batches, unify input with x being always rank 3, and always give
            source_mask and positions parameters.
        fixme: did i mask regular events in the agenda? -- now yes
        :param x:   (encoding) shape: batch * len * input_dim for a list of event,
                    (generation) shape: batch * len=1 * input_dim for a forthcoming event.
        :param positions:
                    (encoding) shape: batch * len
                    (generation) shape: batch * len=1
                    the positions for each item in x, according to which the positional encoding should be added.
                    for encoding sequences' event context, the position parameter might repeat.
        :param source_mask:
                    (encoding) shape: batch * len
                    (generation) shape: batch * len=1
        :return:
        """
        # pylint: disable=arguments-differ
        # add positional encodings
        batch_size = x.size()[0]
        positional_encoding = x.new_zeros(x.size())
        for index in range(batch_size):
            positional_encoding[index, :, :] = \
                self.positional_encoding[0, positions[index, :], :] * (source_mask[index, :].unsqueeze(-1))

        return x + positional_encoding


class MaskedProgressEncoding(torch.nn.Module):
    """
    Progress Encoding. A variant of the transformer positional encoding that has a max_len. Positional encodings of
    elements are added with the 'effective positions' re-scaled to fit into [0, max_len].

    Masked elements do not get positional embeddings.
    fixme: it seems that the original module does not support batching. I will inherit this for now.
    note: not verified yet.
    """

    def __init__(self, input_dim: int, device, max_len: int = 5000) -> None:
        super().__init__()
        self.max_len = max_len
        # shape: 1 * (input_dim / 2)
        self.div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))\
            .unsqueeze(0).to(device)
        # Compute the positional encodings once in log space.

        # self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor, agenda_length: int,
                source_mask: torch.Tensor = None, position: int = None) -> torch.Tensor:
        """
        :param x:   shape: batch = 1 * len * input_dim for a list of event,
                    shape: batch = 1 * input_dim for a forthcoming event.
        :param source_mask: shape: batch = 1 * len
         indicating whether progress encoding should be added to each item of the agenda
        :return:
        """
        # pylint: disable=arguments-differ
        ''' assert exactly one mode is chosen '''
        assert (source_mask is None) != (position is None)
        if source_mask is not None:
            _, length, input_dim = x.size()
            # shape: length, input_dim
            self.progress_encoding = torch.zeros(length, input_dim, requires_grad=False, device=x.device)
            # shape: batchsize=1 * length * input_dim
            effective_mask = source_mask.unsqueeze(-1).expand(x.size())
            effective_mask = effective_mask.to(x.device)
            # shape: length * 1
            indices = torch.arange(0, length).unsqueeze(1).float().to(x.device)
            # shape: length * 1
            effective_positions = self.max_len * indices / (agenda_length - 1)
            self.progress_encoding[:, 0::2] = torch.sin(effective_positions * self.div_term)
            self.progress_encoding[:, 1::2] = torch.cos(effective_positions * self.div_term)
            return x + self.progress_encoding[:x.size(1), :x.size(2)].unsqueeze(0) * effective_mask
        elif position is not None:
            batch_size, input_dim = x.size()
            # shape: batch_size=1 * input_dim
            self.progress_encoding = torch.zeros(batch_size, input_dim, requires_grad=False, device=x.device)
            effective_position = self.max_len * position / (agenda_length - 1)
            self.progress_encoding[:, 0::2] = torch.sin(effective_position * self.div_term)
            self.progress_encoding[:, 1::2] = torch.cos(effective_position * self.div_term)
            emb_with_progress_encoding = x + self.progress_encoding[:, :x.size(-1)]
            return emb_with_progress_encoding
