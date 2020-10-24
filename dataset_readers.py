import os

''' data set reader '''
from allennlp.data import Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from typing import Iterable, Deque
from allennlp.data.dataset import Batch
from collections import deque, namedtuple
import random
from allennlp.common.util import lazy_groups_of

from typing import List, Iterator

from globals import GLOBAL_CONSTANTS
from utils import last_index_of


@DatasetReader.register('irregular_generation')
class IrregularGenerationReader(DatasetReader):
    """
    Dataset reader for arranging data to allow for generating stories with irregular events.
    """

    def __init__(self, global_constants, merge_irregular_events=True):
        super().__init__(lazy=False)
        self.word_indexers = {'words': SingleIdTokenIndexer(namespace='words')}
        self.event_indexers = {'events': SingleIdTokenIndexer(namespace='events')}
        self.merge_irregular_events = merge_irregular_events
        self.global_constants = global_constants

    def merged(self, event: str):
        for prefix in self.global_constants.flesh_event_prefixes:
            if event.__contains__(prefix):
                scenario = event[last_index_of(event, '_') + 1:]
                return '{}_{}'.format(self.global_constants.empty_event_label, scenario)
        return event

    def is_flesh_event(self, event: str):
        for prefix in self.global_constants.flesh_event_prefixes:
            if event.__contains__(prefix):
                return True
        return False

    def text_to_instance(self,
                         agenda: List[str],
                         event_context: List[str],
                         text_context: List[str],
                         indices: List[int],
                         forthcoming_event: str,
                         e_f_index: int,
                         # set to none so it is compartible with previous models. same for model.forward.
                         prev_reg_event: str = None,
                         succ_reg_event: str = None,
                         target: List[str] = None) -> Instance:
        """
        ! a dummy end of story event is added to the end of the agenda. No text will be generated for it.
        :param event_context:
        :param text_context:
        :param indices: the indices for segmentation. collect indices of the first token of the successive segment, so
                        slicing could be used directly.
        :param forthcoming_event:
        :param target: the target sequence, ending with <eos>.
        :return:
        """
        ''' merge context event labels if configured. Forth coming event is always merged. '''
        scenario = forthcoming_event[last_index_of(forthcoming_event, '_') + 1:]
        agenda_field = TextField([Token(self.merged(event)) for event in agenda
                                  + ['{}_{}'.format(GLOBAL_CONSTANTS.ending_event, scenario)]], self.event_indexers)
        if self.merge_irregular_events:
            event_context_field = TextField([Token(self.merged(t)) for t in event_context], self.event_indexers)
        else:
            event_context_field = TextField([Token(t) for t in event_context], self.event_indexers)
        text_context_field = TextField([Token(t) for t in text_context], self.word_indexers)
        forthcoming_event_field = TextField([Token(self.merged(forthcoming_event))], self.event_indexers)

        fields = {'agenda': agenda_field,
                  'event_context': event_context_field,
                  'text_context': text_context_field,
                  'forthcoming_event': forthcoming_event_field,
                  'indices': MetadataField(indices),
                  'is_flesh_event': MetadataField(self.is_flesh_event(forthcoming_event)),
                  'e_f_index': MetadataField(e_f_index)
                  }
        if target:
            target_field = TextField([Token(t) for t in target], self.word_indexers)
            fields['target'] = target_field
        if prev_reg_event:
            prev_reg_event_field = TextField([Token(self.merged(prev_reg_event))], self.event_indexers)
            fields['previous_regular_event'] = prev_reg_event_field
        if succ_reg_event:
            succ_reg_event_field = TextField([Token(self.merged(succ_reg_event))], self.event_indexers)
            fields['succesive_regular_event'] = succ_reg_event_field

        ''' an instance of data is constructed with a dict '''
        return Instance(fields)

    def _read(self, file_path) -> Iterator[Instance]:
        """"""
        files = os.listdir(file_path)
        ''' process each files in the folder '''
        for file in files:
            path = os.path.join(file_path, file)

            text_context = ['{}_{}'.format(self.global_constants.begin_of_story, file)]
            event_context = ['{}_{}'.format(self.global_constants.beginning_event, file)]
            indices = list()
            agenda = event_context.copy()

            instance_buffer = list()
            instance_tuple = namedtuple('instance', ['event_context', 'text_context', 'indices', 'forthcoming_event',
                                                     'target', 'e_f_index', 'prev_reg_event'])
            previous_regular_event = '{}_{}'.format(self.global_constants.beginning_event, file)
            infile = open(path, 'r')
            for line in infile:
                splited_line = line.lower().split('\t')
                if not splited_line[0].__contains__('<end_of_story>'):
                    ''' normal line. yield previous instance first. '''
                    if splited_line[0] == '':
                        splited_line[0] = '{}_{}'.format(self.global_constants.empty_event_label, file)
                    forthcoming_event = splited_line[0]
                    agenda.append(forthcoming_event)
                    # if not self.is_flesh_event(forthcoming_event):
                    #     agenda_context.append(forthcoming_event)
                    indices.append(len(text_context))
                    target = [self.global_constants.begin_of_sequence] + splited_line[1].split() + \
                             [self.global_constants.end_of_sequence]
                    instance_buffer.append(
                        instance_tuple(event_context.copy(), text_context.copy(), indices.copy(), forthcoming_event,
                                       target, len(agenda) - 1, previous_regular_event))
                    ''' prepare for the next instance '''
                    text_context.extend(target)
                    event_context.extend([forthcoming_event] * len(target))
                    if not self.is_flesh_event(forthcoming_event):
                        previous_regular_event = forthcoming_event
                else:
                    ''' end of story line. reset buffers '''
                    for buffed_instance in instance_buffer:
                        index = min(buffed_instance.e_f_index + 1, len(agenda) - 1)
                        if index < len(agenda) - 1:
                            while self.is_flesh_event(agenda[index]):
                                index += 1
                                if index == len(agenda) - 1:
                                    break
                        yield self.text_to_instance(
                            agenda=agenda,
                            event_context=buffed_instance.event_context,
                            text_context=buffed_instance.text_context,
                            indices=buffed_instance.indices.copy(),
                            forthcoming_event=buffed_instance.forthcoming_event,
                            target=buffed_instance.target,
                            e_f_index=buffed_instance.e_f_index,
                            prev_reg_event=previous_regular_event,
                            succ_reg_event=agenda[index])
                    text_context = ['{}_{}'.format(self.global_constants.begin_of_story, file)]
                    event_context = ['{}_{}'.format(self.global_constants.beginning_event, file)]
                    # agenda_context = ['{}_{}'.format(self.global_constants.beginning_event, file)]
                    indices = list()
                    agenda = event_context.copy()
                    instance_buffer = list()
                    previous_regular_event = '{}_{}'.format(self.global_constants.beginning_event, file)
            infile.close()


@DatasetReader.register('segment_for_language_model')
class LanguageModelSegmentReader(DatasetReader):
    """
    Reads in entire stories to train a language model.

    must implement two functions
        1. text_to_instance
            turns data into an allnlp.data.Instance object and return it.
            Construction of any data instance should be done with it.
        2. _read
            define a way to read the data files as an iterator over Instances
    """

    def __init__(self, global_constants):
        super().__init__(lazy=False)
        self.token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self.global_constants = global_constants

    def text_to_instance(self,
                         tokens: List[str],) -> Instance:
        text_field = TextField([Token(t) for t in tokens], self.token_indexers)
        return Instance({'source': text_field})

    def _read(self, file_path) -> Iterator[Instance]:
        """"""
        files = os.listdir(file_path)
        ''' process each files in the folder '''
        for file in files:
            path = os.path.join(file_path, file)
            text_context = ['{}_{}'.format(self.global_constants.begin_of_story, file)]
            with open(path, 'r') as fin:
                line = fin.readline().lower()
                while line != '':
                    splited_line = line.split('\t')
                    if not splited_line[0].__contains__('<end_of_story>'):
                        ''' normal line. append. we add begin and end of sequence tokens to be compatible with the 
                        seq2seq model'''
                        text_context += [self.global_constants.begin_of_sequence] \
                            + splited_line[1].split() + [self.global_constants.end_of_sequence]
                    else:
                        ''' end of story line. yield and reset buffer. '''
                        yield self.text_to_instance(tokens=text_context)
                        text_context = ['{}_{}'.format(self.global_constants.begin_of_story, file)]
                    line = fin.readline()


class AlternatingSeq2seqIterator(DataIterator):
    """
    Alternating Seq2seq needs customized data iterator, as different instances use different components of the model,
    i.e. the 'is_flesh_event' fields must agree in each batch. So we group instances into batches by their
    'is_flesh_event' fields.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            # group the instance into two groups, each of identical 'is_flesh_event' fields
            flesh_iterator = \
                (instance for instance in instance_list if instance.fields['is_flesh_event'].metadata is True)
            sketch_iterator = \
                (instance for instance in instance_list if instance.fields['is_flesh_event'].metadata is False)

            # break each memory-sized list into batches and mix the batches into one list so we could shuffle it
            batches = list()
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(flesh_iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(sketch_iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))
            # shuffle
            if shuffle:
                random.shuffle(batches)
            # generate
            for batch in batches:
                yield batch


def data_split(path=os.path.join('.', 'segments20200517'),
               train_path=os.path.join('.', 'data_seg_train'),
               val_path=os.path.join('.', 'data_seg_val')):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        trainfilepath, valfilepath = os.path.join(train_path, file), os.path.join(val_path, file)
        with open(filepath, 'r') as fin:
            train_out, val_out = open(trainfilepath, 'w'), open(valfilepath, 'w')
            writer = train_out
            counter = 0
            inline = fin.readline()
            while inline != '':
                if inline.__contains__('<end_of_story>'):
                    writer.write(inline)
                    counter += 1
                    if counter % 30 == 0:
                        writer = val_out
                    else:
                        writer = train_out
                    inline = fin.readline()
                else:
                    writer.write(inline)
                    inline = fin.readline()
            train_out.close(), val_out.close()


if __name__ == '__main__':
    # """
    # TEST
    # This snippet should read the data properly.
    # """
    # train_path_ = os.path.join('.', 'data_seg_train')
    # val_path_ = os.path.join('.', 'data_seg_val')
    # reader = IrregularGenerationReader(global_constants=GLOBAL_CONSTANTS)
    # inses = list(reader.read(train_path_)) + list(reader.read(val_path_))
    # pass
    data_split()
