from typing import Dict, List, Tuple

import numpy
import copy
import os
import json

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

import dill

# allenNLP stuff. sorted.
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.language_model import LanguageModel
from allennlp.models.model import Model
from allennlp.modules.attention import DotProductAttention
from allennlp.modules import Attention
# TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
# from allennlp.nn.beam_search import BeamSearch
from beam_search_ import BeamSearch
from allennlp.nn.regularizers import RegularizerApplicator, L2Regularizer
from allennlp.predictors import Predictor
from allennlp.training.callbacks import Callback, handle_event, Events
from allennlp.training.metrics import CategoricalAccuracy

from globals import GLOBAL_CONSTANTS
from AgendaGenerator import Agenda
from dataset_readers import IrregularGenerationReader, LanguageModelSegmentReader
from utils import Struct, MaskedPositionalEncoding, MaskedProgressEncoding, preserve


class AlternatingSeq2Seq(Model):
    """

    """

    def __init__(self,
                 args_hpo,
                 vocab: Vocabulary,
                 dataset_reader,
                 configurations,
                 attention: Attention = DotProductAttention()
                 ) -> None:
        """
        :param args_hpo:
        :param vocab:
        :param dataset_reader:
        :param configurations:
        :param attention:
        """
        regularizer = RegularizerApplicator([('l2', L2Regularizer(alpha=args_hpo.l2))])
        super(AlternatingSeq2Seq, self).__init__(vocab, regularizer)
        self.args_hpo = args_hpo
        self.configurations = configurations
        self.device = configurations.DEVICE
        self.dataset_reader = dataset_reader
        ''' I swear this magic is constant during the entire project '''
        self._target_namespace = 'words'

        ''' for tracking att '''
        self.track_attention_mode = None
        self.attention_log = None
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(GLOBAL_CONSTANTS.begin_of_sequence, self._target_namespace)
        self._end_index = self.vocab.get_token_index(GLOBAL_CONSTANTS.end_of_sequence, self._target_namespace)

        # we tie event_embedding_size up with en_ncoder_size, and fix word embedding size at 300
        args_hpo['word_embedding_size'] = 300
        # as we are using bi-directional encoders, this allows an attention over the agenda which uses decoder hidden as
        # query
        args_hpo['event_embedding_size'] = args_hpo.ed_ncoder_size * 2

        self.decoder_sizes = {'previous_token': args_hpo['word_embedding_size'],
                              'forthcoming_event': args_hpo['event_embedding_size'],
                              'contextualized_event': args_hpo['ed_ncoder_size'] * 2,  # bi-directional
                              'encoder_att': args_hpo['ed_ncoder_size'] * 2,  # bi-directional
                              'agenda_att': args_hpo['event_embedding_size'],
                              'previous_regular_event': args_hpo['event_embedding_size'],
                              'succesive_regular_event': args_hpo['event_embedding_size']
                              }
        ''' 
        the class params 'pop' its parameters i.e. they disappear after first use. So we instantiate a Params 
        instance for each model defining execution. More than that, they turn dicts into Mutable mappings and 
        destroys the original dict. So here's your copy allennlp. Thanks. (I still love you)
        '''
        print('--- LOADING GLOVE ---')
        token_embedding = Embedding.from_params(
            vocab=vocab,
            params=Params(copy.deepcopy(configurations.GLOVE_PARAMS_CONFIG)))

        event_embedding = Embedding(num_embeddings=vocab.get_vocab_size(namespace='events'),
                                    embedding_dim=args_hpo.event_embedding_size)
        ''' embedding of TextFields must be done through a TextFieldEmbedder instead of an Embedding module '''
        token_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({'words': token_embedding})
        event_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({'events': event_embedding})

        ''' define encoder to wrap up an lstm feature extractor '''
        flesh_encoder: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(input_size=args_hpo.word_embedding_size + args_hpo.event_embedding_size
                          if self.configurations.encoder_variant == 'event' else args_hpo.word_embedding_size,
                          hidden_size=args_hpo.ed_ncoder_size,
                          bidirectional=True, batch_first=True,
                          num_layers=args_hpo.num_hidden_layeres,
                          dropout=args_hpo.dropout),
        )
        if configurations.SHARE_ENCODER:
            sketch_encoder = flesh_encoder
        else:
            sketch_encoder: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
                torch.nn.LSTM(input_size=args_hpo.word_embedding_size + args_hpo.event_embedding_size
                              if self.configurations.encoder_variant == 'event' else args_hpo.word_embedding_size,
                              hidden_size=args_hpo.ed_ncoder_size,
                              bidirectional=True, batch_first=True,
                              num_layers=args_hpo.num_hidden_layeres,
                              dropout=args_hpo.dropout))

        self._percent_student = args_hpo.percent_student
        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._max_decoding_steps = self.configurations.MAX_DECODING_LENGTH
        self.beam_size = self.configurations.BEAM_SIZE
        self._beam_search = BeamSearch(self._end_index, max_steps=self._max_decoding_steps, beam_size=self.beam_size)

        self._token_embedder, self._event_embedder = token_embedder, event_embedder
        self._sketch_encoder = flesh_encoder
        self._flesh_encoder = flesh_encoder

        self.dropout = torch.nn.Dropout(args_hpo.dropout)

        num_output_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = args_hpo.word_embedding_size
        self._target_embedder = Embedding(num_output_classes, target_embedding_dim)
        self._attention = attention

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._sketch_encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        '''---------------------- Decoder ------------------------'''
        self._sketch_decoder_input_dim = sum([self.decoder_sizes[component] for component in
                                              configurations.extra_decoder_components
                                              + configurations.default_decoder_components if component in
                                              ['previous_token', 'encoder_att',
                                               'forthcoming_event', 'contextualized_event']])
        self._flesh_decoder_input_dim = sum([self.decoder_sizes[component] for component in
                                             configurations.extra_decoder_components
                                             + configurations.default_decoder_components])
        if 'contextualized_event' in configurations.extra_decoder_components + configurations.final_layer_components:
            self._event_encoder: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
                torch.nn.LSTM(input_size=args_hpo.event_embedding_size,
                              hidden_size=args_hpo.ed_ncoder_size,
                              bidirectional=True, batch_first=True,
                              num_layers=1)
            )

        if self.configurations.positional_encoding in ['transformer', 'transformer_forth', 'transformer_context']:
            self.positional_encoder = MaskedPositionalEncoding(input_dim=self.args_hpo.event_embedding_size,
                                                               max_len=GLOBAL_CONSTANTS.maximum_agenda_length,
                                                               device=self.device)
        elif self.configurations.positional_encoding == 'progress':
            # max_progress_len is only applied to progress encodings
            self.positional_encoder = MaskedProgressEncoding(input_dim=self.args_hpo.event_embedding_size,
                                                             max_len=args_hpo.max_progress_len,
                                                             device=self.device)
        elif self.configurations.positional_encoding != 'none':
            raise Exception('Invalid positional encoding type specified. Must be among {\'transformer\', \'progress\' '
                            'and \'none\' but got {}}'.format(self.configurations.positional_encoding))

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._sketch_decoder_cell = LSTMCell(self._sketch_decoder_input_dim, self._decoder_output_dim)
        self._flesh_decoder_cell = LSTMCell(self._flesh_decoder_input_dim, self._decoder_output_dim)

        self._sketch_decoder_cell = self._flesh_decoder_cell

        self._effective_encoder = None
        self._effective_decoder_cell = None

        '''------------------------- Final output layer ---------------------------'''
        self._extra_sketch_projection_input_dim = sum([self.decoder_sizes[component] for component in
                                                       configurations.final_layer_components if component in
                                                       ['forthcoming_event', 'contextualized_event',
                                                        'previous_regular_event', 'succesive_regular_event']])
        self._extra_flesh_projection_input_dim = sum([self.decoder_sizes[component] for component in
                                                      configurations.final_layer_components])
        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = None
        self._sketch_output_projection_layer = Linear(self._decoder_output_dim + self._extra_sketch_projection_input_dim
                                                      , num_output_classes)
        self._flesh_output_projection_layer = Linear(self._decoder_output_dim + self._extra_flesh_projection_input_dim,
                                                     num_output_classes)
        '''  metrics to track  '''
        self.accuracy = CategoricalAccuracy()
        self.to(configurations.DEVICE)

    def forward(self,
                agenda: Dict[str, torch.Tensor],
                event_context: Dict[str, torch.Tensor],
                text_context: Dict[str, torch.Tensor],
                indices: List[List[int]],
                e_f_index: List[int],
                forthcoming_event: Dict[str, torch.Tensor],
                is_flesh_event: List[bool],
                previous_regular_event: Dict[str, torch.Tensor] = None,
                succesive_regular_event: Dict[str, torch.Tensor] = None,
                target: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        note: the inputs come in batches, so the metadata fields are to be lists, instead of their original type.
        the forward method takes a specific input and evaluates the ENTIRE output sequence. It consists of a few steps:
            0. the effective encoder / decoder components are set according to the target event type
            1. encoding
                performed by self._encode
                encodes the input sequences one by one, with their respective event contexts.
            2. decoding
                a.  during training, takes the one-best output recursively. could either use teacher-forcing or sample
                    from the model generations. performed by self._forward_loop
                b.  during inference, employs a beam search. performed by self._forward_beam
            3. loss
                The token average over the entire sequence.

        :param event_context:
        :param text_context:
        :param indices:
        :param forthcoming_event:
        :param target:
        :return: output_dict
            1. training 'predictions' and 'loss'
            2. inference 'predictions' and 'class_log_probabilities'
        """
        if is_flesh_event[0] is True:
            self._effective_encoder = self._flesh_encoder
            self._effective_decoder_cell = self._flesh_decoder_cell
            self._output_projection_layer = self._flesh_output_projection_layer
        else:
            self._effective_encoder = self._sketch_encoder
            self._effective_decoder_cell = self._sketch_decoder_cell
            self._output_projection_layer = self._sketch_output_projection_layer

        ''' use only the flesh encoder if in baseline mode '''
        if self.configurations.is_seq2seq_baseline:
            self._effective_encoder = self._flesh_encoder
            self._effective_decoder_cell = self._flesh_decoder_cell

        if self.track_attention_mode is True:
            # for <bose> tokens as query
            self.attention_log["0"]['attentions'][0]['value'].append([0.088])
            if 'agenda_att' in self.configurations.extra_decoder_components:
                # for <bose> tokens as query
                self.attention_log["1"]['attentions'][0]['value'].append([0.088])

        encoder_outputs = self._encode(text_context=text_context, event_context=event_context,
                                       agenda=agenda, indices=indices, e_f_index=e_f_index)
        encoder_outputs['forth_coming_event'] = forthcoming_event['events']
        if previous_regular_event:
            encoder_outputs['previous_regular_event'] = previous_regular_event['events']
        if succesive_regular_event:
            encoder_outputs['succesive_regular_event'] = succesive_regular_event['events']
        state = self._init_decoder_state(encoder_outputs)
        if target:
            ''' training and validation '''
            # The `_forward_loop` decodes the input sequence and computes the loss during training and validation.
            output_dict = self._forward_loop(state, target)
        else:
            ''' inference '''
            output_dict = self._forward_beam_search(state)

        return output_dict

    def beam_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict) -> Tuple:
        """
        with the current state, evaluate the state after taking the step in last_predictions and the log probabilities
        of possible successive steps.
        This is called by the beam search class.
        if GlobalConstants.USE_MMI_SCORING is True, include the MMI term.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        last_step_mmi_score = 0.
        if self.configurations.USE_MMI_SCORE:
            ''' evaluate MMI scores according to the seq2seq vocab '''
            last_step_mmi_score = self.evaluate_mmi_probability(last_predictions, state)
            ''' apply the correct coefficient '''
            if self._effective_encoder == self._flesh_encoder:
                ''' irregular generation '''
                last_step_mmi_score = self.configurations.IRREGULAR_MMI_COEFFICIENT * last_step_mmi_score
            elif self._effective_encoder == self._sketch_encoder:
                ''' regular generation '''
                last_step_mmi_score = self.configurations.REGULAR_MMI_COEFFICIENT * last_step_mmi_score
            else:
                raise Exception('what the hell? Check your encoder!')

        # shape: (group_size, num_classes = vocabsize)
        output_projections, state = self._decode_step(last_predictions, state)

        # shape: (group_size, num_classes = vocabsize)
        ''' logits from LM are re-indexed to fit the seq2seq vocabulary '''
        class_log_probabilities = F.log_softmax(output_projections, dim=-1) - last_step_mmi_score

        state['text_context'] = torch.cat(
            [state['text_context'],
             last_predictions.unsqueeze(1) if len(last_predictions.size()) == 1 else last_predictions],
            dim=-1)

        text = list()
        for idx in range(state['text_context'].size()[0]):
            text_context_list = list(state['text_context'][idx])
            text_context_tokens = \
                [self.vocab.get_token_from_index(int(i), namespace='words') for i in text_context_list]
            text_context_text = ' '.join(text_context_tokens[state['encoder_outputs'].size()[1]:])
            text.append(text_context_text)

        return class_log_probabilities, state

    def decode(self, output_dict: Dict):
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at TEST
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        # loop over the batch
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace) for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def evaluate_mmi_probability(self, last_predictions, state):
        """
        evaluates the MMI term of the probabilities of possible next steps, after steps in last_predictions taken at
        <state>.
        :param state:
        :return:
        """
        group_size, group_length = state['text_context'].size()
        step_log_softmax = state['encoder_outputs'].new_zeros(size=(group_size, self.vocab.get_vocab_size('words')))
        ''' we return a 0 vector if the segment in question is '<bost>, <bose>'. when this happens, we have <bost>
        in text_context and <bose> in last_predictions, so group_size is 1. '''
        if group_length == 1:
            return step_log_softmax
        else:
            # text_context
            for idx in range(group_size):
                text_context_list = list(state['text_context'][idx])
                text_context = [self.vocab.get_token_from_index(int(i), namespace='words') for i in text_context_list]
                text_extended = text_context + [self.vocab.get_token_from_index(int(last_predictions[idx]), 'words')]
                extended_instance = self.language_model_reader.text_to_instance(tokens=text_extended)
                # shape: [lm_encoder_dim]
                lm_embeddings = self.language_model_predictor.predict_instance(extended_instance)['lm_embeddings']
                logits = torch.tensor(lm_embeddings, dtype=torch.float)[-1, :].cuda(self.device)
                with torch.no_grad():
                    choice_losses = torch.nn.functional.log_softmax(
                        torch.matmul(logits, self.language_model._softmax_loss.softmax_w) +
                        self.language_model._softmax_loss.softmax_b, dim=-1)
                step_log_softmax[idx, :] = choice_losses
                # [:46] for toy testing

            ''' map the losses in the LM vocab back to the seq2seq vocab '''
            step_log_softmax = step_log_softmax[:, self.lm2seq_permutation]

            return step_log_softmax

    def generate_samples(self, script_s, include_irregular_events):
        dataset_reader = IrregularGenerationReader(global_constants=GLOBAL_CONSTANTS)
        results = list()
        for i, script in enumerate(script_s):
            agenda = Agenda.generate_agenda(script=script, with_irregular_events=include_irregular_events[i])
            text_context, event_context, generated_seqs = self.inference(dataset_reader=dataset_reader,
                                                                         agenda=agenda,
                                                                         scenario=script)
            results.append({'agenda': agenda, 'sequences': generated_seqs})
        return results

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        optional, for monitoring metrics so early-stopping etc could be used.
        return a dict of the metrics in numbers
        """
        return {"accuracy": self.accuracy.get_metric(reset)}

    def inference(self, dataset_reader, agenda: List[str], scenario: str):
        predictor = Predictor(model=self, dataset_reader=dataset_reader)
        text_context = ['{}_{}'.format(GLOBAL_CONSTANTS.begin_of_story, scenario)]
        event_context = ['{}_{}'.format(GLOBAL_CONSTANTS.beginning_event, scenario)]
        previous_regular_event = '{}_{}'.format(GLOBAL_CONSTANTS.beginning_event, scenario)
        indices = [1]
        generations = list()

        # skip the last event, end of story, in the agenda because it does not require text generation.
        for event_pointer, forthcoming_event in enumerate(agenda[:-1]):
            forthcoming_event = agenda[event_pointer]
            index = min(len(generations) + 1, len(agenda) - 1)
            if index < len(agenda) - 1:
                while dataset_reader.is_flesh_event(agenda[index]):
                    index += 1
                    if index == len(agenda) - 1:
                        break
            #  fixme: make sure agenda does not include bost
            state_instance = dataset_reader.text_to_instance(
                event_context=event_context,
                text_context=text_context,
                indices=indices,
                forthcoming_event=forthcoming_event,
                e_f_index=len(generations),
                agenda=agenda,
                prev_reg_event=previous_regular_event,
                succ_reg_event=agenda[index])
            output_dict = predictor.predict_instance(state_instance)
            sequence = output_dict['predicted_tokens']
            text_context.extend(sequence)
            event_context.extend([agenda[event_pointer]] * len(sequence))
            if not dataset_reader.is_flesh_event(forthcoming_event):
                previous_regular_event = forthcoming_event
            if len(sequence) > 0:
                indices.append(indices[-1] + len(sequence))
            generations.append(sequence)
        return text_context, event_context, generations

    def load_language_model(self, vocab_folder, model_path, aug_mode=False):
        ''' somehow the combination file is corrupted. we rebuild it by hand. '''
        combination = Struct({
            'dropout': 0.2,
            'ed_ncoder_size': 512,
            'word_embedding_size': 300,
            'l2': 0.2
        }) if not aug_mode else Struct({
            'dropout': 0.2,
            'ed_ncoder_size': 1024,
            'word_embedding_size': 300,
            'l2': 0.2
        })

        vocabulary = Vocabulary.from_files(vocab_folder)
        ''' the language model used Glove but we just build an embedder to load the trained parameters '''
        token_embedding = Embedding(num_embeddings=vocabulary.get_vocab_size(namespace='tokens'),
                                    embedding_dim=combination.word_embedding_size, padding_index=0)
        token_embedder: TextFieldEmbedder = BasicTextFieldEmbedder({'tokens': token_embedding})
        ''' define encoder to wrap up an lstm feature extractor '''
        contextualizer: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(input_size=combination.word_embedding_size,
                          hidden_size=combination.ed_ncoder_size,
                          bidirectional=False, batch_first=True))
        model = LanguageModel(vocab=vocabulary,
                              text_field_embedder=token_embedder,
                              contextualizer=contextualizer,
                              dropout=combination.dropout,
                              regularizer=RegularizerApplicator([('l2', L2Regularizer(alpha=combination.l2))]),
                              ) \
            .cuda(self.device)
        model.load_state_dict(torch.load(open(model_path, 'rb')), strict=True)
        self.language_model = model
        self.language_model_predictor = Predictor(
            model=model, dataset_reader=LanguageModelSegmentReader(global_constants=GLOBAL_CONSTANTS))
        self.language_model_reader = LanguageModelSegmentReader(global_constants=GLOBAL_CONSTANTS)

        ''' construct a permutation of the language model vocabulary to map it to the seq2seq vocabulary '''
        vocab_size = self.vocab.get_vocab_size('words')
        self.lm2seq_permutation = torch.zeros(size=[vocab_size], dtype=torch.int64, device=self.device)
        for index in range(vocab_size):
            ''' perm[i] is the index of the verb in seq2seq vocab that is indexed i in the lm vocab '''
            self.lm2seq_permutation[index] = self.vocab.get_token_index(
                self.language_model.vocab.get_token_from_index(index, 'tokens'), 'words')

    def prepare_embeddings_for_tensorboard(self):
        word_embedding_data = dict()
        word_embedding_data['mat'] = self._token_embedder._token_embedders['words'].weight
        word_embedding_data['tag'] = 'words'
        word_embedding_data['metadata'] = \
            [self.vocab.get_token_from_index(i, 'words') for i in range(self.vocab.get_vocab_size('words'))]

        event_embedding_data = dict()
        event_embedding_data['mat'] = self._event_embedder._token_embedders['events'].weight
        event_embedding_data['tag'] = '_events'
        event_embedding_data['metadata'] = \
            [self.vocab.get_token_from_index(i, 'events') for i in range(self.vocab.get_vocab_size('events'))]

        return [word_embedding_data, event_embedding_data]

    def track_attention(self, input_folder):
        """
        track the attention values of the generation of a story in input_file, which should have the same format as the
        training data. output a json file to be read by the java attention visualizer.

        we log 2 attentions.
            1. attention over encoder states, recorded in example '0'
            2. attention over the agenda, recorded in example '1', if the model has this attention
        note: currently, each input file should contain only one story.
        :param input_folder:
        :return:
        """
        output_file = os.path.join(*['.', 'decoder', 'bath_2_3.attention'])
        self.track_attention_mode = True
        self._percent_student = 0.
        instance_s = self.dataset_reader.read(input_folder)
        story = ' '.join([token.text for token in instance_s[-1].fields['text_context'].tokens
                          + instance_s[-1].fields['target'].tokens][:-1])
        self.attention_log = dict()

        self.attention_log = {"0": {
            "source": story,  # the source sentence (without <s> and </s> symbols)
            "translation": story,  # the target sentence (without <s> and </s> symbols)
            "attentions":
                [  # various attention results
                    {
                        "type": "simple",  # the type of this attention (simple or multihead)
                        "name": "encoder_decoder_attention",  # a unique name for this attention
                        # the attention weights, a json array. add initial value as '<bost>' is not generated from a
                        # decoding step
                        "value": [
                            ([0.099] * (len(story.split()) + 1)).copy(),  # for <bost> as query
                        ]
                        # note: Mij: attention of i-th target token to j-th source token
                        #       size = [len(target) + 1] * [len(source) + 1], for EOS.
                    },  # end of one attention result
                ]  # end of various attention results
        }}

        if 'agenda_att' in self.configurations.extra_decoder_components:
            agenda_original = [token.text for token in instance_s[-1].fields['agenda'].tokens]
            agenda_simplified = [event[6:-5] if event[0] == 's' else event[:9] for event in agenda_original]
            agenda = ' '.join(agenda_simplified)
            self.attention_log["1"] = {
                "source": agenda,  # the source sentence (without <s> and </s> symbols)
                "translation": story,  # the target sentence (without <s> and </s> symbols)
                "attentions":
                    [  # various attention results
                        {
                            "type": "simple",  # the type of this attention (simple or multihead)
                            "name": "encoder_decoder_attention",  # a unique name for this attention
                            # the attention weights, a json array. add initial value as '<bost>' is not generated from a
                            # decoding step
                            "value": [
                                ([0.099] * (len(agenda.split()) + 1)).copy(),  # for <bost> as query
                            ]
                            # note: Mij: attention of i-th target token to j-th source token
                            #       size = [len(target i.e. query) + 1] * [len(source) + 1], as the visualizer
                            #       automatically adds <EOS> to both source and translation.
                        },  # end of one attention result
                    ]  # end of various attention results
            }

        predictor = Predictor(model=self, dataset_reader=self.dataset_reader)
        for instance in instance_s:
            predictor.predict_instance(instance)
        for example in self.attention_log.values():
            length = len(example['source'].split()) + 1
            for attention in example['attentions']:
                for attention_list in attention['value']:
                    if len(attention_list) < length:
                        attention_list += [0.0] * (length - len(attention_list))
        json.dump(self.attention_log, open(output_file, 'w'))

    def _shrink_event_context(self, embedded_event_context: Dict[str, torch.Tensor], indices: List[List[int]]) \
            -> torch.Tensor:
        """
        shrink the event_context to be used by the positional encoding module
        :param embedded_event_context: shape: batch_size=1 * len * event_dim
        :param indices:
        :return:
        """
        shrinked_event_context = embedded_event_context.new_zeros(
            size=[1, len(indices[0]), self.args_hpo.event_embedding_size])
        for shrink_index, agenda_index in enumerate(indices[0]):
            # indices stores indices for slicing, so we need to use agenda_index - 1
            shrinked_event_context[0, shrink_index, :] = embedded_event_context[0, agenda_index - 1, :]
        return shrinked_event_context

    def _expand_event_context(self, shrinked_event_context: torch.Tensor, indices: List[List[int]]) -> torch.Tensor:
        """
        expand the shrinked_event_context to be used by the encoder
        :param shrinked_event_context:
        :param indices:
        :return:
        """
        length = indices[0][-1]
        expanded_event_context = shrinked_event_context.new_zeros(size=[1, length, self.args_hpo.event_embedding_size])
        pointer = 0
        for shrink_index, agenda_index in enumerate(indices[0]):
            expanded_event_context[0, pointer: agenda_index, :] = shrinked_event_context[0, shrink_index, :].expand(
                size=[1, agenda_index - pointer, self.args_hpo.event_embedding_size])
        return expanded_event_context

    def _prepare_positional_positions_and_mask(self,
                                               agenda: Dict[str, torch.Tensor],
                                               event_context: Dict[str, torch.Tensor],
                                               context_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        given agenda and event context, generate positions according to which positional encoding should be added.
        applies to both encoding and generation
        :param agenda: the indexed agendas
            {'events': shape batch * max_agenda_len}
        :param event_context: the indexed event context
            shape: batch * max_len_in_batch
        :param context_mask: the mask indicating whether each item in event_context is valid
        :return:
            positions: shape batch * max_len_in_batch
            event_mask:
                shape batch * max_len_in_batch
                    1: the event is IRREGULAR
                    0: the event is regular
        """
        event_context_indices, agenda_indices = event_context['events'], agenda['events']
        positions = event_context_indices.new_zeros(size=event_context_indices.size())
        event_mask = event_context_indices.new_ones(size=event_context_indices.size())
        for b in range(event_context_indices.size(0)):
            agenda_pointer = 0
            agenda_list = list(agenda_indices[b].cpu().numpy())
            for i in range(event_context_indices.size(1)):
                try:
                    agenda_pointer = agenda_list.index(int(event_context_indices[b, i]), agenda_pointer)
                except ValueError:
                    agenda_pointer = 0
                # for the story_begin event, which is not in any agenda
                positions[b, i] = agenda_pointer
                if self.configurations.mask_regular_events_for_pe:
                    event_mask[b, i] = int(self.dataset_reader.
                                           is_flesh_event(self.vocab.get_token_from_index(agenda_pointer, 'events')))
        return positions * context_mask, event_mask * context_mask

    def _encode(self,
                text_context: Dict[str, torch.Tensor],
                event_context: Dict[str, torch.Tensor],
                agenda: Dict[str, torch.Tensor],
                e_f_index: List[int],
                indices: List[List[int]]) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, len)
        source_mask = util.get_text_field_mask(text_context)
        # shape: (batch_size, len, token_embedding_size)
        embedded_text_context = self._token_embedder(text_context)
        embedded_text_context_dropped = self.dropout(embedded_text_context)
        # shape: (batch_size, len, event_embedding_size)
        embedded_event_context = self._event_embedder(event_context)
        embedded_event_context_dropped = self.dropout(embedded_event_context)
        # apply transformer's weight adjustment
        if self.configurations.multiply_emb:
            embedded_event_context_dropped = embedded_event_context * \
                                             torch.sqrt(
                                                 torch.tensor(self.args_hpo.event_embedding_size, device=self.device,
                                                              dtype=torch.float))
        context_events = [[self.vocab.get_token_from_index(int(agenda['events'][b][i]), namespace='events')
                           for i in range(len(indices[b]))] for b in range(agenda['events'].size()[0])]
        if self.configurations.positional_encoding in ['transformer', 'progress']:
            positions, event_mask = self._prepare_positional_positions_and_mask(agenda=agenda,
                                                                                event_context=event_context,
                                                                                context_mask=source_mask)
            positional_encoded_event_context = self.positional_encoder(
                x=embedded_event_context_dropped,
                source_mask=event_mask,
                positions=positions)
            # shape: (batch_size, len, token_embedding_size + event_embedding_size)
            encoder_input = torch.cat([embedded_text_context_dropped, positional_encoded_event_context], dim=-1) \
                if self.configurations.encoder_variant == 'event' else embedded_text_context_dropped
        else:
            # shape: (batch_size, len, token_embedding_size + event_embedding_size)
            encoder_input = torch.cat([embedded_text_context_dropped, embedded_event_context_dropped], dim=-1) \
                if self.configurations.encoder_variant == 'event' else embedded_text_context_dropped

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._effective_encoder(encoder_input, source_mask)

        state = {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
            'text_context': text_context['words'],
            'beam_log_probs': torch.zeros(size=text_context['words'].size(), dtype=torch.float, device=self.device),
            # the index of the forthcoming event in the agenda
            'position': torch.tensor([len(context_events[b]) for b in range(agenda['events'].size()[0])]
                                     ).unsqueeze(0).to(self.device).transpose(0, 1),
            'agenda_length': torch.tensor(len(agenda['events'][0])).unsqueeze(0).to(self.device),
            'agenda': agenda['events']
        }

        if 'contextualized_event' in self.configurations.extra_decoder_components \
                + self.configurations.final_layer_components:
            embedded_agenda_dropped = self.dropout(self._event_embedder(agenda))
            agenda_mask = util.get_text_field_mask(agenda)
            # shape: batch * len * (num_directions * event_emb_size)
            encoded_agenda_context = self._event_encoder(embedded_agenda_dropped, agenda_mask)
            # it seems advanced indexing needs torch.Longtensor, but let's see
            state['encoded_agenda_context'] = torch.cat([
                encoded_agenda_context[ind, e_f_index[ind], :].unsqueeze(0) for ind in range(len(e_f_index))], 0)
            # state['encoded_agenda_context'] = encoded_agenda_context[:, e_f_index, :]

        return state

    def _decode_step(self,
                     last_predictions: torch.Tensor,
                     state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        encoder_outputs_dropped = self.dropout(encoder_outputs)
        # shape: (group_size, max_input_sequence_length)
        # this is the mask for attention
        source_mask = state["source_mask"]
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]
        forth_coming_event_dict = {'events': state['forth_coming_event']}
        # shape: (group_size, event_emb_size)
        forth_coming_event_emb = self._event_embedder(forth_coming_event_dict).squeeze(1)
        forth_coming_event_emb_dropped = self.dropout(forth_coming_event_emb)

        # if we do not mask regular events or if we do but the forth coming event is irregular, add positional encodings
        # prepare positional mask: which events are irregular (thus needs positional encoding). As our iterator already
        # insures this is consistent within each batch, we only need to check once here
        positional_mask = forth_coming_event_emb.new_ones(size=[forth_coming_event_emb.size(0), 1])
        if self.configurations.mask_regular_events_for_pe and self._effective_encoder is self._sketch_encoder:
            positional_mask *= 0.
        if self.configurations.positional_encoding in ['transformer', 'transformer_forth']:
            forth_coming_event_emb_dropped = self.positional_encoder(x=forth_coming_event_emb_dropped.unsqueeze(1),
                                                                     positions=state['position'],
                                                                     source_mask=positional_mask).squeeze(1)
        # fixme: currently not working
        elif self.configurations.positional_encoding == 'progress':
            forth_coming_event_emb_dropped = self.positional_encoder(x=forth_coming_event_emb_dropped,
                                                                     position=state['position'][0],
                                                                     agenda_length=state['agenda_length'][0]).squeeze(1)

        ''' ----------------- assemble decoder inputs ----------------- '''
        # shape: (group_size, word_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)
        # apply attention over encoder states
        # shape: (group_size, encoder_output_dim)
        attended_input, input_weights = self._prepare_attended_input(decoder_hidden,
                                                                     encoder_outputs_dropped,
                                                                     source_mask)
        ''' ----------------- log attention if requested ------------------ '''
        if self.track_attention_mode is True:
            att_array = input_weights.squeeze(0).detach().cpu().numpy().tolist()
            self.attention_log["0"]["attentions"][0]["value"].append([preserve(f) for f in att_array])

        ''' ----------------- assemble decoder components ------------------- '''
        # shape: (group_size, decoder_output_dim + target_embedding_dim)
        decoder_input = torch.cat((attended_input, embedded_input), -1)
        # apply attention over the agenda. only used for irregular events.
        if 'agenda_att' in self.configurations.extra_decoder_components:
            # forward pass
            if self._effective_encoder is self._flesh_encoder:
                embedded_agenda = self._event_embedder({'events': state['agenda']})
                # we use the hidden state of decoder to query the agenda items without pe.
                # all agenda items will be available, but regular events should not access agenda attention.
                # trivial attention masks can be used, as pad tokens receives 0 embeddings
                attended_agenda_input, input_weights = \
                    self._prepare_attended_input(decoder_hidden, embedded_agenda,
                                                 state['agenda'].new_ones(state['agenda'].size()))
                decoder_input = torch.cat([attended_agenda_input, decoder_input], -1)
                # logging attention
                if self.track_attention_mode is True:
                    att_array = input_weights.squeeze(0).detach().cpu().numpy().tolist()
                    self.attention_log["1"]["attentions"][0]["value"].append([preserve(f) for f in att_array])
            else:
                # decoder_input = torch.cat([forth_coming_event_emb_dropped, decoder_input], -1)
                # logging dummy attention
                if self.track_attention_mode is True:
                    self.attention_log["1"]["attentions"][0]["value"].append([0.077])

        # event encoder
        if 'contextualized_event' in self.configurations.extra_decoder_components:
            if self.configurations.positional_encoding == 'transformer_context':
                contextualized_event_with_pe = self.positional_encoder(x=state['encoded_agenda_context'].unsqueeze(1),
                                                                       positions=state['position'],
                                                                       source_mask=positional_mask).squeeze(1)
                decoder_input = torch.cat([contextualized_event_with_pe, decoder_input], -1)
            else:
                decoder_input = torch.cat([state['encoded_agenda_context'], decoder_input], -1)
        if 'forthcoming_event' in self.configurations.extra_decoder_components:
            # shape: (group_size, word_embedding_dim + event_embedding_dim)
            decoder_input = torch.cat([decoder_input, forth_coming_event_emb_dropped], dim=-1)
        if 'previous_regular_event' in self.configurations.extra_decoder_components \
                and self._effective_encoder is self._flesh_encoder:
            previous_regular_event_embedding = self._event_embedder({'events': state['previous_regular_event']}) \
                .squeeze(1)
            previous_regular_event_embedding_dropped = self.dropout(previous_regular_event_embedding)
            decoder_input = torch.cat([decoder_input, previous_regular_event_embedding_dropped], dim=-1)
        if 'succesive_regular_event' in self.configurations.extra_decoder_components \
                and self._effective_encoder is self._flesh_encoder:
            succesive_regular_event_embedding = self._event_embedder({'events': state['succesive_regular_event']}) \
                .squeeze(1)
            succesive_regular_event_embedding_dropped = self.dropout(succesive_regular_event_embedding)
            decoder_input = torch.cat([decoder_input, succesive_regular_event_embedding_dropped], dim=-1)

        ''' ----------------- decode ------------------- '''
        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._effective_decoder_cell(
            decoder_input,
            (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        ''' ----------------- assemble final layer components ------------------- '''
        final_input = decoder_hidden
        # # apply attention over the agenda. only used for irregular events.
        if 'agenda_att' in self.configurations.final_layer_components:
            # forward pass
            if self._effective_encoder is self._flesh_encoder:
                embedded_agenda = self._event_embedder({'events': state['agenda']})
                # we use the hidden state of decoder to query the agenda items without pe.
                # all agenda items will be available, but regular events should not access agenda attention.
                # trivial attention masks can be used, as pad tokens receives 0 embeddings
                attended_agenda_input, input_weights = \
                    self._prepare_attended_input(final_input, embedded_agenda,
                                                 state['agenda'].new_ones(state['agenda'].size()))
                final_input = torch.cat([attended_agenda_input, final_input], -1)
        # # event encoder
        if 'contextualized_event' in self.configurations.final_layer_components:
            if self.configurations.positional_encoding == 'transformer_context':
                contextualized_event_with_pe = self.positional_encoder(x=state['encoded_agenda_context'].unsqueeze(1),
                                                                       positions=state['position'],
                                                                       source_mask=positional_mask).squeeze(1)
                final_input = torch.cat([contextualized_event_with_pe, final_input], -1)
            else:
                final_input = torch.cat([state['encoded_agenda_context'], final_input], -1)
        if 'forthcoming_event' in self.configurations.final_layer_components:
            # shape: (group_size, word_embedding_dim + event_embedding_dim)
            final_input = torch.cat([final_input, forth_coming_event_emb_dropped], dim=-1)
        if 'previous_regular_event' in self.configurations.final_layer_components:  # \
            # and self._effective_encoder is self._flesh_encoder:
            previous_regular_event_embedding = self._event_embedder({'events': state['previous_regular_event']}) \
                .squeeze(1)
            previous_regular_event_embedding_dropped = self.dropout(previous_regular_event_embedding)
            final_input = torch.cat([final_input, previous_regular_event_embedding_dropped], dim=-1)
        if 'succesive_regular_event' in self.configurations.final_layer_components:  # \
            # and self._effective_encoder is self._flesh_encoder:
            succesive_regular_event_embedding = self._event_embedder({'events': state['succesive_regular_event']}) \
                .squeeze(1)
            succesive_regular_event_embedding_dropped = self.dropout(succesive_regular_event_embedding)
            final_input = torch.cat([final_input, succesive_regular_event_embedding_dropped], dim=-1)
        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(final_input)

        return output_projections, state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        performs sequence generation in training. Generates a sequence of the same length as the target.
        :param state: contains 'source_mask'
        :param target: contains both <bose> and <eose>
        :return:
        """
        # shape: (batch_size, len)
        source_mask = state["source_mask"]
        batch_size = source_mask.size()[0]

        # shape: (batch_size, len)
        target_tokens = target["words"]
        _, target_sequence_length = target_tokens.size()
        num_decoding_steps = target_sequence_length - 1

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        target_mask = util.get_text_field_mask(target)
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._percent_student:
                ''' use generation '''
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                ''' teacher-forcing '''
                # shape: (batch_size,)
                input_choices = target_tokens[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._decode_step(input_choices, state)
            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)
            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)
            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))

            ''' track accuracy. Note that we only evaluate for n-1 tokens in target '''
            self.accuracy(predictions=class_probabilities,
                          gold_labels=target['words'][:, timestep + 1],
                          mask=target_mask[:, timestep + 1])

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)
        output_dict = {"predictions": predictions}

        ''' get loss '''
        # shape: (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        # Compute loss.
        # .contiguous is necessary for .view() operations
        target_mask_for_loss = target_mask[:, 1:].contiguous()
        target_tokens = target_tokens[:, 1:].contiguous()
        loss = util.sequence_cross_entropy_with_logits(logits, target_tokens, target_mask_for_loss)
        output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.beam_step)

        ''' predictions include a 'batch size' dimension, which is always 0 during inference time, thus the shape is 
        different from the output of forward_loop. but decode() takes care of this. '''
        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs=state["encoder_outputs"],
            mask=state["source_mask"],
            bidirectional=self._effective_encoder.is_bidirectional())
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)
        return state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.Tensor = None,
                                encoder_outputs: torch.Tensor = None,
                                encoder_outputs_mask: torch.Tensor = None):
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
            decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input, input_weights

    @staticmethod
    def load_model(combination_file, index, vocab_path, model_path, configurations):
        combination_s = dill.load(open(combination_file, 'rb'))
        combination = combination_s[index]
        vocabulary = Vocabulary.from_files(vocab_path)
        model = AlternatingSeq2Seq(combination, vocabulary, configurations=configurations,
                                   dataset_reader=IrregularGenerationReader(global_constants=GLOBAL_CONSTANTS))
        ''' 
        because we use fields like 'effective_encoder', model.effective_encoder will be saved as parameters. However,
        a newly defined model does not need these parameters to be loaded, therefore the needed dict keys do not match 
        the saved dict keys. So we set strict=False.
        '''
        model.load_state_dict(torch.load(open(model_path, 'rb')), strict=False)
        return model

    @staticmethod
    def print_sample(agenda, sequences, print_to_terminal=False):
        log = ''
        for i, event in enumerate(agenda[:-1]):
            if print_to_terminal:
                print('event: {}'.format(event))
                print('sequence: {}'.format(' '.join(sequences[i])))
            log += '{}\t{}\n'.format(event, ' '.join(sequences[i]))
        log += '-----------------------------------------\n'
        return log

    class SampleGeneration(Callback):
        """
        performs sample generation in each epoch.
        at each epoch end, calls the model.inference to generate sequences according to the agenda.
        """

        def __init__(self, agenda_s, dataset_reader, scenario_s, index):
            self.agenda_s = agenda_s
            self.dataset_reader = dataset_reader
            self.scenario_s = scenario_s
            self.index = index
            self.cummulative_log = '===========Samples from training session {}=========\n'.format(index)
            super().__init__()

        @handle_event(Events.EPOCH_START)
        def turn_off_MMI(self, trainer):
            trainer.model.configurations.USE_MMI_SCORE = False

        @handle_event(Events.EPOCH_END)
        def generate_sample(self, trainer):
            print('+++++++++++++++++++++++++++++++++Sample Generation epoch {}+++++++++++++++++++++++++++++++++'.
                  format(trainer.epoch_number))
            self.cummulative_log += \
                '+++++++++++++++++++++++++++++++++Sample Generation epoch {}+++++++++++++++++++++++++++++++++\n'.\
                format(trainer.epoch_number)

            for ind, agenda in enumerate(self.agenda_s):
                text_context, event_context, generations = trainer.model.inference(
                    dataset_reader=self.dataset_reader,
                    agenda=agenda,
                    scenario=self.scenario_s[ind])
                self.cummulative_log += AlternatingSeq2Seq.print_sample(agenda=agenda, sequences=generations,
                                                                        print_to_terminal=True) \
                                        + '----------------------------\n'
            if trainer.epoch_number > 20:
                trainer.model.configurations.USE_MMI_SCORE = True
                trainer.model.configurations.BEAM_SIZE = 100
                # trainer.model.configurations.IRREGULAR_MMI_COEFFICIENT = 0.1
                # self.cummulative_log += \
                #     '-------------------------------Sample .1 MMI Generation epoch {}-----------------------------\n' \
                #         .format(trainer.epoch_number)
                # for ind, agenda in enumerate(self.agenda_s):
                #     if ind % 2 == 0:
                #         continue
                #     text_context, event_context, generations = trainer.model.inference(
                #         dataset_reader=self.dataset_reader,
                #         agenda=agenda,
                #         scenario=self.scenario_s[ind])
                #     self.cummulative_log += AlternatingSeq2Seq. \
                #                                 print_sample(agenda=agenda, sequences=generations,
                #                                              print_to_terminal=True) \
                #                             + '----------------------------\n'
                # trainer.model.configurations.IRREGULAR_MMI_COEFFICIENT = 0.08
                # self.cummulative_log += \
                #     '-------------------------------Sample .08 MMI Generation epoch {}-----------------------------\n' \
                #         .format(trainer.epoch_number)
                # for ind, agenda in enumerate(self.agenda_s):
                #     if ind % 2 == 0:
                #         continue
                #     text_context, event_context, generations = trainer.model.inference(
                #         dataset_reader=self.dataset_reader,
                #         agenda=agenda,
                #         scenario=self.scenario_s[ind])
                #     self.cummulative_log += AlternatingSeq2Seq. \
                #                                 print_sample(agenda=agenda, sequences=generations,
                #                                              print_to_terminal=True) \
                #                             + '----------------------------\n'
                trainer.model.configurations.IRREGULAR_MMI_COEFFICIENT = 0.06
                self.cummulative_log += \
                    '-------------------------------Sample .06 MMI Generation epoch {}-----------------------------\n' \
                        .format(trainer.epoch_number)
                for ind, agenda in enumerate(self.agenda_s):
                    if ind % 2 == 0:
                        continue
                    text_context, event_context, generations = trainer.model.inference(
                        dataset_reader=self.dataset_reader,
                        agenda=agenda,
                        scenario=self.scenario_s[ind])
                    self.cummulative_log += AlternatingSeq2Seq. \
                                                print_sample(agenda=agenda, sequences=generations,
                                                             print_to_terminal=True) \
                                            + '----------------------------\n'

            trainer.model.configurations.USE_MMI_SCORE = False
            trainer.model.configurations.BEAM_SIZE = 5

        @handle_event(Events.TRAINING_END)
        def save_samples(self, trainer):
            with open('sample_session_{}'.format(self.index), 'w') as _fout:
                _fout.write(self.cummulative_log)

    class TrackRegularAndIrregularLoss(Callback):
        """
        we use this callback to track regular and irregular losses. As we need more than predictions and golden labels
        to do so, a Metric class will not suffice. we hack the code a bit to integrate the metrics into
            trainer.train_metrics
        and
            trainer.val_metrics
        """

        def __init__(self, tensorboard_writer):
            super().__init__()
            self.tensorboard_writer = tensorboard_writer

        def reset(self):
            self.cumulated_irregular_loss = 0.0
            self.cumulated_regular_loss = 0.0
            self.irregular_batch_count = 0.0
            self.regular_batch_count = 0.0

        @handle_event(Events.EPOCH_START)
        def init_states(self, trainer):
            self.reset()

        @handle_event(Events.BATCH_END)
        def update_state(self, trainer):
            if trainer.model._effective_encoder is trainer.model._sketch_encoder:
                self.cumulated_regular_loss += trainer.train_metrics['loss']
                self.regular_batch_count += 1.
            else:
                self.cumulated_irregular_loss += trainer.train_metrics['loss']
                self.irregular_batch_count += 1.

        @handle_event(Events.VALIDATE, priority=-9999)
        def log_train_metric(self, trainer):
            if self.irregular_batch_count > 0:
                self.tensorboard_writer.add_scalar(
                    tag='train_irregular_loss',
                    scalar_value=self.cumulated_irregular_loss / self.irregular_batch_count,
                    global_step=trainer.epoch_number)
            if self.regular_batch_count > 0:
                self.tensorboard_writer.add_scalar(
                    tag='train_regular_loss',
                    scalar_value=self.cumulated_regular_loss / self.regular_batch_count,
                    global_step=trainer.epoch_number)
            self.reset()

        @handle_event(Events.VALIDATE, priority=9999)
        def log_val_metric(self, trainer):
            """
            log val metrics and initialize state. this should happen after the actual validation, so we set priority to
            minimum
            """
            if self.irregular_batch_count > 0:
                self.tensorboard_writer.add_scalar(
                    tag='val_irregular_loss',
                    scalar_value=self.cumulated_irregular_loss / self.irregular_batch_count,
                    global_step=trainer.epoch_number)
            if self.regular_batch_count > 0:
                self.tensorboard_writer.add_scalar(
                    tag='val_regular_loss',
                    scalar_value=self.cumulated_regular_loss / self.regular_batch_count,
                    global_step=trainer.epoch_number)
            self.reset()
