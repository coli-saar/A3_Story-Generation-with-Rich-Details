"""
An example showing how to use an LSTM trained on InScript
   2020.4.26 Fangzhou
Note:
    1. the dataset reader expects input file with event annotations (but only filter them out). Check
    ./data_seg_val_toytoy/grocery
"""
from globals import GLOBAL_CONSTANTS
from dataset_readers import LanguageModelSegmentReader
import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
import torch
from allennlp.models.language_model import LanguageModel
from allennlp.nn.regularizers import RegularizerApplicator, L2Regularizer
from allennlp.predictors import Predictor

from utils import Struct

language_model_path = os.path.join(*['.', 'language_model', 'models_lm_LM_opt_2020-02-10_2', 'best.th'])
language_model_vocab_path = os.path.join(*['.', 'language_model', 'vocab_lm_LM_opt_2020-02-10_2'])
language_model_combination_file = os.path.join(*['.', 'language_model', 'combs_LM_opt_2020-02-10'])
language_model_index = 2
combination = Struct({
    'dropout': 0.2,
    'ed_ncoder_size': 512,
    'word_embedding_size': 300,
    'l2': 0.2
})
device = 0

vocabulary = Vocabulary.from_files(language_model_vocab_path)
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
    .cuda(device)
model.load_state_dict(torch.load(open(language_model_path, 'rb')), strict=True)
dataset_reader = LanguageModelSegmentReader(global_constants=GLOBAL_CONSTANTS)
language_model_predictor = Predictor(model=model, dataset_reader=dataset_reader)
val_data_path = os.path.join('.', 'data_seg_val_toytoy')
instances = dataset_reader.read(val_data_path)
predictions = [language_model_predictor.predict_instance(instance) for instance in instances]


