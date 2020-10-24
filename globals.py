import os


# noinspection PyPep8Naming
class GLOBAL_CONSTANTS:
    """
    constants that does not change during the
            *****PROJECT*****
    for execution settings, see specific scripts
    """

    '''---------------------------------------------'''
    ''' preprocessing '''
    ''' special tokens to add per script '''
    beginning_event = '<story_begins>'
    empty_event_label = 'irregular'
    begin_of_story = '<bost>'
    ending_event = '<story_ends>'

    ''' other special tokens '''
    begin_of_sequence = '<bose>'
    end_of_sequence = '<eose>'
    end_of_story_label = '<end_of_story>'
    padding_token = '<pad>'

    ''' maximum agenda length for positional encoding. It is actually 51, but we set to 60 in case the agenda generator
    does something crazy. '''
    maximum_agenda_length = 60

    flesh_event_prefixes = ['unrelev_', 'relnscrev_', 'screv_other', 'unclear_', 'irregular']
    effective_script_s = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']

    language_model_path = os.path.join(*['.', 'language_model', 'models_lm_LM_opt_2020-02-10_2', 'best.th'])
    language_model_vocab_path = os.path.join(*['.', 'language_model', 'vocab_lm_LM_opt_2020-02-10_2'])
    language_model_combination_file = os.path.join(*['.', 'language_model', 'combs_LM_opt_2020-02-10'])
    language_model_index = 2

    aug_language_model_path = os.path.join(*['.', 'language_model', 'models_lm_aug_lm_4', 'best.th'])
    aug_language_model_vocab_path = os.path.join(*['.', 'language_model', 'vocab_lm_aug_lm_4'])
    aug_language_model_combination_file = os.path.join(*['.', 'language_model', 'hyper_combs_aug_lm'])
    aug_language_model_index = 4

    GLOVE_PARAMS_CONFIG = {'pretrained_file': '/local/fangzhou/data/glove.840B.300d.txt',
                           'embedding_dim': 300,
                           'vocab_namespace': 'tokens',
                           'padding_index': 0,
                           'trainable': True}
