"""
optimization script

Zhai Fangzhou
"""
from optimization import FleshGenMetaOptimizer
# from globals import GlobalConstants
from flesh_model_s import AlternatingSeq2Seq
import datetime
import os


class Configurations:
    """
    settings concerning a specific execution of meta optimization. should only be accessed by the meta optimizer.
    """
    # fixme: should this be associated with the meta optimizer, instead of being a stupid, independent class here?
    #  yes; this is just data / argument, not really a class. yes this is ugly, but it cleanly separates and
    #  works atm.
    MODE = FleshGenMetaOptimizer.Mode.TEST
    if os.getcwd().__contains__('exe'):
        assert MODE != FleshGenMetaOptimizer.Mode.TEST
    LOG_TO_TENSORBOARD = True
    SAVE_MODEL = True
    '''-------------------------- hyper optimization specifications --------------------------'''
    param_domains = {'batch_size': [16, 255],
                     'ed_ncoder_size': [128, 1023],
                     'lr': [1e-5, 2e-3] if MODE != FleshGenMetaOptimizer.Mode.AUGMENTED_OPTIMIZATION else [1e-4, 2e-2],
                     'percent_student': [1e-4, 0.2],
                     'l2': [1e-4, 5e-2],
                     'dropout': [0.05, 0.8],
                     'clip': [1, 10],
                     'max_progress_len': [1, 100],
                     'num_hidden_layeres': [1, 2.99]}
    if MODE is FleshGenMetaOptimizer.Mode.TEST:
        num_trials = 1
        DEVICE = 6
        MAX_EPOCHS = 8
        PATIENCE = 20
    else:
        num_trials = 15
        DEVICE = 4
        MAX_EPOCHS = 80
        PATIENCE = 40

    # the components of decoder and their corresponding dimensionality
    positional_encoding = 'none'  # 'transformer_forth', 'transformer_context'
    mask_regular_events_for_pe = True
    # simple: simple text encoder; event: text embeddings concatenated with context event
    encoder_variant = ['simple', 'event'][0]
    default_decoder_components = ['previous_token', 'encoder_att']
    # extra_decoder_components = ['forthcoming_event', 'contextualized_event', 'agenda_att', 'previous_regular_event',
    #                             'succesive_regular_event']
    extra_decoder_components = ['agenda_att']
    # sketch encoder does not admit agenda_att as final layer input
    final_layer_components = []
    tag = '{}_{}_{}'.format(' '.join(extra_decoder_components), MODE, str(datetime.date.today()))
    '''-------------------------- training --------------------------'''
    SAMPLE_AGENDA_REGULAR = ['evoking_grocery', 'screv_make_list_grocery',
                             'screv_go_grocery_grocery', 'screv_take_shop_cart_grocery',
                             'screv_move_section_grocery',
                             'screv_get_groceries_grocery', 'screv_check_off_grocery',
                             'screv_go_checkout_grocery',  'screv_put_conveyor_grocery',
                             'screv_wait_grocery',
                             'screv_cashier_scan/weight_grocery',
                             'screv_pay_grocery', 'screv_pack_groceries_grocery',
                             'screv_bring_vehicle_grocery',
                             'screv_return_shop_cart_grocery', 'screv_leave_grocery'] \
        if MODE != FleshGenMetaOptimizer.Mode.TEST \
        else ['screv_bring_vehicle_grocery', 'screv_leave_grocery', 'screv_bring_vehicle_grocery']
    SAMPLE_BATH_REGULAR = ['evoking_bath', 'screv_enter_bathroom_bath', 'screv_get_towel_bath',
                           'screv_take_clean_clothes_bath', 'screv_turn_water_on_bath', 'screv_check_temp_bath',
                           'screv_close_drain_bath', 'screv_fill_water/wait_bath', 'screv_turn_water_off_bath',
                           'screv_undress_bath', 'screv_sink_water_bath', 'screv_apply_soap_bath',
                           'screv_relax_bath', 'screv_wash_bath', 'screv_get_out_bath_bath',
                           'screv_open_drain_bath', 'screv_dry_bath', 'screv_get_dressed_bath', 'screv_leave_bath']
    SAMPLE_BATH_IRREGULAR = ['evoking_bath', 'screv_enter_bathroom_bath', 'irregular_bath', 'screv_get_towel_bath',
                             'screv_take_clean_clothes_bath', 'irregular_bath', 'screv_turn_water_on_bath',
                             'screv_check_temp_bath', 'screv_close_drain_bath', 'irregular_bath',
                             'screv_fill_water/wait_bath', 'irregular_bath', 'screv_turn_water_off_bath',
                             'screv_undress_bath', 'irregular_bath', 'screv_sink_water_bath',
                             'screv_apply_soap_bath', 'screv_relax_bath', 'irregular_bath', 'screv_wash_bath',
                             'irregular_bath', 'screv_get_out_bath_bath', 'screv_open_drain_bath',
                             'irregular_bath', 'screv_dry_bath', 'irregular_bath', 'screv_get_dressed_bath',
                             'screv_leave_bath']
    SAMPLE_AGENDA_IRREGULAR = ['evoking_grocery', 'screv_make_list_grocery', 'irregular_grocery',
                               'screv_go_grocery_grocery', 'irregular_grocery',
                               'screv_take_shop_cart_grocery', 'screv_move_section_grocery', 'irregular_grocery',
                               'screv_get_groceries_grocery', 'irregular_grocery', 'screv_check_off_grocery',
                               'screv_go_checkout_grocery', 'irregular_grocery',
                               'screv_wait_grocery',
                               'screv_put_conveyor_grocery',
                               'screv_cashier_scan/weight_grocery',
                               'irregular_grocery', 'screv_pay_grocery', 'screv_pack_groceries_grocery',
                               'screv_bring_vehicle_grocery', 'irregular_grocery',
                               'screv_return_shop_cart_grocery', 'irregular_grocery', 'screv_leave_grocery',
                               'irregular_grocery'] \
        if MODE != FleshGenMetaOptimizer.Mode.TEST \
        else ['screv_bring_vehicle_grocery', 'screv_leave_grocery', 'irregular_grocery',
              'screv_bring_vehicle_grocery']
    SAMPLE_AGENDA_S = [SAMPLE_AGENDA_REGULAR, SAMPLE_AGENDA_IRREGULAR] if MODE == FleshGenMetaOptimizer.Mode.TEST \
        else [SAMPLE_AGENDA_REGULAR, SAMPLE_AGENDA_IRREGULAR, SAMPLE_BATH_REGULAR, SAMPLE_BATH_IRREGULAR]
    SAMPLE_SCENARIO_S = ['grocery', 'grocery', 'bath', 'bath']
    '''-------------------------- paths --------------------------'''
    if MODE == FleshGenMetaOptimizer.Mode.OPTIMIZATION:
        train_data_path = os.path.join('.', 'data_seg_train')
        val_data_path = os.path.join('.', 'data_seg_val')
    elif MODE == FleshGenMetaOptimizer.Mode.TEST:
        train_data_path = os.path.join('.', 'data_seg_train_toytoy')
        val_data_path = os.path.join('.', 'data_seg_val_toytoy')
    else:
        train_data_path = os.path.join('.', 'segments_paraphrased_20200324')
        val_data_path = os.path.join('.', 'data_seg_val_toytoy')

    '''----------------------------------------- model specifications -------------------------------------------'''
    BEAM_SIZE = 19
    MAX_DECODING_LENGTH = 20
    MERGE_IRREGULAR_EVENTS = True
    multiply_emb = False
    is_seq2seq_baseline = False
    SHARE_ENCODER = False

    ''' whether to include the MMI term during beam search. Note that it is tured OFF during training '''
    USE_MMI_SCORE = False

    ''' MMI_coefficient for the irregular decoder '''
    IRREGULAR_MMI_COEFFICIENT = 1.
    ''' MMI coefficient for the regular decoder '''
    REGULAR_MMI_COEFFICIENT = 0.
    GLOVE_PARAMS_CONFIG = {'pretrained_file': '/local/fangzhou/data/glove.840B.300d.txt',
                           'embedding_dim': 300,
                           'vocab_namespace': 'words',
                           'padding_index': 0,
                           'trainable': True}


class ConfigContext:
    COMBINATION_FILE = os.path.join(*['.', 'decoder1', 'hyper_combs_contextualized_event_Mode.OPTIMIZATION_2020-04-17'])
    INDEX = 12
    VOCAB_PATH = os.path.join(*['.', 'decoder1', 'vocab_contextualized_event_Mode.OPTIMIZATION_2020-04-17_12'])
    MODEL_PATH = os.path.join(*['.', 'decoder1', 'models_contextualized_event_Mode.OPTIMIZATION_2020-04-17_12',
                                'model_state_epoch_45.th'])
    SAMPLE_FILE = os.path.join(*['.', 'decoder1', 'sample'])


class ConfigAgendaAtt:
    COMBINATION_FILE = os.path.join(*['.', 'decoder',
                                      'hyper_combs_agenda_att_Mode.OPTIMIZATION_2020-05-05'])
    INDEX = 13
    VOCAB_PATH = os.path.join(*['.', 'decoder', 'vocab_agenda_att_Mode.OPTIMIZATION_2020-05-05_13'])
    MODEL_PATH = os.path.join(*['.', 'decoder', 'models_agenda_att_Mode.OPTIMIZATION_2020-05-05_13',
                                'model_state_epoch_46.th'])
    SAMPLE_FILE = os.path.join(*['.', 'decoder', 'sample'])


if __name__ == '__main__':
    # note: configurations must match that of the saved model! should we again separate 'model specifications' ?
    #  i.e. one for creating / recreating the model, the other for the rest
    model = AlternatingSeq2Seq.load_model(combination_file=ConfigAgendaAtt.COMBINATION_FILE,
                                          index=ConfigAgendaAtt.INDEX,
                                          vocab_path=ConfigAgendaAtt.VOCAB_PATH,
                                          model_path=ConfigAgendaAtt.MODEL_PATH,
                                          configurations=Configurations)
    model.track_attention(input_folder=ConfigAgendaAtt.SAMPLE_FILE)
