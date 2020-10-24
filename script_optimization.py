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
    MODE = FleshGenMetaOptimizer.Mode.OPTIMIZATION
    if os.getcwd().__contains__('exe'):
        assert MODE != FleshGenMetaOptimizer.Mode.TEST
    LOG_TO_TENSORBOARD = True
    SAVE_MODEL = True
    '''-------------------------- hyper optimization specifications --------------------------'''
    param_domains = {'batch_size': [16, 127],
                     'ed_ncoder_size': [256, 2047],
                     'lr': [1e-4, 2e-3] if MODE != FleshGenMetaOptimizer.Mode.AUGMENTED_OPTIMIZATION else [1e-4, 2e-2],
                     'percent_student': [1e-3, 0.3],
                     'l2': [1e-4, 5e-2] if MODE != FleshGenMetaOptimizer.Mode.AUGMENTED_OPTIMIZATION else [1e-3, 2e-1],
                     'dropout': [0.05, 0.8],
                     'clip': [1, 10],
                     'max_progress_len': [1, 100],
                     'num_hidden_layeres': [1, 1]}
    if MODE is FleshGenMetaOptimizer.Mode.TEST:
        num_trials = 1
        DEVICE = 3
        MAX_EPOCHS = 80
        PATIENCE = 20
    elif MODE is FleshGenMetaOptimizer.Mode.OPTIMIZATION:
        num_trials = 5
        DEVICE = 4
        MAX_EPOCHS = 40
        PATIENCE = 35
    elif MODE is FleshGenMetaOptimizer.Mode.AUGMENTED_OPTIMIZATION:
        num_trials = 15
        DEVICE = 5
        MAX_EPOCHS = 100
        PATIENCE = 75
    # the components of decoder and their corresponding dimensionality
    positional_encoding = 'none'  # 'transformer_forth', 'transformer_context'
    mask_regular_events_for_pe = True
    # simple: simple text encoder; event: text embeddings concatenated with context event
    encoder_variant = ['simple', 'event'][1]
    default_decoder_components = ['previous_token', 'encoder_att']
    # extra_decoder_components = ['forthcoming_event', 'contextualized_event', 'agenda_att', 'previous_regular_event',
    #                             'succesive_regular_event']
    extra_decoder_components = ['forthcoming_event', 'previous_regular_event', 'succesive_regular_event']
    # sketch encoder does not admit agenda_att as final layer input
    final_layer_components = []
    assert 'agenda_att' not in final_layer_components

    tag = '{}_{}_{}'.format('_'.join(extra_decoder_components + final_layer_components),
                            MODE, str(datetime.date.today()))
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
    SAMPLE_TREE_REGULAR = ['evoking_tree', 'screv_choose_tree_tree', 'screv_get_tree_tree', 'screv_get_tools_tree',
                           'screv_take_home_tree', 'screv_find_place_tree', 'screv_dig_hole_tree', 'screv_place_fertilizers_tree',
                           'screv_unwrap_root_tree', 'screv_place_root_tree', 'screv_refill_hole_tree', 'screv_tamp_dirt_tree',
                           'screv_water_tree']
    SAMPLE_TREE_IRREGULAR = ['evoking_tree', 'screv_choose_tree_tree', 'irregular_tree', 'screv_get_tree_tree',
                             'irregular_tree', 'screv_get_tools_tree',
                             'screv_take_home_tree', 'irregular_tree',
                             'screv_find_place_tree', 'screv_dig_hole_tree', 'irregular_tree',
                             'screv_place_fertilizers_tree',
                             'irregular_tree',
                             'screv_unwrap_root_tree', 'irregular_tree',
                             'screv_place_root_tree', 'screv_refill_hole_tree', 'irregular_tree',
                             'screv_tamp_dirt_tree',
                             'screv_water_tree', 'irregular_tree']
    SAMPLE_BICYCLE_REGULAR = ['evoking_bicycle', 'screv_lay_bike_down_bicycle',
                              'screv_get_tools_bicycle', 'screv_loose_nut_bicycle',
                              'screv_pull_air_pin_bicycle',
                              'screv_remove_tire_bicycle',
                              'screv_get_tire_bicycle', 'screv_take_tire_off_bicycle',
                              'screv_examine_tire_bicycle',
                              'screv_put_patch/seal_bicycle',
                              'screv_put_new_tire_bicycle',
                              'screv_refill_tire_air_bicycle',
                              'screv_check_new_tire_bicycle',
                              'screv_ride_bike_bicycle']
    SAMPLE_BICYCLE_IRREGULAR = ['evoking_bicycle', 'irregular_bicycle', 'screv_lay_bike_down_bicycle', 'irregular_bicycle',
                                'screv_get_tools_bicycle', 'screv_loose_nut_bicycle',
                                'irregular_bicycle', 'screv_pull_air_pin_bicycle',
                                'irregular_bicycle', 'screv_remove_tire_bicycle',
                                'screv_get_tire_bicycle', 'screv_take_tire_off_bicycle', 'irregular_bicycle',
                                'screv_examine_tire_bicycle', 'irregular_bicycle',
                                'screv_put_patch/seal_bicycle',
                                'screv_put_new_tire_bicycle', 'irregular_bicycle',
                                'screv_refill_tire_air_bicycle',
                                'screv_check_new_tire_bicycle', 'irregular_bicycle',
                                'screv_ride_bike_bicycle' 'irregular_bicycle']
    SAMPLE_FLIGHT_IRREGULAR = ['evoking_flight',
                               'screv_get_ticket_flight', 'screv_present_id/ticket_flight', 'irregular_flight',
                               'screv_pack_luggage_flight', 'irregular_flight',
                               'screv_get_airport_flight', 'irregular_flight',
                               'screv_go_check_in_flight', 'irregular_flight',
                               'screv_check_in_flight',
                               'screv_check_luggage_flight',  'irregular_flight', 'screv_go_security_checks_flight',
                               'screv_find_terminal_flight',
                               'screv_wait_boarding_flight', 'irregular_flight',
                               'screv_present_boarding_pass_flight', 'irregular_flight',
                               'screv_board_plane_flight', 'irregular_flight',
                               'screv_stow_away_luggage_flight', 'screv_take_seat_flight', 'irregular_flight',
                               'screv_buckle_seat_belt_flight', 'screv_listen_crew_flight',  'irregular_flight',
                               'screv_take_off_preparations_flight',  'irregular_flight',
                               'screv_take_off_flight', 'irregular_flight',
                               'screv_spend_time_flight_flight', 'irregular_flight',
                               'screv_landing_flight',
                               'screv_exit_plane_flight', 'irregular_flight',
                               'screv_retrieve_luggage_flight']
    SAMPLE_FLIGHT_REGULAR = ['evoking_flight',
                             'screv_get_ticket_flight', 'screv_present_id/ticket_flight',
                             'screv_pack_luggage_flight',
                             'screv_get_airport_flight',
                             'screv_go_check_in_flight',
                             'screv_check_in_flight',
                             'screv_check_luggage_flight', 'screv_go_security_checks_flight',
                             'screv_find_terminal_flight',
                             'screv_wait_boarding_flight',
                             'screv_present_boarding_pass_flight',
                             'screv_board_plane_flight',
                             'screv_stow_away_luggage_flight', 'screv_take_seat_flight',
                             'screv_buckle_seat_belt_flight', 'screv_listen_crew_flight',
                             'screv_take_off_preparations_flight',
                             'screv_take_off_flight',
                             'screv_spend_time_flight_flight',
                             'screv_landing_flight',
                             'screv_exit_plane_flight',
                             'screv_retrieve_luggage_flight']
    SAMPLE_AGENDA_S = [SAMPLE_AGENDA_REGULAR, SAMPLE_AGENDA_IRREGULAR] if MODE == FleshGenMetaOptimizer.Mode.TEST \
        else [SAMPLE_AGENDA_REGULAR, SAMPLE_AGENDA_IRREGULAR, SAMPLE_BATH_REGULAR, SAMPLE_BATH_IRREGULAR,
              SAMPLE_BICYCLE_REGULAR, SAMPLE_BICYCLE_IRREGULAR, SAMPLE_FLIGHT_REGULAR, SAMPLE_FLIGHT_IRREGULAR,
              SAMPLE_TREE_REGULAR, SAMPLE_TREE_IRREGULAR]
    SAMPLE_SCENARIO_S = ['grocery', 'grocery', 'bath', 'bath', 'bicycle', 'bicycle', 'flight', 'flight', 'tree', 'tree']
    '''-------------------------- paths --------------------------'''
    if MODE == FleshGenMetaOptimizer.Mode.OPTIMIZATION:
        train_data_path = os.path.join('.', 'data_seg_train')
        val_data_path = os.path.join('.', 'data_seg_val')
    elif MODE == FleshGenMetaOptimizer.Mode.TEST:
        train_data_path = os.path.join('.', 'data_seg_train_toytoy')
        val_data_path = os.path.join('.', 'data_seg_val_toytoy')
    else:
        train_data_path = os.path.join('.', 'segments_paraphrased_20200324')
        val_data_path = os.path.join('.', 'data_seg_val')

    '''----------------------------------------- model specifications -------------------------------------------'''
    BEAM_SIZE = 5
    MAX_DECODING_LENGTH = 20
    MERGE_IRREGULAR_EVENTS = True
    multiply_emb = False
    is_seq2seq_baseline = False
    SHARE_ENCODER = False

    ''' whether to include the MMI term during beam search. Note that it is tured OFF during training '''
    USE_MMI_SCORE = True
    ''' MMI_coefficient for the irregular decoder '''
    IRREGULAR_MMI_COEFFICIENT = 0.1
    ''' MMI coefficient for the regular decoder '''
    REGULAR_MMI_COEFFICIENT = 0.
    GLOVE_PARAMS_CONFIG = {'pretrained_file': '/local/fangzhou/data/glove.840B.300d.txt',
                           'embedding_dim': 300,
                           'vocab_namespace': 'words',
                           'padding_index': 0,
                           'trainable': True}


if __name__ == '__main__':
    meta_optimizer = FleshGenMetaOptimizer(configurations=Configurations, model=AlternatingSeq2Seq)
    meta_optimizer.search()
