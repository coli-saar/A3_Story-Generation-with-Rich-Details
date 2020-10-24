import dill
import numpy as np
import random

from globals import GLOBAL_CONSTANTS


class Seed(object):
    def __init__(self, x_context, e_p_context, e_f_context, agenda):
        self.scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight',
                        'grocery', 'haircut', 'library', 'train', 'tree']
        self.x_context = x_context
        self.e_p_context = e_p_context
        self.e_f_context = e_f_context
        self.agenda = agenda
        self.script = [s for s in self.scripts if self.agenda[0].find(s) != -1][0]


class Samples(object):
    x_context =  list(['yesterday', 'i'])
    sample_bicycle_agenda = ['Story_Begin_bicycle', 'Evoking_bicycle', 'ScrEv_notice_problem', 'ScrEv_get_tools', 'ScrEv_loose_nut', 'ScrEv_remove_tire', 'ScrEv_get_tire', 'ScrEv_take_tire_off', 'ScrEv_examine_tire', 'ScrEv_refill_tire_air', 'ScrEv_check_new_tire', 'ScrEv_ride_bike', 'Story_End_bicycle']
    sample_bicycle_seed = Seed(x_context, ['Story_Begin_' + 'bicycle'] * 2, ['Evoking_' + 'bicycle'] * 2, sample_bicycle_agenda)

    sample_grocery_agenda = ['Story_Begin_grocery', 'Evoking_grocery', 'ScrEv_make_list', 'ScrEv_go_grocery', 'ScrEv_take_shop_cart', 'ScrEv_move_section', 'ScrEv_get_groceries', 'ScrEv_check_off', 'ScrEv_go_checkout', 'ScrEv_cashier_scan/weight', 'ScrEv_wait', 'ScrEv_put_conveyor', 'ScrEv_pay', 'ScrEv_pack_groceries', 'ScrEv_bring_vehicle', 'ScrEv_return_shop_cart', 'ScrEv_leave', 'Story_End_grocery']
    sample_grocery_seed = Seed(x_context, ['Story_Begin_' + 'grocery'] * 2, ['Evoking_' + 'grocery'] * 2, sample_grocery_agenda)

    problematic_flight_agenda = ['Story_Begin_flight', 'Evoking_flight', 'ScrEv_get_ticket', 'ScrEv_present_ID/ticket', 'ScrEv_pack_luggage', 'ScrEv_get_airport', 'ScrEv_go_check_in',
                                 'ScrEv_check_in', 'ScrEv_check_luggage', 'ScrEv_go_security_checks', 'ScrEv_find_terminal', 'ScrEv_wait_boarding',
                                 'ScrEv_board_plane', 'ScrEv_take_seat', 'ScrEv_buckle_seat_belt', 'ScrEv_take_off', 'ScrEv_spend_time_flight',
                                 'ScrEv_landing', 'Story_End_flight']
    p_flight_seed = Seed(x_context, ['Story_Begin_' + 'flight'] * 2, ['Evoking_' + 'flight'] * 2, problematic_flight_agenda)

    problematic_grocery_agenda = ['Story_Begin_grocery', 'Evoking_grocery', 'ScrEv_make_list', 'ScrEv_go_grocery', 'ScrEv_move_section', 'ScrEv_get_groceries', 'ScrEv_check_off', 'ScrEv_go_checkout', 'ScrEv_cashier_scan/weight', 'ScrEv_wait', 'ScrEv_put_conveyor', 'ScrEv_pay', 'ScrEv_pack_groceries', 'ScrEv_bring_vehicle', 'ScrEv_return_shop_cart', 'ScrEv_leave', 'Story_End_grocery']
    p_grocery_seed = Seed(x_context, ['Story_Begin_' + 'grocery'] * 2, ['Evoking_' + 'grocery'] * 2, problematic_grocery_agenda)


class TSG(object):
    bicycle = [
        ['lay_bike_down'], ['get_tools', 'loose_nut', 'pull_air_pin'],
        ['remove_tire'],
        ['get_tire', 'take_tire_off'],
        ['examine_tire'],
        ['put_patch/seal'],
        ['put_new_tire'],
        ['refill_tire_air'],
        ['check_new_tire'],
        ['ride_bike']
    ]
    bus = [
        ['check_time-table', 'find_bus'],
        ['get_bus_stop'],
        ['wait'],
        ['bus_comes'],
        ['board_bus'],
        ['get_ticket'],
        ['find_place'],
        ['ride', 'spend_time_bus'],
        ['press_stop_button'],
        ['bus_stops'],
        ['go_exit'],
        ['get_off'],
    ]
    bath = [
        ['enter_bathroom', 'get_towel'],
        ['prepare_bath'],
        ['take_clean_clothes', 'turn_water_on'],
        ['check_temp', 'close_drain'],
        ['put_bubble_bath_scent', 'fill_water/wait'],
        ['turn_water_off', 'undress'],
        ['sink_water'],
        ['apply_soap', 'relax'],
        ['wash'],
        ['get_out_bath', 'open_drain'],
        ['dry'],
        ['get_dressed', 'leave'],
    ]
    cake = [
        ['choose_recipe'],
        ['get_ingredients'],
        ['preheat'],
        ['get_utensils'],
        ['add_ingredients'],
        ['prepare_ingredients'],
        ['grease_cake_tin'],
        ['pour_dough'],
        ['put_cake_oven'],
        ['set_time'],
        ['wait'],
        ['check', 'take_out_oven'],
        ['turn_off_oven', 'cool_down', 'take_out_cake_tin'],
        ['decorate'],
        ['eat'],
    ]
    flight = [
        ['get_ticket', 'present_id/ticket'],
        ['pack_luggage'],
        ['get_airport'],
        ['go_check_in'],
        ['check_in'],
        ['check_luggage', 'go_security_checks'],
        ['find_terminal'],
        ['wait_boarding'],
        ['present_boarding_pass'],
        ['board_plane'],
        ['stow_away_luggage', 'take_seat'],
        ['buckle_seat_belt', 'listen_crew'],
        ['take_off_preparations'],
        ['take_off'],
        ['spend_time_flight'],
        ['landing'],
        ['exit_plane'],
        ['retrieve_luggage']
    ]
    grocery = [
        ['make_list'],
        ['go_grocery'],
        ['enter'],
        ['take_shop_cart'],
        ['move_section'],
        ['get_groceries', 'check_off'],
        ['check_list'], ['go_checkout'],
        ['cashier_scan/weight'], ['wait'],
        ['put_conveyor'],
        ['pay'],
        ['pack_groceries', 'get_receipt'], ['bring_vehicle'],
        ['return_shop_cart'],
        ['leave']
    ]
    haircut = [
        ['make_appointment'],
        ['get_salon'],
        ['enter'],
        ['check_in'],
        ['wait'],
        ['sit_down'],
        ['put_on_cape'],
        ['talk_haircut'],
        ['move_in_salon', 'wash'],
        ['comb'],
        ['cut'],
        ['brush_hair', 'dry', 'make_hair_style'],
        ['look_mirror'],
        ['customer_opinion'],
        ['pay'],
        ['leave_tip'],
        ['leave'],
    ]
    library = [
        ['get_library', 'browse_releases'],
        ['ask_librarian', 'get_shelf'],
        ['look_for_book', 'obtain_card'],
        ['use_computer'],
        ['check_catalog'],
        ['note_shelf', 'take_book'],
        ['go_check_out'],
        ['show_card'],
        ['register'],
        ['get_receipt'],
        ['leave'],
        ['return_book'],
    ]
    train = [
        ['check_time-table'],
        ['get_train_station'],
        ['get_tickets'],
        ['get_platform'],
        ['wait'],
        ['train_arrives'],
        ['get_on'],
        ['find_place', 'conductor_checks', 'arrive_destination'],
        ['spend_time_train'],
        ['get_off'],
    ]
    tree = [
        ['choose_tree'],
        ['get_tree'],
        ['get_tools'],
        ['take_home'],
        ['find_place'],
        ['dig_hole'],
        ['place_fertilizers', 'unwrap_root'],
        ['place_root'],
        ['refill_hole'],
        ['tamp_dirt'],
        ['water'],
    ]

    script_representations = {'bath': bath, 'bicycle': bicycle, 'bus': bus, 'cake': cake, 'flight': flight,
                              'grocery': grocery, 'haircut': haircut, 'library': library, 'train': train, 'tree': tree}


class Agenda(object):

    event_counter = dill.load(open('e_counters', 'rb'))

    @staticmethod
    def generate_agenda(script, temperature=0.5, with_irregular_events=False):
        """
        generate an agenda randomly. randomness comes from:
            1. randomly drop events from a plausible sequence according to the TSG.
                The probability of dropping is estimated according to the event chains in the corpus.
            2. randomly add irregular events to an agenda generated from step 1.
                Irregular events take ca. 30%. of the segments.
        :param script:
        :param temperature:
        :return:
        """
        'this guy uses segmentation vocabulary'
        irr_counter = dill.load(open('irr_counter', 'rb'))[script]
        'this guy uses original inscript vocabulay'
        dfa = TSG.script_representations[script]
        count = Agenda.event_counter[script]
        agenda = list()
        agenda.append('evoking_' + script)
        for segment in dfa:
            random.shuffle(segment)
            for event in segment:
                sample = np.random.uniform(0, 1)
                if sample < np.power(min(count[event]/100, 1), temperature):
                    agenda.append('screv_' + event + '_' + script)
        #agenda.append('story_end_' + script)
        if with_irregular_events:
            for i, event in enumerate(agenda):
                if np.random.uniform(0,1) < irr_counter[event][2]:
                    agenda.insert(i, '{}_{}'.format(GLOBAL_CONSTANTS.empty_event_label, script))
        return agenda

    @staticmethod
    def generate_random_agenda(script, length):
        dfa = TSG.script_representations[script]
        agenda = list()
        agenda.append('Story_Begin_' + script)
        agenda.append('Evoking_' + script)
        for _ in range(length):
            candidate_list = random.choice(dfa)
            random_event = random.choice(candidate_list)
            agenda.append('ScrEv_' + random_event)
        agenda.append('Story_End_' + script)
        return agenda

    @staticmethod
    def generate_seed(script, temperature=1):
        x_context = list(['yesterday', 'i'])
        e_p_context = ['Story_Begin_' + script] * 2
        e_f_context = ['Evoking_' + script] * 2
        agenda = Agenda.generate_agenda(script, temperature)
        return Seed(x_context, e_p_context, e_f_context, agenda)

    @staticmethod
    def generate_random_seed(script, length):
        x_context = list(['yesterday', 'i'])
        e_p_context = ['Story_Begin_' + script] * 2
        e_f_context = ['Evoking_' + script] * 2
        agenda = Agenda.generate_random_agenda(script, length)
        return Seed(x_context, e_p_context, e_f_context, agenda)

    @staticmethod
    def generate_seeds(scripts, lengths):
        """
        generate 4 seeds for each script: 2 rational and 2 random of length lengths[script]
        :param scripts:
        :param lengths: dict script->length of agenda
        :return:
        """
        seeds = list()
        for script in scripts:
            seeds.append(Agenda.generate_random_seed(script, lengths[script]))
            seeds.append(Agenda.generate_random_seed(script, lengths[script]))
            seeds.append(Agenda.generate_seed(script))
            seeds.append(Agenda.generate_seed(script))
        return seeds


if __name__ == '__main__':
    a = Agenda.generate_agenda('tree')
