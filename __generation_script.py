import datetime
import time
import os

from optimization import FleshGenMetaOptimizer
from flesh_model_s import AlternatingSeq2Seq


class ExecutionSettings:
    ''''''
    class MetaOptimizationSettings:
        """
        settings concerning a specific execution of meta optimization. should only be accessed by the meta optimizer.
        """
        '''-------------------------- hyper optimization specifications --------------------------'''

        DEVICE = 6

        # '''-------------------------- training --------------------------'''
        # SAMPLE_AGENDA_REGULAR = ['screv_bring_vehicle_grocery', 'screv_leave_grocery', 'screv_bring_vehicle_grocery']
        # SAMPLE_AGENDA_IRREGULAR = ['screv_bring_vehicle_grocery', 'screv_leave_grocery', 'irregular_grocery',
        #                            'screv_bring_vehicle_grocery']
        # SAMPLE_AGENDA_S = [SAMPLE_AGENDA_REGULAR, SAMPLE_AGENDA_IRREGULAR]
        # SAMPLE_SCENARIO = 'grocery'
        # '''-------------------------- paths --------------------------'''
        # data_path = os.path.join('.', 'segments20191217')
        # train_data_path = os.path.join('.', 'data_seg_train_toytoy')
        # val_data_path = os.path.join('.', 'data_seg_val_toytoy')

    class ModelSpecifications:
        """
        settings that have nothing to do with a specific meta optimization session, but define the model structure to
        be optimized.
        """
        BEAM_SIZE = 19
        MAX_DECODING_LENGTH = 20

        story_per_scenario = 8
        iregular_story_per_scenario = 4
        regular_story_per_scenario = story_per_scenario - iregular_story_per_scenario

        iregular_MMIs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        regular_MMIs = [0., 0.05, 0.1, 0.15]
        generation_scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
        '''-------------------------- model specifications --------------------------'''
        ''' use a same 'irregular event' label for all irregular events. 
        Note that the events could be different in training and inference '''
        is_seq2seq_baseline = False
        MERGE_IRREGULAR_EVENTS = True
        USE_GLOVE = False
        SHARE_ENCODER = False
        SAVE_MODEL = False

        ''' whether to include the MMI term during beam search. Note that it is tured OFF during training '''
        USE_MMI_SCORE = True

        ''' MMI_coefficient for the irregular decoder '''
        IRREGULAR_MMI_COEFFICIENT = 1.
        ''' MMI coefficient for the regular decoder '''
        REGULAR_MMI_COEFFICIENT = 0.
        GLOVE_PARAMS_CONFIG = {'pretrained_file': '/local/fzhai/data/glove.840B.300d.txt',
                               'embedding_dim': 300,
                               'vocab_namespace': 'tokens',
                               'padding_index': 0,
                               'trainable': True}

        ''' paths '''
        LANGUAGE_MODEL_PATH = os.path.join('.', 'language_model')
        LANGUAGE_MODEL_INDEX = 2
        LANGUAGE_MODEL_COMBINATION_FILE_NAME = 'combs_LM_opt_2020-02-10'
        LANGUAGE_MODEL_VOCAB_FOLDER = 'vocab_lm_LM_opt_2020-02-10_2'
        LANGUAGE_MODEL_MODEL_PATH = 'models_lm_LM_opt_2020-02-10_2/best.th'

        # ''' transformer positional encoding '''
        # decoder_model_index = 4
        # decoder_model_path = r'./decoder/models_transformer_no_mask_OPTIM_2020-03-05_4/best.th'
        # decoder_model_vocab_path = r'./decoder/vocab_transformer_no_mask_OPTIM_2020-03-05_4'
        # decoder_combination_file_path = './decoder/hyper_combs_transformer_no_mask_OPTIM_2020-03-05'

        ''' progress encoding '''
        decoder_model_index = 0
        decoder_model_path = r'./decoder/models_progress_no_mask_OPTIM_2020-03-05_0/best.th'
        decoder_model_vocab_path = r'./decoder/vocab_progress_no_mask_OPTIM_2020-03-05_0'
        decoder_combination_file_path = './decoder/hyper_combs_progress_no_mask_OPTIM_2020-03-05'


if __name__ == '__main__':

    ''' script for generation '''
    start = time.time()
    model = AlternatingSeq2Seq.load_model(combination_file=ExecutionSettings.ModelSpecifications.decoder_combination_file_path,
                                           index=ExecutionSettings.ModelSpecifications.decoder_model_index,
                                           vocab_path=ExecutionSettings.ModelSpecifications.decoder_model_vocab_path,
                                           model_path=ExecutionSettings.ModelSpecifications.decoder_model_path,
                                           model_specifications=ExecutionSettings.ModelSpecifications,
                                           device=ExecutionSettings.MetaOptimizationSettings.DEVICE)
    model.load_language_model(path = ExecutionSettings.ModelSpecifications.LANGUAGE_MODEL_PATH,
                               vocab_folder=ExecutionSettings.ModelSpecifications.LANGUAGE_MODEL_VOCAB_FOLDER,
                               model_path=ExecutionSettings.ModelSpecifications.LANGUAGE_MODEL_MODEL_PATH)
    for ire_rate in ExecutionSettings.ModelSpecifications.iregular_MMIs:
        for reg_rate in ExecutionSettings.ModelSpecifications.regular_MMIs:
            ExecutionSettings.ModelSpecifications.IRREGULAR_MMI_COEFFICIENT = ire_rate
            ExecutionSettings.ModelSpecifications.REGULAR_MMI_COEFFICIENT = reg_rate
            print(time.time() - start)
            for scenario in ExecutionSettings.ModelSpecifications.generation_scripts:
                gen_result = model.generate_samples(
                    script_s=[scenario] * ExecutionSettings.ModelSpecifications.story_per_scenario,
                    include_irregular_events=([True] * ExecutionSettings.ModelSpecifications.iregular_story_per_scenario +
                                             [False] * ExecutionSettings.ModelSpecifications.regular_story_per_scenario))
                print(time.time() - start)
                samples = ''
                for sample in gen_result:
                    samples += AlternatingSeq2Seq.print_sample(sample['agenda'], sample['sequences'])
                    print('.', end='')

                with open('samples_{}_{}_{}_{}_{}'.format(
                        ExecutionSettings.ModelSpecifications.USE_MMI_SCORE,
                        ExecutionSettings.ModelSpecifications.IRREGULAR_MMI_COEFFICIENT,
                        ExecutionSettings.ModelSpecifications.REGULAR_MMI_COEFFICIENT,
                        ExecutionSettings.ModelSpecifications.BEAM_SIZE,
                        datetime.date.today()), 'a') as fout:
                    fout.write(samples)
                    fout.write('----------------------------------------------------------------------\n')
