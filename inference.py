"""
generate stories with specified model checkpoint and random agendas. Result is saved in a dill file.
"""

import datetime
import time
import os
import dill

from flesh_model_s import AlternatingSeq2Seq
from utils import Struct


model_config = Struct(
    {
        'combination_file': os.path.join(
            *['.', 'base_single_decoder_n1', 'hyper_combs_forthcoming_event_previous_regular_event_succe'
                                             'sive_regular_event_Mode.OPTIMIZATION_2020-05-22']),
        'index': 1,
        'vocab_path': os.path.join(
            *['.', 'base_single_decoder_n1',
              'vocab_forthcoming_event_previous_regular_event_succesive_regular_event_Mode.OPTIMIZATION_2020-05-22_1']),
        'model_path': os.path.join(
            *['.', 'base_single_decoder_n1',
              'models_forthcoming_event_previous_regular_event_succesive_regular_event_Mode.OPTIMIZATION_2020-05-22_1',
              'best.th']),
        'config': None
    })

generation_scripts = ['bath', 'grocery', 'flight', 'cake']
irregular_story_per_scenario = 3
regular_story_per_scenario = 1
story_per_scenario = irregular_story_per_scenario + regular_story_per_scenario
mmi_s = [0.1]

output_binary = 'generations_{}'.format(str(datetime.date.today()))

if __name__ == '__main__':
    # todo: add mmi
    ''' script for generation '''
    # note: specify model here
    model_config = model_config

    start = time.time()
    model = AlternatingSeq2Seq.load_model(configurations=model_config['config'],
                                          index=model_config['index'],
                                          combination_file=model_config['combination_file'],
                                          vocab_path=model_config['vocab_path'],
                                          model_path=model_config['model_path'])
    print(time.time() - start)
    results = dict()
    with open('samples_{}_{}'.format('tag', datetime.date.today()), 'w') as fout:
        for scenario in generation_scripts:
            results[scenario] = dict()
            for mmi in mmi_s:
                if mmi != 0.0:
                    model_config['config'].USE_MMI_SCORE = True
                    model_config['config'].IRREGULAR_MMI_COEFFICIENT = mmi

                fout.write('======= MMI: {} ===== SCENARIO: {} ========\n'.format(mmi, scenario))

                gen_result = model.generate_samples(
                    script_s=[scenario] * story_per_scenario,
                    include_irregular_events=([True] * irregular_story_per_scenario +
                                              [False] * regular_story_per_scenario))
                print(time.time() - start)
                samples = ''
                for sample in gen_result:
                    samples += AlternatingSeq2Seq.print_sample(sample['agenda'], sample['sequences'])
                    print('.', end='')

                fout.write(samples)
                fout.write('----------------------------------------------------------------------\n')
                results[scenario][str(mmi)] = gen_result
    dill.dump(obj=gen_result, file=open(output_binary, 'wb'))
    print('==generated {} stories in {} seconds'.format(story_per_scenario * len(mmi_s) * len(generation_scripts),
                                                        time.time() - start))
