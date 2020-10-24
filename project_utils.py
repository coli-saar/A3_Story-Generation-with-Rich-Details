"""
misc scripts

1. one file with 1k stories for translation
2. 4 files each with 1k stories for back translation

dict: index: (story, agenda, scenario, paraphrases)

2020.03.24
Fangzhou
"""
import os
from googletrans import Translator
import random


from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


language_s = ['zh-cn', 'fr', 'sl', 'sw']
scenario_s = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
data = dict()
target_folder="segments_paraphrased_20200324"


def data_augmentation(source_folder="segments20191217"):
    """
    we paraphrase the data with back-translation
    :param source_folder:
    :param target_folder:
    :return:
    """
    for scenario in scenario_s:
        print('\n' + scenario)
        paraphrased_stories = list()
        source_reader = open(os.path.join(source_folder, scenario), 'r')
        target_writer = open(os.path.join(target_folder, scenario), 'w')
        segment_buffer, event_buffer = list(), list()
        for line in source_reader:
            if line.find('<end_of_story>') == -1:
                # process normal line
                event, segment = line.split('\t')
                event_buffer.append(event)
                segment_buffer.append(segment)
            else:
                # process end of story line
                # note: target story is without <end_of_story>
                source_story = ' # '.join(segment_buffer)
                for language in language_s:
                    target_text = Translator().translate(text=source_story, dest=language).text
                    back_translated_text = Translator().translate(text=target_text, dest='EN').text
                    # try:
                    #     target_text = translator.translate(text=source_story, dest=language).text
                    #     back_translated_text = translator.translate(text=target_text, dest='en').text
                    # except json.decoder.JSONDecodeError:
                    #     print('+', end='')
                    #     continue
                    target_story = back_translated_text.split('#')
                    paraphrased_stories.append((target_story, event_buffer))
                segment_buffer, event_buffer = list(), list()
                for _ in range(2):
                    paraphrased_stories.append((source_story, event_buffer))
                print('.', end='')
        random.shuffle(paraphrased_stories)
        for story, agenda in paraphrased_stories:
            for index, segment in enumerate(story):
                target_writer.write('{}\t{}\n'.format(agenda[index], segment))
            target_writer.write('<end_of_story>\n')
        source_reader.close()
        target_writer.close()


def generate_tsv(source_folder="segments20191217"):
    """
    we paraphrase the data with back-translation
    :param source_folder:
    :return:
    """

    target_writer = open(os.path.join('translation.tsv'), 'w')
    counter = 0
    for scenario in scenario_s:
        print('\n' + scenario)
        paraphrased_stories = list()
        source_reader = open(os.path.join(source_folder, scenario), 'r')
        segment_buffer, event_buffer = list(), list()
        for line in source_reader:
            if line.find('<end_of_story>') == -1:
                # process normal line
                event, segment = line.split('\t')
                event_buffer.append(event)
                segment_buffer.append(segment)
            else:
                # process end of story line
                # note: target story is without <end_of_story>
                data[counter] = (segment_buffer, event_buffer, scenario, list())
                counter += 1
                segment_buffer, event_buffer = list(), list()
    for index in range(counter):
        target_writer.write(' # '.join(data[index][0]).replace('\n', '') + '\n')
    target_writer.close()


def enrich_data():
    for language in language_s:
        file_name = language + '.tsv'
        with open(file_name, 'r') as fin:
            for index, line in enumerate(fin):
                story = line.split('\t')[2].replace('himself', 'myself').lower().split('#')
                data[index][-1].append(story)


def generate_data():
    stories_by_scenario = dict()
    for index, datum in data.items():
        story, agenda, scenario, back_translations = datum
        for back_translation in back_translations:
            if len(back_translation) != len(story):
                # this happens to 52 stories outof 4 * 910
                continue
            if scenario not in stories_by_scenario:
                stories_by_scenario[scenario] = [(agenda, back_translation)]
            else:
                stories_by_scenario[scenario].append((agenda, back_translation))
        for _ in range(2):
            stories_by_scenario[scenario].append((agenda, story))

    for scenario in scenario_s:
        with open(os.path.join(target_folder, scenario), 'w') as writer:
            random.shuffle(stories_by_scenario[scenario])
            for agenda, story in stories_by_scenario[scenario]:
                for index, event in enumerate(agenda):
                    writer.write('{}\t{}'.format(event,
                                                 story[index] if story[index][-1] == '\n' else story[index] + '\n'))
                writer.write('<end_of_story>\n')




if __name__ == '__main__':
    generate_tsv()
    enrich_data()
    generate_data()
    s = 4
