from configurations import GlobalConstant
from my_utils import tokenizer
import os
import spacy
import sys


def load_data(script_list=GlobalConstant.effective_scripts, corpus_folder=GlobalConstant.data_path):
    """
    load data, decide how to perform segmentation with segment_surface()
     and add split markers
    :param script_list:
    :param corpus_folder:
    :return:
    """
    corpus = list()
    text = list()
    events = list()
    stories = dict()
    split_markers = dict()

    ''' segment stories '''
    tmp_story = list()
    for script in script_list:
        if script not in stories:
            stories[script] = list()
        with open(os.path.join(corpus_folder, script), 'r') as fin:
            inline = fin.readline()
            while inline != '':
                if inline.find(GlobalConstant.end_of_story_label) == -1:
                    splited_inline = inline.split()
                    splited_inline[0] = splited_inline[0].lower()
                    corpus.append(splited_inline)
                    tmp_story.append(splited_inline)
                    text.append(splited_inline[0])
                    events.extend(splited_inline[1:])
                else:
                    stories[script].append(tmp_story)
                    tmp_story=list()
                inline = fin.readline()

    spacy_nlp = spacy.load("en_core_web_sm")

    ''' add pos tags '''
    for script in script_list:
        print('.', end='')
        sys.stdout.flush()
        for story in stories[script]:
            story_doc = spacy_nlp.tokenizer.tokens_from_list([line[0] for line in story])
            spacy_nlp.tagger(story_doc)
            assert len(story) == len(story_doc)
            #len(story_doc) == len(story)
            for i in range(len(story)):
                story[i].insert(1, story_doc[i].pos_)


    ''' build tokenizers '''
    text_tokenizer = tokenizer(text)
    event_tokenizer = tokenizer(events)

    text_tokenizer.append_type(GlobalConstant.Padding_token)

    for sc in script_list:
        event_tokenizer.append_type(GlobalConstant.Beginning_Event + '_' + sc)
        event_tokenizer.append_type(GlobalConstant.Ending_Event + '_' + sc)

    GlobalConstant.Active_vocabulary_size = text_tokenizer.vocabulary_size
    # to include beginning and ending events in the event list
    GlobalConstant.Event_vocabulary_size = event_tokenizer.vocabulary_size
    print('\n vocabulary size:', text_tokenizer.vocabulary_size, 'text tokens:', len(text),
          'events:', GlobalConstant.Event_vocabulary_size)

    ''' add spliters '''
    for script in script_list:
        if script not in split_markers:
            split_markers[script] = list()
        for story in stories[script]:
            tmp_markers = dict()
            pointer1 = 0
            while len(story[pointer1]) == 2:
                pointer1 += 1
            pointer2 = pointer1 + 1
            while len(story[pointer2]) == 2:
                pointer2 += 1
            while True:
                item = dict()
                try:
                    item['text'] = [story[i][0] for i in range(pointer1, pointer2 + 1, 1)]
                    item['pos'] = [story[i][1] for i in range(pointer1, pointer2 + 1, 1)]
                except IndexError:
                    item['text'] = [story[i][0] for i in range(pointer1, pointer2, 1)]
                    item['pos'] = [story[i][1] for i in range(pointer1, pointer2, 1)]
                item['ann1'] = story[pointer1][2]
                try:
                    item['ann2'] = story[pointer2][2]
                except IndexError:
                    item['ann2'] = 'End_Of_Story'

                item['context'] = ' '.join(
                    [line[0] for line in story[max(pointer1 - 8, 0): min(pointer2 + 9, len(story))]])

                segment_surface(item)
                ''' tmp_marker marks token-wise how to perform segmentation 
                    False: not spliter; True: spliter; 'Erase': delete annotation
                '''
                for i in range(len(item['text'])):
                    if item['spliter'] == 'merge':
                        tmp_markers[pointer1 + i] = False
                        ''' performs merging '''
                        if dominates(item['ann1'], item['ann2']) and len(story[pointer2]) > 2:
                            tmp_markers[pointer2] = 'Erase'
                            tmp_markers[pointer1] = True
                        elif len(story[pointer1]) > 2:
                            tmp_markers[pointer1] = 'Erase'
                            tmp_markers[pointer2] = True
                        #story[pointer1 + i].insert(0, False)
                    elif i == item['spliter'] and pointer1 + i not in tmp_markers:
                        tmp_markers[pointer1 + i] = True
                        #story[pointer1 + i].insert(0, True)
                    elif i != item['spliter'] and pointer1 + i not in tmp_markers:
                        tmp_markers[pointer1 + i] = False
                        #story[pointer1 + i].insert(0, False)

                if pointer2 >= len(story) - 1:
                    break

                pointer1 = pointer2
                pointer2 += 1
                while len(story[pointer2]) == 2 and pointer2 < len(story) - 1:
                    pointer2 += 1
            split_markers[script].append(tmp_markers)

    return stories, split_markers, text_tokenizer, event_tokenizer


def split_data(stories, split_markers, script_list=GlobalConstant.effective_scripts):
    segments = dict()
    for script in script_list:
        if script not in segments:
            segments[script] = list()
        script_stories = stories[script]
        script_split_markers = split_markers[script]
        tmp_segments = list()
        for i in range(len(script_stories)):
            story = script_stories[i]
            markers = script_split_markers[i]

            pointer = 0
            segment = list()
            event = ''
            while pointer < len(story):
                ''' capture segment annotation 
                 possible markers: True, False, 'Erase' 
                 the logic is a bit messy so here you go:
                    marker = True:
                        spliter, split and output
                    marker = Erase:
                        erase event annotation
                    marker = False:
                        not spliter. could have event annotation, so shift event when so.
                        erase annotation before doing so
                 '''
                if len(story[pointer]) > 2:
                    ''' event annotation detected, update segment event annotation if necessary. 
                    also processes case 
                        markers[pointer] == 'Erase' 
                    '''
                    if pointer in markers:
                        if markers[pointer] == 'Erase':
                            ''' remove event annotation '''
                            story[pointer].pop(-1)
                        elif dominates(story[pointer][-1], event):
                            event = story[pointer][-1]
                    segment.append(story[pointer][0])
                    pointer += 1
                elif len(story[pointer]) <= 2:
                    if pointer not in markers:
                        segment.append(story[pointer][0])
                        pointer += 1
                    elif markers[pointer] is False:
                        segment.append(story[pointer][0])
                        pointer += 1
                    elif markers[pointer] is True:
                        segment.append(story[pointer][0])
                        pointer += 1

                        tmp_segments.append([event, segment])
                        event = ''
                        segment = list()

            segments[script].append(tmp_segments)
            tmp_segments = list()

    return segments


def instance_extract(stories, script_list=GlobalConstant.effective_scripts):
    instances = list()
    for script in script_list:
        for story in stories[script]:
            pointer1 = 0
            while len(story[pointer1]) == 2:
                pointer1 += 1
            pointer2 = pointer1 + 1
            while len(story[pointer2]) == 2:
                pointer2 += 1
            while True:
                item = dict()
                try:
                    item['text'] = [story[i][0] for i in range(pointer1, pointer2 + 1, 1)]
                    item['pos'] = [story[i][1] for i in range(pointer1, pointer2 + 1, 1)]
                except IndexError:
                    item['text'] = [story[i][0] for i in range(pointer1, pointer2, 1)]
                    item['pos'] = [story[i][1] for i in range(pointer1, pointer2, 1)]
                item['ann1'] = story[pointer1][2]
                try:
                    item['ann2'] = story[pointer2][2]
                except IndexError:
                    item['ann2'] = 'End_Of_Story'

                item['context'] = ' '.join([line[0] for line in story[max(pointer1 - 8, 0) : min(pointer2 + 9, len(story))]])

                instances.append(item)

                if pointer2 >= len(story) - 1:
                    break

                pointer1 = pointer2
                pointer2 += 1
                while len(story[pointer2]) == 2 and pointer2 < len(story) - 1:
                    pointer2 += 1

    return instances


def dominates(event1, event2):
    """
    determines which event annotation should be discarded when merging
    :param event1:
    :param event2:
    :return:
    """
    if len(event2) == 0:
        return True
    elif len(event1) == 0:
        return False
    elif event2.__contains__('ScrEv') and not event2.__contains__('RelNSc'):
        return False
    elif event1.__contains__('ScrEv') and not event1.__contains__('RelNSc'):
        return True
    elif event2.__contains__('RelNSc'):
        return False
    elif event1.__contains__('RelNSc'):
        return True
    elif event1.__contains__('UnrelEv'):
        return False
    elif event2.__contains__('UnrelEv'):
        return True
    else:
        return False


def instance_filtering(instances):
    punkt_instances = list()
    print('instances: ' + str(len(instances)) + '\n')
    total_count = len(instances)

    ''' [., !, ?, ,] filtering, 1-> 0.655 -> 0.646 -> 0.519'''
    for item in instances:
        for punkt in GlobalConstant.spliter_punctuals:
            if punkt in item['text']:
                item['spliter'] = item['text'].index(punkt)
                punkt_instances.append(item)
                break
    for item in punkt_instances:
        instances.remove(item)
    print('filtered [., !, ?, ,] cases. \n')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')


    ''' merge close ones '''
    merge_instances = list()
    for item in instances:
        if len(item['text']) <= 6:
            item['spliter'] = 'merge'
            merge_instances.append(item)

    for item in merge_instances:
        instances.remove(item)
    print('filtered merge <=5 cases. \n')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')


    ''' and / but as CCONJ '''
    andcc_instances = list()
    for item in instances:
        for i in range(len(item['text'])):
            if item['text'][i] in ['and', 'but'] and item['pos'][i] == 'CCONJ':
                item['spliter'] = i
                andcc_instances.append(item)
                break
    for item in andcc_instances:
        instances.remove(item)
    print('filtered and / but cconj cases.')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')

    ''' valid ADPs '''
    adp_instances = list()
    adps = ['before', 'because', 'after', 'as', 'until', 'but', 'till', 'while', 'so', 'if', 'with']
    advs = ['when', 'while', 'so', 'then', 'once', 'where']
    for item in instances:
        for i in range(len(item['text'])):
            if item['text'][i] in adps and item['pos'][i] == 'ADP'\
                    or item['text'][i] in advs and item['pos'][i] == 'ADV':
                item['spliter'] = i
                adp_instances.append(item)
                break
    for item in adp_instances:
        instances.remove(item)
    print('filtered adp / adv cases.')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')

    ''' to s '''
    to_instances = list()
    for item in instances:
        if item['text'][-2] == 'to' or item['text'][-3] == 'to':
            item['spliter'] = 'to'
            to_instances.append(item)
    for item in to_instances:
        instances.remove(item)
    print('filtered to cases.')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')


    rr_insts = list()
    n_insts = list()
    for item in instances:
        if not item['ann1'].__contains__('UnrelEv') and not item['ann2'].__contains__('UnrelEv') \
                and not item['ann1'].__contains__('RelNSc') and not item['ann2'].__contains__('RelNSc') \
                and not item['ann1'].__contains__('Unclear') and not item['ann2'].__contains__('Unclear')\
                and not item['ann1'].__contains__('other') and not item['ann2'].__contains__('other'):
            rr_insts.append(item)
        else:
            n_insts.append(item)

    ''' misc stuff: not a verb or same annotation '''
    misc_insts = list()
    for item in instances:
        if item['pos'][-1] != 'VERB' or item['pos'][0] != 'VERB':
            misc_insts.append(item)
        elif item['ann1'] == item['ann2']:
            misc_insts.append(item)
    for item in misc_insts:
        instances.remove(item)
    print('filtered misc cases.')
    print('instances: ' + str(len(instances)) + '. Retains ' + str(round(len(instances) / total_count, 3)) + '\n')


def segment_surface(item:dict):
    """
    split instance, return the item enriched with spliter information
    :param item:
    """

    ''' punktuals. periods first, commas second '''
    ''' [., !, ?, ,] '''
    for punkt in ['.', '?', '!']:
        if punkt in item['text']:
            item['spliter'] = len(item['text']) - item['text'][::-1].index(punkt) - 1
            return

    if  item['ann1'] == item['ann2']:
        item['spliter'] = 'merge'
        return

    if ',' in item['text']:
        ''' if there is more than one commas, use the last one '''
        item['spliter'] = len(item['text']) - item['text'][::-1].index(',') - 1
        return

    ''' merge close ones '''
    if len(item['text']) <= 9:
        item['spliter'] = 'merge'
        return

    ''' and / but as CCONJ '''
    for i in range(len(item['text'])):
        if item['text'][i] in ['and', 'but'] and item['pos'][i] == 'CCONJ':
            item['spliter'] = i
            return

    ''' valid ADPs '''
    adps = ['before', 'because', 'after', 'as', 'until', 'but', 'till', 'while', 'so', 'if', 'with']
    advs = ['when', 'while', 'so', 'then', 'once', 'where']
    for i in range(len(item['text'])):
        if item['text'][i] in adps and item['pos'][i] == 'ADP'\
                or item['text'][i] in advs and item['pos'][i] == 'ADV':
            item['spliter'] = i
            return

    ''' to s '''
    if item['text'][-2] == 'to' or item['text'][-3] == 'to':
        item['spliter'] = 'merge' #to'
        return

    ''' whatever else, to be refined with 'Noun / Pronoun' '''

    item['spliter'] = 'merge'

    ''' misc stuff: not a verb or same annotation '''

    # if item['pos'][-1] != 'VERB' or item['pos'][0] != 'VERB' or item['ann1'] == item['ann2']:
    #     item['spliter'] = 'merge'
    #     return


def export_samples(segments, path=os.path.join('.', 'segments')):
    with open(path, 'w') as fout:
        for script in GlobalConstant.effective_scripts:
            fout.write('=================================================\n')
            fout.write('SCRIPT: ' + script + '\n')
            fout.write('=================================================\n')
            for segment in segments[script]:
                fout.write('---------------------------------------------------\n')
                for pair in segment:
                    fout.write(pair[0] + '\t' + ' '.join(pair[1]) + '\n')


def export_data(segments, folder=os.path.join('.', 'segments20191217')):
    for script in GlobalConstant.effective_scripts:
        with open(os.path.join(folder, script), 'w') as fout:
            for story_segment in segments[script]:
                for pair in story_segment:
                    fout.write((pair[0] + '\t' + ' '.join(pair[1]) + '\n').lower().replace('523', '23').replace('+', ' '))
                fout.write(GlobalConstant.end_of_story_label + '\n')


if __name__ == '__main__':

    sc_stories, sc_split_markers, tt, et = load_data()
    sc_segments = split_data(sc_stories, sc_split_markers)
    export_data(sc_segments)
    export_samples(sc_segments)

    exit()

    # sc_instances = instance_extract(sc_stories)
    # instance_filtering(sc_instances)
    # a = 2

    # # Load English tokenizer, tagger, parser, NER and word vectors
    # nlp = spacy.load("en_core_web_sm")
    #
    # # Process whole documents
    # text = ("When Sebastian Thrun started working on self-driving cars at "
    #         "Google in 2007, few people outside of the company took him "
    #         "seriously. “I can tell you very senior CEOs of major American "
    #         "car companies would shake my hand and turn away because I wasn’t "
    #         "worth talking to,” said Thrun, in an interview with Recode earlier "
    #         "this week.")
    # doc = nlp(text)
    #
    # # Analyze syntax
    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    #
    # # Find named entities, phrases and concepts
    # for entity in doc.ents:
    #     print(entity.text, entity.label_)
    #
    # pass

