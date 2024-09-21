import math
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import jdump, jload
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def quality_eval(file_in, file_out):

    input_list = jload(file_in)
    print('number of input file', len(input_list))

    reward_name = "../models/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).cuda(), AutoTokenizer.from_pretrained(reward_name)
    question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
    inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
    score = rank_model(**inputs).logits[0].detach()
    print(float(score))

    result_list = []
    for element in input_list:
        instruction = element['instruction']
        _input = ''
        if 'input' in element.keys():
            _input = element['input']
        _output = element['output']
        question = ''
        if _input == '':
            question = instruction
        else:
            question = instruction + '\n' +_input
    
        answer = _output
    
        try:
            inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
            score = rank_model(**inputs).logits[0].detach()
        except:
            print(instruction)
            print(_output)
            continue
        final_result = {'instruction':instruction,'input':_input,'output':_output,'reward_score':float(score)}
        result_list.append(final_result)

    print('number of data', len(result_list))

    jdump(result_list,file_out)


def instruct_dict(json_list):
    result_dict = {}
    for item in json_list:
        if item['instruction'] in result_dict:
            print('Exist the same instruction in this dataset!')
        else:
            result_dict[item['instruction']] = 0
    return result_dict

def instruct_category(json_file):
    category_set = []
    instruct_category_map = {}

    json_data = jload('./category.json')
    
    category_set = json_data.keys()
    
    for k, v in json_data.items():
        for ins in v:
            if ins in instruct_category_map and instruct_category_map.get(ins) == k:
                print('Category error!')
                print('category', k)
                print('instruction', ins)
            else:
                instruct_category_map[ins] = k
    
    return category_set, instruct_category_map

def select_high_quality_data(threshold, quality_evaluation_file, high_quality_file):

    quality_evaluation_list = jload(quality_evaluation_file)

    all_num = len(quality_evaluation_list)
    print('all number of instructions', len(quality_evaluation_list))

    num_dict = {}

    result_json = []

    for item in quality_evaluation_list:
        upper_num = math.ceil(item['reward_score'])
        lower_num = math.floor(item['reward_score'])
        num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
        if float(item['reward_score']) > threshold:
            result_json.append(item)

    print('The percent of each score interval:')
    for k, v in num_dict.items():
        print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

    print('num of good case : ',len(result_json))

    #jdump(result_json,result_file)
    jdump(result_json,high_quality_file)