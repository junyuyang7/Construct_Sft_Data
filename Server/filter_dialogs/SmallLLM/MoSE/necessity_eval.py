import os
import sys
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from utils import jdump, jload
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def necessity_eval(inference_result_file, necessity_evaluation_file):

    inference_list = jload(inference_result_file)

    print('number of input file', len(inference_list))

    reward_name = "../models/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).cuda(), AutoTokenizer.from_pretrained(reward_name)
    
    # For test
    question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
    inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
    score = rank_model(**inputs).logits[0].detach()
    print(float(score))

    result_list = []
    for element in inference_list:
        instruction = element['instruction']
        _input = element['input']
        _output = element['output']
        _generated = element['generated'][:-4]
      
        question = instruction+'\n'+_input
        
        answer = _generated
        
        try:
            inputs = tokenizer(question, answer, return_tensors='pt').to("cuda")
            score = rank_model(**inputs).logits[0].detach()
            final_result = {'instruction':instruction,'input':_input,'output':_output,'generated':_generated,'reward_score':float(score)}
            result_list.append(final_result)
        except:
            print(instruction)
            print(_generated)
            continue

    print('number of data', len(result_list))

    jdump(result_list,necessity_evaluation_file)
    
def select_high_necessity_data(threshold, reward_score_file, result_file):

    reward_score_list = jload(reward_score_file)
    all_num = len(reward_score_list)
    print('all number of instructions', len(reward_score_list))

    num_dict = {}
    result_json = []

    for item in reward_score_list:
        upper_num = math.ceil(item['reward_score'])
        lower_num = math.floor(item['reward_score'])
        num_dict[(lower_num, upper_num)] = num_dict.get((lower_num,upper_num),0) + 1
        if float(item['reward_score']) < threshold:
            result_json.append(item)

    print('The percent of each score:')
    for k, v in num_dict.items():
        print(str(k)+'  :  '+str(v)+'  '+str(float(v)/all_num))

    print('num of bad case : ',len(result_json))
    jdump(result_json,result_file)