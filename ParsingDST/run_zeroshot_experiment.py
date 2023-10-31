import os
import json
import argparse
import copy
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from utils.helper import SpeedLimitTimer
from utils.typo_fix import typo_fix
from config import CONFIG

from utils.sql import sql_pred_parse, sv_dict_to_string
from evaluate_metrics import evaluate

from prompt_utils import prompt_func
from prompt_utils import chat_reply
pf = prompt_func()

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./expts/zero-shot",
                    help="directory to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.1",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--test_fn', type=str, default='',
                    help="file to evaluate on, empty means use the test set")
args = parser.parse_args()

# create the output folder
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

NUM_EXAMPLE = 10

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    if args.test_fn == "":
        test_set_path = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    if args.test_fn == "":
        test_set_path = "./data/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_set = json.load(f)


def run(test_set, all_result, info, result_dict, turn=-1, use_gold=False, domain=""):
    
    timer = SpeedLimitTimer(second_per_step=3.1)

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError(
                "can only evaluate particular turn when using gold context") 
        selected_set = [d for d in test_set if len(d['dialog']['usr']) == turn + 1] 


    data_n = 0
    info_path = os.path.join(args.output_dir, "info.json")
    result_dict_path = os.path.join(args.output_dir, "result_dict.json")
    all_result_path = os.path.join(args.output_dir, "running_log.json")
    n_total = info['n_total']
    n_correct = info['n_correct']
    total_acc = info['total_acc']
    total_f1 = info['total_f1']
    finish_test = False
    api_n = 0
    total_api_n = len(CONFIG["api_key"])
    
    for data_item in tqdm(selected_set):
    # for data_item in tqdm(selected_set):
        if data_n<n_total:
            data_n+=1
            continue
        data_n = n_total
        data_n += 1
        n_total += 1
        completion = ""
        data_dialogue = []
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = ""
        if data_item['turn_id']!=0: 
            predicted_context = all_result[-1]['pred']
        else:
            predicted_context = {}

        data_domains = data_item["domains"]

        if data_item['turn_id']==0:
            data_dialogue.append("{\"user\": {\"reject\": {}, \"request\":{}}}")
            data_dialogue.append('user: \"' + data_item['dialog']["usr"][-1] + '\"')
        else:
            pre_prompt_state = pf.state2pre(**predicted_context)
            pre_state_text =  str(pre_prompt_state).replace('\'', '\"')  
            data_dialogue.append(pre_state_text)
            data_dialogue.append('system: \"' + data_item['dialog']["sys"][-1] + '\"')
            data_dialogue.append('user: \"' + data_item['dialog']["usr"][-1] + '\"')

        complete_flag = False
        finish_test = False
        parse_error_count = 0
        prompt_state = {}
        sys_filter_dic = {"system": {"not_avaliable": {}, "info": {}, "ask_for": {}}} 
        retrieved_state = {}
        completion = ""
        if predicted_context:
            for s in predicted_context:
                retrieved_state[s] = [predicted_context[s]]
                
        if data_item['turn_id']!=0:
            while not complete_flag:
                try:
                    prompt_sys = pf.get_prompt([data_dialogue[0], data_dialogue[1]], \
                                               data_domains, prompt_path = "./exms/sys_gpt35.txt") 
                    completion = chat_reply(prompt_sys, 
                                            api_key = CONFIG["api_key"][api_n], api_base = CONFIG["api_base"], 
                                            api_organization = CONFIG["api_organization"], stop=['[END]'])
                    data_item['completion'] += completion+" \n"
                except Exception as e: 
                    try:
                        if e.user_message.startswith("You exceeded your current quota"):
                            print("out of api {}".format(api_n))
                            finish_test = True
                            break
                        else:
                            timer.sleep(10)
                    except:
                        print("error")
                        finish_test = True
                        break
                else:
                    try:
                        temp_state_dict = eval(completion)
                    except:
                        parse_error_count += 1
                        if parse_error_count >= 5:
                            complete_flag = True
                    else:
                        complete_flag = True
                timer.step()
            if finish_test:
                break
            # aggregate the prediction and the history states
            print_data = prompt_sys.split('context: ')[-1].split('output JSON: ')[0]
            print('context: '+print_data+"output JSON: ")
            print(completion)
            state_dict = {"system": {"not_avaliable": {}, "info": {}, "ask_for": {}}}
            try:
                state_dict = eval(completion)
                print()
                print('sys_utt_dic: ')
                sys_utt_dic = state_dict["system"]
                # sys_utt_dic_copy = deepcopy(sys_utt_dic)
                print(sys_utt_dic)
                print('prompt_state: ')
                prompt_state, sys_filter_dic, _ = pf.sys_filter(state=retrieved_state, state_dic=sys_utt_dic, user_pre=pre_prompt_state)
                # prompt_state, sys_filter_dic, user_pre_corrected = pf.sys_filter(state=retrieved_state, state_dic=sys_utt_dic, user_pre=pre_prompt_state)
                # sys_filter_dic = sys_utt_dic_copy
                user_pre_corrected = pre_prompt_state
                pre_state_text =  str(user_pre_corrected).replace('\'', '\"')  
                sys_filter_dic = {"system": sys_filter_dic}
                
            except:
                print("the output is not a query: ")
                data_item['not_valid'] = 1


        completion = ""
        sys_filter_dic = str(sys_filter_dic).replace('\'', '\"')
        complete_flag = False
        while not complete_flag:
            try:
                if data_item['turn_id']!=0:
                    _data_dialogue = [pre_state_text+" \n"+sys_filter_dic, data_dialogue[2]]
                else:
                    _data_dialogue = [data_dialogue[0], data_dialogue[1]]
                    
                prompt_usr = pf.get_prompt(_data_dialogue, data_domains, prompt_path = "./exms/usr_gpt35.txt") 
                completion = chat_reply(prompt_usr, 
                                        api_key = CONFIG["api_key"][api_n], api_base = CONFIG["api_base"], 
                                        api_organization = CONFIG["api_organization"], stop=['[END]'])
                data_item['completion'] += completion
            except Exception as e: 
                try:
                    if e.user_message.startswith("You exceeded your current quota"):
                        print("out of api {}".format(api_n))
                        finish_test = True
                        break
                    else:
                        timer.sleep(10)
                except:
                    print("error")
                    finish_test = True
                    break
            else:
                try:
                    temp_state_dict = eval(completion)
                except:
                    parse_error_count += 1
                    if parse_error_count >= 5:
                        complete_flag = True
                else:
                    complete_flag = True

            timer.step()
        if finish_test:
            break
        # aggregate the prediction and the history states
        print_data = prompt_usr.split('context: ')[-1].split('output JSON: ')[0]
        print('context: '+print_data+"output JSON: ")
        print(completion)
        state_dict = {"user": {"reject": {}, "request": {}}}
        try:
            state_dict = eval(completion)
            print('usr_utt_dic: ')
            usr_utt_dic = state_dict["user"]
            print(usr_utt_dic)
            print('prompt_state: ')
            prompt_state = pf.dialogdic2state(only_entity=False, state = prompt_state, state_dic = usr_utt_dic)
            print(prompt_state)
            print()

        except:
            print("the output is not a query: ")
            data_item['not_valid'] = 1

        
        predicted_slot_values = {}
        for s in prompt_state:
            if prompt_state[s] and prompt_state[s][0] not in [[],'', '[DELETE]', 'empty','null','none','special',\
                                                              'specific','particular','certain','care','matter']:
                predicted_slot_values[s] = prompt_state[s][0]

        data_item['pred'] = predicted_slot_values
        print('predicted_slot_values: ')
        print(predicted_slot_values)
        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        all_slot_values = {}
        # all_slot_values = prediction_recorder.state_retrieval(data_item).copy()
        all_slot_values = {k:v for k,v in predicted_slot_values.items() if k in ontology}

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0]
                           for k, v in all_slot_values.items()}

        all_result.append(data_item)

        # print the result
        print()
        print(
            f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(
            f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        print(
            f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(
            f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(all_slot_values, data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']]=1
            # result_dict[data_item['turn_id']].append(1)
            print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']]=0
            # result_dict[data_item['turn_id']].append(0)
            print("\n=====================wrong!=======================") 
        # print('\nn_total: ' + str(n_total) + '\nn_correct: ' +  str(n_correct) + '\ntotal_acc: ' + str(total_acc) + '\ntotal_f1: ' + str(total_f1))
        # print("\n")
        info['n_total'] = n_total
        info['n_correct'] = n_correct
        info['total_acc'] = total_acc
        info['total_f1'] = total_f1
        
        # with open(info_path, 'w') as f:
        #     json.dump(info, f, indent=4)
        # with open(all_result_path, 'w') as f:
        #     json.dump(all_result, f, indent=4)
        # with open(result_dict_path, 'w') as f:
        #     json.dump(result_dict, f, indent=4)
             
    # print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    # print(f"Slot Acc {total_acc/n_total}")
    # print(f"Joint F1 {total_f1/n_total}")
    # print()

    # calculate the accuracy of each turn
    # for k, v in result_dict.items():
    #     print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")
    return all_result, info, result_dict

if __name__ == "__main__":
    
    all_result_path = os.path.join(args.output_dir, "running_log.json")
    if os.path.exists(all_result_path):
        with open(all_result_path, 'r') as f:
            all_result = json.load(f)
    else:
        all_result = []
        
    # use to record the accuracy
    result_dict_path = os.path.join(args.output_dir, "result_dict.json")
    if os.path.exists(result_dict_path):
        with open(result_dict_path, 'r') as f:
            result_dict = json.load(f)
    else:
        result_dict = {}
        
    info_path = os.path.join(args.output_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {'n_total': 0, 'n_correct': 0, 'total_acc': 0, 'total_f1': 0}
    
    all_result, info, result_dict = run(test_set, all_result, info, result_dict)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    with open(all_result_path, 'w') as f:
        json.dump(all_result, f, indent=4)
    with open(result_dict_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
