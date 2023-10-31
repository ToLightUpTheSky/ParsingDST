import os, json, re
import openai
from copy import deepcopy

class prompt_func:
    def __init__(self, schema_path = './slot_description.json', exm_dir_path = "./exms/", \
                 end_token = "[EOS]", key_word_token = "[KW]", pre_state_input_token = "[PSI]", \
                 pre_state_output_token = "[PSO]", domain_exms_token = "[EXM]"):
        self.domains = ["train", "taxi", "restaurant", "lodging", "attraction"]
        self.key_domains = \
        {"train": ["time", "num"],
         "taxi": ["time"],
         "restaurant": ["direction", "num", "price_range", "time"],
         "lodging": ["direction", "internet", "num", "parking", "type_lodging", "price_range"], 
         "attraction": ["direction", "type_attraction"]}
        self.value_keys = \
        {"direction": "\"direction\" in [\"centre\", \"east\", \"north\", \"south\", \"west\", \"special\"], for example: input: \"in the centre part of ...\" or \"close to centre of ...\" or \"nearby centre of ...\", output: \"centre\" ", 
         # "day": "week_day in: monday to sunday ", 
         "internet": "\"internet\" in [\"yes\", \"no\", \"special\"] ", 
         "num": "\"num\" is int type ", 
         "parking": "\"parking\" in [\"yes\", \"no\", \"special\"] ",
         "price_range": "\"price_range\" in [\"cheap\", \"moderate\", \"expensive\", \"special\"], for example: input: \"affordable price\" or \"low price\", output: \"cheap\" ", 
         "time": "\"clock\" is 24-hour clock format, for example: input: \"7:00 a.m\" and \"2:00 p.m\", output: \"07:00\" and \"14:00\" ",
         "type_lodging": "\"lodging_type\" in [\"hotel\", \"guest house\", \"special\"] ", 
         "type_attraction": "\"attraction_type\" in [\"architecture\", \"boat\", \"church\", \"cinema\", \"college\", \"concert hall\", \"entertainment\", \"hotspot\", \"multiple sports\", \"museum\", \"nightclub\", \"park\", \"swimming pool\", \"theatre\", \"special\"] "}
        self.kw2schema = \
        {'train':{'num_people':'train-book people',  'week_day': 'train-day', 'departure':'train-departure', 
                  'clock_leave_at':'train-leaveat', 'destination':'train-destination', 'clock_arrive_by':'train-arriveby'},
         'taxi':{'departure':'taxi-departure', 'clock_leave_at':'taxi-leaveat', 'destination':'taxi-destination', 'clock_arrive_by':'taxi-arriveby'},
         'restaurant':{'cuisine': 'restaurant-food', 'price_range': 'restaurant-pricerange', 'direction': 'restaurant-area', 
                       'num_people':'restaurant-book people', 'clock_book':'restaurant-book time', 'week_day': 'restaurant-book day', 'full_name':'restaurant-name'},
         'lodging':{'lodging_type':'hotel-type', 'price_range': 'hotel-pricerange', 'num_stars': 'hotel-stars', 'direction':'hotel-area', 'week_day':'hotel-book day', 
                  'num_people':'hotel-book people', 'num_duration':'hotel-book stay', 'internet': 'hotel-internet', 'parking': 'hotel-parking', 'full_name':'hotel-name'},
         'attraction':{'attraction_type':'attraction-type', 'direction':'attraction-area', 'full_name':'attraction-name'}}


        with open(schema_path) as f:
            self.schema_set = json.load(f)
        for d in self.kw2schema:
            for s in self.kw2schema[d]:
                assert self.kw2schema[d][s] in self.schema_set
        schema2kw = {}
        for d in self.kw2schema:
            schema2kw[d] = {}
            for k in self.kw2schema[d]:
                # schema2kw[self.kw2schema[d][k]] = k
                schema2kw[d][self.kw2schema[d][k]] = k
        self.schema2kw = schema2kw

        domain_exms = {}
        files = os.listdir(exm_dir_path)
        for file in files:
            if 'exms_' in file:
                domain = file.split('_')[-1].split('.txt')[0].strip()
                domain_exms[domain] = ""
                file_path = exm_dir_path+file
                with open(file_path, "r", encoding='utf-8') as f:  #打开文本
                    d = f.read().strip()   #读取文本
                    if d:
                        domain_exms[domain] += d
        self.domain_exms = domain_exms
        
    def get_prompt(self, data_dialogue, data_domains, prompt_path):
        # pre_state_exms_prompt = ""
        kws_dic = []
        # ps_exms_input = ""
        # ps_exms_output = "{\"agree\": {}, \"reject\": {}, \"request\": {"
        data_exms = ""
        prompt_domain = "domain in ["
        slots = []
        data_exms = []
        skip_exm = False
        for di, dm in enumerate(data_domains):
            dm = dm.replace('hotel', 'lodging')
            slots.append("slot of {} in ".format(dm)+str(list(self.kw2schema[dm].keys())).replace("\'",'\"'))
            prompt_domain+= "\"" + dm + "\"" +", "
            
            if dm in ['lodging','attraction']:
                if not skip_exm:
                    data_exms.append(self.domain_exms[dm].strip())
                    skip_exm = True
            else:
                if dm!="restaurant":
                    data_exms.append(self.domain_exms[dm])
                
            # ps_exms_input += self.pre_state_exms[dm][0]+" "
            # ps_exms_output += self.pre_state_exms[dm][1]+", "
            kws = self.key_domains[dm]
            for kw in kws:
                if kw not in kws_dic:
                    kws_dic.append(kw)
        kws_dic.sort()
        slots_text = ' \n'.join(slots)
        data_exms_text = " \n\n".join(data_exms)
        prompt_domain = prompt_domain[:-2]+"]"
        # ps_exms_input = ps_exms_input[:-1]
        # ps_exms_output = ps_exms_output[:-2]+"}}"
        # data_exms_text = data_exms_text.strip()
        kws_prompt = ""
        for kw in kws_dic:
            kws_prompt += self.value_keys[kw]+"\n"
        kws_prompt = kws_prompt.strip()
        # kso = ', '.join(kso)
        with open(prompt_path, "r", encoding='utf-8') as f:  #打开文本
            data_prompt = f.read()
        data_prompt = data_prompt.replace("[DM]",prompt_domain).replace("[KW]",kws_prompt).replace("[EXM]",data_exms_text).replace("[ST]", slots_text)
        data_prompt = data_prompt.replace("[PREDIC]",data_dialogue[0]).replace("[DIALOG]",data_dialogue[-1])

        return data_prompt


        
    def sys_filter(self, state={}, state_dic={}, user_pre={}):
        state_dic_copy = deepcopy(state_dic)
        if state_dic:
            for act in state_dic:
                if act != 'ask_for':
                    for d in state_dic[act]: 
                        if d in self.domains: 
                            type_words = {'lodging_type': ['hotel-type', 'full_name'], \
                                          'attraction_type': ['attraction-type', 'full_name'], \
                                          'cuisine': ['restaurant-food', 'full_name']}
                            for tpw in type_words: 
                                if tpw in state_dic[act][d] and type_words[tpw][1] in state_dic[act][d] and type_words[tpw][0] not in state:
                                    del state_dic[act][d][tpw] 
                                    del state_dic_copy[act][d][tpw] 
                            for s in state_dic[act][d]: 
                                if s in self.kw2schema[d]:
                                    trans_s = self.kw2schema[d][s]
                                    
                                    if trans_s in state and state[trans_s]:
                                        for vv in state_dic[act][d][s]:
                                            if vv in state[trans_s] or vv=="" or vv=="special" or \
                                            (self.schema_set[trans_s]['values']!=[] and vv not in self.schema_set[trans_s]['values']):
                                                state_dic_copy[act][d][s].remove(vv)
                                        if state_dic_copy[act][d][s]:
                                            if act != 'not_find':
                                                state[trans_s] = state_dic_copy[act][d][s]
                                        else:
                                            del state_dic_copy[act][d][s]
                                        continue
                                    
                                    if state_dic[act][d][s] not in [[], [''], [' ']] and (self.schema_set[trans_s]['entity'] == 'True'):
                                        
                                        for vv in state_dic[act][d][s]:
                                            if vv=="" or vv=="special" or \
                                            (self.schema_set[trans_s]['values']!=[] and vv not in self.schema_set[trans_s]['values']):
                                                state_dic_copy[act][d][s].remove(vv)
                                        if state_dic_copy[act][d][s]:
                                            if act != 'not_find':
                                                state[trans_s] = state_dic_copy[act][d][s]
                                        else:
                                            del state_dic_copy[act][d][s]
                                                
                                    else:
                                        del state_dic_copy[act][d][s] 
                                        
                                else:
                                    del state_dic_copy[act][d][s] 
                                    
                        else:
                            del state_dic_copy[act][d]   
        user_pre_copy = deepcopy(user_pre)
        user_pre_copy["user"]["reject"] = {}
        for d in user_pre["user"]["request"]:
            for s in user_pre["user"]["request"][d]:
                trans_s = self.kw2schema[d][s]
                if trans_s not in state or user_pre["user"]["request"][d][s] != state[trans_s]:
                    del user_pre_copy["user"]["request"][d][s]
                    
        return state, state_dic_copy, user_pre_copy
    
    def dialogdic2state(self, only_entity=False, state={}, state_dic={}):
        if state_dic:
            if 'reject' in state_dic and state_dic['reject']:
                for d in state_dic['reject']: 
                    if d in self.domains:
                        for s in state_dic['reject'][d]: 
                            if s in self.kw2schema[d]: 
                                trans_s = self.kw2schema[d][s]
                                state[trans_s]=["[DELETE]"]
                                
            for act in state_dic:
                if act == 'request':
                    for d in state_dic[act]: 
                        if d in self.domains:  
                            
                            type_words = {'lodging_type': ['hotel-type', 'full_name'], \
                                          'attraction_type': ['attraction-type', 'full_name'], \
                                          'cuisine': ['restaurant-food', 'full_name']}
                            for tpw in type_words: 
                                if tpw in state_dic[act][d] and type_words[tpw][1] in state_dic[act][d] and type_words[tpw][0] not in state:
                                    del state_dic[act][d][tpw] 
                                    
                            for s in state_dic[act][d]: 
                                if s in self.kw2schema[d]:
                                    trans_s = self.kw2schema[d][s] 
                                    
                                    if state_dic[act][d][s] and \
                                    state_dic[act][d][s][0] in [[],'','empty','null','none','special','specific','particular','certain','care','matter']: 
                                        continue
                                        
                                    if state_dic[act][d][s] and state_dic[act][d][s][0] in ['any', 'any is ok', 'uncertain']: 
                                        state[trans_s]=["dontcare"]
                                        continue
                                        
                                    if len(state_dic[act][d][s])>=2 and self.schema_set[trans_s]['values'] and \
                                    len(state_dic[act][d][s])==len(self.schema_set[trans_s]['values']):
                                        state[trans_s]=["dontcare"]
                                        continue
                                        
                                    if trans_s in state and state[trans_s]:
                                        slot_temp = state_dic[act][d][s]
                                        for vv in state_dic[act][d][s]:
                                            if vv in state[trans_s] or vv=="" or  vv=="special" or \
                                            (self.schema_set[trans_s]['values']!=[] and vv not in self.schema_set[trans_s]['values']):
                                                slot_temp.remove(vv)
                                        if slot_temp:
                                            state[trans_s] = slot_temp
                                        continue 
                                        
                                    if (self.schema_set[trans_s]['entity'] == 'True' or not only_entity) and state_dic[act][d][s] not in [[], [''], [' ']]: 
                                        slot_temp = state_dic[act][d][s]
                                        for vv in state_dic[act][d][s]:
                                            if vv=="" or vv=="special" or \
                                            (self.schema_set[trans_s]['values']!=[] and vv not in self.schema_set[trans_s]['values']):
                                                slot_temp.remove(vv)

                                        if slot_temp:
                                            state[trans_s] = slot_temp
        else:
            return state
        return state

    def state2pre(self, **slot_values): 
        prompt_state = {"user": {"reject": {}, "request": {}}} 
        for s in slot_values:
            d = s.split('-')[0].replace("hotel", "lodging")
            v = slot_values[s] 
            trans_s = self.schema2kw[d][s] 
            # if v == 'dontcare': 
            #     if d not in prompt_state["user"]["indifferent"]:
            #         prompt_state["user"]["indifferent"][d] = []
            #     prompt_state["user"]["indifferent"][d].append(trans_s)
            # else: 
            if d not in prompt_state["user"]["request"]: 
                prompt_state["user"]["request"][d] = {} 
            if v == 'dontcare': 
                v = 'any'
            prompt_state["user"]["request"][d][trans_s] = [v] 
        # state_text =  json.dumps(prompt_state, sort_keys=False, indent=None).replace('\'', '\"')
        return prompt_state

def chat_reply(data_prompt, api_key = "", api_base = "", api_organization = "", stop=['[END]']):
    if api_key:
        openai.api_key = api_key
    if api_base:
        openai.api_base = api_base
    if api_organization:
        openai.api_organization = api_organization

    # text = openai.Completion.create(
    #           model="text-davinci-003",
    #           max_tokens=256,
    #           temperature=0.0,
    #           stop=stop[0],
    #           # top_p=0.1, 
    #           prompt=data_prompt,
    #             )["choices"][0]["text"]
    # text = text.strip()+' \n'
    # return text
    
    text = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-0301",
              max_tokens=256,
              temperature=0.0,
              stop=stop,
              # top_p=0.1, 
              messages=[
                    {"role": "user", "content": data_prompt},
                ])["choices"][0]["message"]["content"]
    text = text.strip()+' \n'
    return text
