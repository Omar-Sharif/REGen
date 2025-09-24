import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from pprint import pprint
from datetime import datetime
import copy
import pickle
from ast import literal_eval
import re, string

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_huggingface import HuggingFaceEndpoint
try:
    from google.colab import files  # optional
except Exception:  # pragma: no cover
    files = None

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import torch
import copy

def read_json_file(name):
    with open(name, 'r') as f:
        data = json.load(f)
        return data

#just take the path and dataset name
def get_processed_predictions_and_schema(path, model_name, prompt_type, dataset_name, version):
    data_path = os.path.join(path, 'Result')
    data = read_json_file(os.path.join(data_path, dataset_name, f'{model_name}-{prompt_type}-processed-predictions-{dataset_name}-v{version}.json'))
    schema_path = os.path.join(path, 'Data')
    event_schema = read_json_file(os.path.join(schema_path, dataset_name, f'{dataset_name}_schema.json'))
    return data, event_schema

#get files for complex matching
def get_predictions_for_complex_matching(path, model_name, prompt_type, dataset_name, version):
    data_path = os.path.join(path, 'Result')
    data = read_json_file(os.path.join(data_path, dataset_name, 
                                       f'{model_name}-{prompt_type}-after-exact-relaxed-match-predictions-{dataset_name}-v{version}.json'))
    schema_path = os.path.join(path, 'Data')
    event_schema = read_json_file(os.path.join(schema_path, dataset_name, f'{dataset_name}_schema.json'))
    return data, event_schema

#get files for results
def get_predictions_for_results(path, model_name, dataset_name, prompt_type='zero-shot', version=0):
    data_path = os.path.join(path, 'Result')
    data = read_json_file(os.path.join(data_path, dataset_name, f'{model_name}-{prompt_type}-after-complex-match-predictions-{dataset_name}-v{version}.json'))
    schema_path = os.path.join(path, 'Data')
    event_schema = read_json_file(os.path.join(schema_path, dataset_name, f'{dataset_name}_schema.json'))
    return data, event_schema

#--------Level-1 (Exact Match)---------functions
#function for doing exact-match
def doing_exact_match(predictions, unique_roles):
    new_pred_dictionary = []

    for role in unique_roles:
        print(f"-----{role}------")
        cnt = 0
        for dt in predictions:
            if(dt['role']!=role): continue #skiping the roles that does not match
            new_dt = copy.deepcopy(dt)
            normalized_actual_labels = dt['initial-ground-truth']  #this labels and predictions are normalized
            normalized_predictions = dt['initial-predictions']

            #needed for sanity printing. can be commented later
            # initial_labels = copy.deepcopy(normalized_actual_labels)
            # initial_predictions = copy.deepcopy(normalized_predictions)

            em_pair = []
            for p in normalized_predictions[:]:
                if(len(normalized_predictions)<=0 or len(normalized_actual_labels)<=0):
                    break #if any of the list becomes zero then no need for comparison
                for g in normalized_actual_labels[:]:
                    if(len(normalized_predictions)<=0 or len(normalized_actual_labels)<=0):
                        break #if any of the list becomes zero then no need for comparison
                    if p==g: #if we find a match
                      normalized_actual_labels.remove(g) ##removing this item from the ground-truth list as it predicted correctly
                      normalized_predictions.remove(p) ##removing from prediction list as it predicted correctly.
                      em_pair.append((p, g))
                      print(role, "||", p,"||", g)

            #this is for sanity printing// can be commented later
            # if(len(em_pair)>0):
            #     print(len(em_pair), em_pair)
            #     cnt += len(em_pair)
            #     print("GT:", len(normalized_actual_labels),"---", normalized_actual_labels, "---",  initial_labels)
            #     print("PD:",len(normalized_predictions),"---", normalized_predictions,"---", initial_predictions)

            #new list of ground-truth and predictions after exact match
            new_dt['after-exact-match-ground-truth'] = normalized_actual_labels
            new_dt['after-exact-match-predictions'] = normalized_predictions
            new_dt['exact-match-pairs'] = em_pair

            ##creating new dictionary with exact-match
            new_pred_dictionary.append(new_dt)

        #print(role, cnt)
    return new_pred_dictionary

#--------Level-2 (Relaxed Match)---------functions
def clac_sbert_score(text1, text2):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    cosine_score = util.cos_sim(embeddings1, embeddings2)
    #print(cosine_score.item())
    return cosine_score.item()

#semantic-score calculation for all pairs for each role
def calculating_semantic_score(prediction_dictionary, unique_roles):
    new_pred_dictionary = []
    cnt = 0
    for role in unique_roles:
        print(f"-----{role}------")
        cnt = 0
        for dt in prediction_dictionary:
            if(dt['role']!=role): continue #skiping the roles that does not match
            new_dt = copy.deepcopy(dt)

            #taking the groun-truth-arguments after exact match
            normalized_actual_labels = copy.deepcopy(dt['after-exact-match-ground-truth'])  #this labels and predictions are normalized
            normalized_predictions = copy.deepcopy(dt['after-exact-match-predictions'])

            lst_of_pairs = []
            print("GT:", normalized_actual_labels)
            print("PD:", normalized_predictions)
            #calculating pairwise matching score between all the ground-truth and predicted arguments
            for p in normalized_predictions:
                if p =='null': continue #if a prediction is 'null' we do not have to do any relaxed match
                if(len(normalized_predictions)<=0 or len(normalized_actual_labels)<=0):
                        break #if any of the list is zero then no need for comparison
                for g in normalized_actual_labels:
                      sbert_score = clac_sbert_score(p, g)
                      print(p, "||",  g, "||", sbert_score)
                      lst_of_pairs.append(((p, g), sbert_score))
                      cnt+=1

            ##keeping these scores for future
            new_dt['relaxed-match-sim-score-all-pairs'] = copy.deepcopy(lst_of_pairs)
            new_pred_dictionary.append(new_dt)
        print(cnt)
    return new_pred_dictionary

#calculating relaxed-match score on different threshold
def relaxed_match_thresholding(predictions, unique_roles, threshold):
    new_pred_dictionary = []
    for role in unique_roles:
        print(f"-----{role}-----{threshold}----")
        cnt = 0
        for dt in predictions:
            if(dt['role']!=role): continue #skiping the roles that does not match
            new_dt = copy.deepcopy(dt)

            #taking the groun-truth-arguments after exact match
            #initialization of individual ground-truth and predictions
            initial_labels = copy.deepcopy(dt['after-exact-match-ground-truth'])  #this labels and predictions are normalized
            initial_pred = copy.deepcopy(dt['after-exact-match-predictions'])

            lst_of_pairs = copy.deepcopy(dt['relaxed-match-sim-score-all-pairs']) #similarity score between all pairs of ground-truth and predictions
            rm_pair = []

            for value in lst_of_pairs:
                pred, gt, score = value[0][0], value[0][1], value[1]
                if(score>=threshold):
                    print(pred,"||", gt, "||", score)
                    rm_pair.append((pred, gt))
                    if(pred in initial_pred): initial_pred.remove(pred)
                    if(gt in initial_labels): initial_labels.remove(gt)

            #new ground-truth and predictions list
            new_dt[f'after-relaxed-match-{threshold}-ground-truth'] = copy.deepcopy(initial_labels)
            new_dt[f'after-relaxed-match-{threshold}-predictions'] = copy.deepcopy(initial_pred)

            #new relaxed-match-pairs
            new_dt[f'relaxed-match-{threshold}-pairs'] = copy.deepcopy(rm_pair)

            #creating new dictionary with relaxed-match
            new_pred_dictionary.append(new_dt)
    return new_pred_dictionary

#take predictions and outputs a dictionary with exact and relaxed-match on different thresholds
def getting_exact_relaxed_match_predictions_dictionary(path, model_name, prompt_type, dataset_name, version):
    initial_predictions, event_schema = get_processed_predictions_and_schema(path, model_name,
                                                                                  prompt_type, dataset_name, version)
    unique_roles = list({d['role'] for d in initial_predictions if 'role' in d})

    #predicitions dictionary after exact-match
    after_em_predictions_dictionary = copy.deepcopy(doing_exact_match(copy.deepcopy(initial_predictions), unique_roles))
    #print(after_em_predictions_dictionary[0].keys())

    #after semantic-score calculation prediction dictionary
    after_ssc_predictions_dictionary = calculating_semantic_score(copy.deepcopy(after_em_predictions_dictionary), unique_roles)
    #print(after_ssc_predictions_dictionary[0].keys())

    #relaxed-matching on different threshold-levels
    thresholds = [0.95, 0.85, 0.75]
    after_relaxed_match_predictions_dictionary = copy.deepcopy(after_ssc_predictions_dictionary)
    for threshold in thresholds:
        after_relaxed_match_predictions_dictionary = copy.deepcopy(relaxed_match_thresholding(after_relaxed_match_predictions_dictionary,
                                                                                              unique_roles, threshold))
    print(after_relaxed_match_predictions_dictionary[0].keys())
    return copy.deepcopy(after_relaxed_match_predictions_dictionary)


#--------Level-3 (Complex Match)---------functions
#create the chain for LLM to invoke for complex-matching
#this is final judge prompt after trail and error process
def pair_matching_prompt_chain(model):
    prompt_template = PromptTemplate(
        input_variables=['x', 'y', 'context'],
        template = '''
        ## Instruction ##
        Find whether text-1 and text-2 are semantically similar or not based on the context provided.

        ## Context ##
        {context}

        ## Texts ##
        text-1: {x}
        text-2: {y}

        Are text-1 and text-2 semantically similar even though they are structurally different? Return "yes" if they are similar and "no" otherwise. Do not provide any extra description.
        '''
    )

    prompt_chain = prompt_template | model | StrOutputParser()
    return prompt_chain

def complex_pair_matching(x, y, context, model):
      input_dict = {
          'x': x,
          'y': y,
          'context': context
      }
      prompt_chain = pair_matching_prompt_chain(model) ##creating the prompt chain

      while True: ## to get rid of model overload error
          try:
              output = prompt_chain.invoke(input_dict)
              break
          except Exception as e:
              print(e)
              time.sleep(3)

      if "yes" in output.lower():
          return "yes"
      else: return "no"


#saving those pairs that falls under complex match
def getting_complex_match_pairs(predictions, unique_roles, judge_model_name, threshold):
    new_pred_dictionary = []
    for role in unique_roles:
        print(f"-----{role}-----")
        cnt = 0
        for dt in predictions:
            if(dt['role']!=role): continue #skiping the roles that does not match
            new_dt = copy.deepcopy(dt)

            #taking the groun-truth and predicted arguments after exact and relaxed match
            initial_labels = copy.deepcopy(dt[f'after-relaxed-match-{threshold}-ground-truth'])
            initial_pred = copy.deepcopy(dt[f'after-relaxed-match-{threshold}-predictions'])

            lst_of_pairs = copy.deepcopy(dt[f'complex-match-{judge_model_name}-all-pairs']) #similarity score between all pairs of ground-truth and predictions
            cm_pair = []

            for value in lst_of_pairs:
                pred, gt, context, matching_output = value[0][0], value[0][1], value[0][2], value[1]
                if(matching_output == 'yes'):
                    print(pred,"||", gt, "||", matching_output)
                    cm_pair.append((pred, gt, context))
                    if(pred in initial_pred): initial_pred.remove(pred)
                    if(gt in initial_labels): initial_labels.remove(gt)


            #new ground-truth and predictions list
            new_dt[f'after-complex-match-{judge_model_name}-ground-truth'] = copy.deepcopy(initial_labels)
            new_dt[f'after-complex-match-{judge_model_name}-predictions'] = copy.deepcopy(initial_pred)

            #new complex-match-pairs
            new_dt[f'complex-match-{judge_model_name}-pairs'] = copy.deepcopy(cm_pair)

            #creating new dictionary with complex-match
            new_pred_dictionary.append(new_dt)
    return new_pred_dictionary

#complex-match finding for all the argument paris for each role
def doing_complex_matching(prediction_dictionary, unique_roles, judge_model, judge_model_name, threshold):
    new_pred_dictionary = []
    cnt = 0
    for role in unique_roles:
        print(f"-----{role}------")
        for dt in prediction_dictionary:
            if(dt['role']!=role): continue #skiping the roles that does not match
            new_dt = copy.deepcopy(dt)

            #taking the groun-truth and predicted arguments after exact and relaxed match
            normalized_actual_labels = copy.deepcopy(dt[f'after-relaxed-match-{threshold}-ground-truth'])
            normalized_predictions = copy.deepcopy(dt[f'after-relaxed-match-{threshold}-predictions'])
            context = copy.deepcopy(dt['context'])

            lst_of_pairs = []
            # print("GT:", normalized_actual_labels)
            # print("PD:", normalized_predictions)
            #find the complex matching output between all the ground-truth and predicted arguments
            for p in normalized_predictions:
                if (p=='null'): continue #if a prediction is 'null' or emnpty string we do not have to do any complex match
                if(len(normalized_predictions)<=0 or len(normalized_actual_labels)<=0):
                        break #if any of the list is zero then no need for comparison
                for g in normalized_actual_labels:
                      matching_output = complex_pair_matching(p, g, context, judge_model)
                      print(p, "||",  g, "||", matching_output)
                      lst_of_pairs.append(((p, g, context), matching_output))
                      cnt+=1

            ##keeping these scores for future
            new_dt[f'complex-match-{judge_model_name}-all-pairs'] = copy.deepcopy(lst_of_pairs)
            new_pred_dictionary.append(new_dt)
            cnt += len(lst_of_pairs) #number of inference for that role in each sample

    return new_pred_dictionary, cnt

def getting_complex_match_predictions_dictionary(predictions, unique_roles, judge_model_access_string,
                                                 judge_model_name, model_origin, threshold):

    if model_origin == 'openai':
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError('OPENAI_API_KEY is not set')
        judge_model = ChatOpenAI(openai_api_key=openai_key, temperature=0.0, model=judge_model_access_string)
    elif model_origin == 'hug_api':
        hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not hf_token:
            raise ValueError('HUGGINGFACEHUB_API_TOKEN is not set')
        judge_model = HuggingFaceEndpoint(repo_id= judge_model_access_string, temperature=0.01,
                                    max_new_tokens=10,  huggingfacehub_api_token=hf_token)
    elif model_origin == 'anthropic':
        pass

    #getting the dictionary after generating complex-match pairs and whether it is yes or no
    new_pred_dictionary, inference_cnt = doing_complex_matching(copy.deepcopy(predictions),
                                                 unique_roles, judge_model, judge_model_name, threshold)

    print(f"Number of inferences: {inference_cnt}")
    return new_pred_dictionary



##getting counts and scores
def getting_role_wise_scores(predictions, unique_roles, threshold, judge_model,
                             dataset_name, judgement_deviation_score):
    '''
    Function for calculating exact, relaxed, complex, and judgement-aligned matching socres.
    '''
    print(len(predictions))
    results = {}
    for role in unique_roles:
        #print(f"-----{role}----")
        gt_cnt, pd_cnt, na_e, np_e, na_r, np_r, na_c, np_c = 0, 0, 0, 0, 0, 0, 0, 0
        em_cnt, rm_cnt, cm_cnt = 0, 0, 0
        for dt in predictions:
            if(dt['role']!=role): continue #skiping the roles that does not match

            #sanity_print(dt)
            gt_cnt += len(dt['initial-ground-truth'])
            pd_cnt += len(dt['initial-predictions'])

            #number of arguments predicted correct from the ground-truth and predicted list of arguments under exact-match
            na_e += len(dt['initial-ground-truth']) - len(dt['after-exact-match-ground-truth'])
            np_e += len(dt['initial-predictions']) - len(dt['after-exact-match-predictions'])

            #number of arguments predicted correcly from the ground-truth and predited list of argument under relaxed-match
            na_r += (len(dt['after-exact-match-ground-truth']) -
                    len(dt[f'after-relaxed-match-{threshold}-ground-truth']))
            np_r += (len(dt['after-exact-match-predictions']) -
                    len(dt[f'after-relaxed-match-{threshold}-predictions']))

            #number of arguments predicted correcly from the ground-truth and predited list of argument under complex-match
            # Check if complex matching was performed
            if f'after-complex-match-{judge_model}-ground-truth' in dt:
                na_c += (len(dt[f'after-relaxed-match-{threshold}-ground-truth']) -
                        len(dt[f'after-complex-match-{judge_model}-ground-truth']))
                np_c += (len(dt[f'after-relaxed-match-{threshold}-predictions']) -
                        len(dt[f'after-complex-match-{judge_model}-predictions']))
            else:
                # If complex matching wasn't performed, complex match results are 0
                na_c += 0
                np_c += 0

            #number of exact, relaxed, and complex match counts this might have corefferences
            em_cnt += len(dt['exact-match-pairs'])
            rm_cnt += len(dt[f'relaxed-match-{threshold}-pairs'])
            # Check if complex matching was performed
            if f'complex-match-{judge_model}-pairs' in dt:
                cm_cnt += len(dt[f'complex-match-{judge_model}-pairs'])
            else:
                cm_cnt += 0

        epsilon = 1e-10
        total_cnt = em_cnt + rm_cnt + cm_cnt
        print(role)
        print(f"GT: {gt_cnt}, PD: {pd_cnt}, EM:{na_e, np_e}, RM:{na_r, np_r}, CM: {na_c, np_c}, EM+RM+CM: {total_cnt}")

        #exact-match precision, recall, f1-score calculation
        em_precision = np_e/max(pd_cnt, epsilon)
        em_recall = na_e/max(gt_cnt, epsilon)
        em_f1 = (2 * em_precision * em_recall)/max((em_precision + em_recall), epsilon)

        #relaxed-match precision, recall, f1-score calculation
        rm_precision = (np_e + np_r)/max(pd_cnt, epsilon)
        rm_recall = (na_e + na_r)/max(gt_cnt, epsilon)
        rm_f1 = (2 * rm_precision * rm_recall)/max((rm_precision + rm_recall), epsilon)

        #complex-match precision, recall, f1-score calculation
        cm_precision = (np_e + np_r + np_c)/max(pd_cnt, epsilon)
        cm_recall = (na_e + na_r + na_c)/max(gt_cnt, epsilon)
        cm_f1 = (2 * cm_precision * cm_recall)/max((cm_precision + cm_recall), epsilon)

        #judgement-aligned matching (JAM) score claculation
        E_rm = judgement_deviation_score[dataset_name][f'relaxed-match-{threshold}-threshold']
        # Check if complex matching was performed
        if f'complex-match-{judge_model}-judge' in judgement_deviation_score[dataset_name]:
            E_cm = judgement_deviation_score[dataset_name][f'complex-match-{judge_model}-judge']
            jam_precision = (np_e + ((1-E_rm)*np_r) + ((1-E_cm) * np_c))/max(pd_cnt, epsilon)
            jam_recall = (na_e + ((1-E_rm)*na_r) + ((1-E_cm) *na_c))/max(gt_cnt, epsilon)
        else:
            # If complex matching wasn't performed, JAM only uses exact and relaxed matching
            jam_precision = (np_e + ((1-E_rm)*np_r))/max(pd_cnt, epsilon)
            jam_recall = (na_e + ((1-E_rm)*na_r))/max(gt_cnt, epsilon)
        jam_f1 = (2 * jam_precision * jam_recall)/max((jam_precision + jam_recall), epsilon)

        results[role] = {
            'exact-match-precision': round(em_precision*100, 2),
            'exact-match-recall': round(em_recall*100, 2),
            'exact-match-f1': round(em_f1*100, 2),

            'relaxed-match-precision': round(rm_precision*100, 2),
            'relaxed-match-recall': round(rm_recall*100, 2),
            'relaxed-match-f1': round(rm_f1*100, 2),

            'complex-match-precision': round(cm_precision*100, 2),
            'complex-match-recall': round(cm_recall*100, 2),
            'complex-match-f1': round(cm_f1*100, 2),

            'jam-precision': round(jam_precision*100, 2),
            'jam-recall': round(jam_recall*100, 2),
            'jam-f1': round(jam_f1*100, 2),

            'ground-truth-count': gt_cnt,
            'prediction-count': pd_cnt,
            'exact-match-count': em_cnt,
            'relaxed-match-count': rm_cnt,
            'complex-match-count': cm_cnt,
        }
    return results