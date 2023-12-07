#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:52:15 2020
@author: pathou
"""

import praw
import pandas as pd
import pytz
from datetime import datetime
from utils import *


subreddit_channel = 'politics'

reddit = praw.Reddit(
     client_id="F_vJTAGrNVQRV9JONcToBQ",
     client_secret="Qjv1Eur8u3tXlFJOY0wUjk0zJSz0LA",
     user_agent="testscript by u/fakebot3",
     username="yueyuem77",
     password="Betty0630",
     check_for_async=False
 )

print(reddit.read_only)

# def conv_time(var):
#     tmp_df = pd.DataFrame()
#     t = pd.DataFrame({"created_at": var}, index = [0])
#     tmp_df = pd.concat([tmp_df,t])
#     tmp_df.created_at = pd.to_datetime(
#         tmp_df.created_at, unit='s').dt.tz_localize(
#             'utc').dt.tz_convert('US/Eastern')
#     return datetime.fromtimestamp(var).astimezone(pytz.utc)

# def get_reddit_data(var_in):
#     import pandas as pd
#     tmp_dict = pd.DataFrame()
#     tmp_time = None
#     try:
#         t = pd.DataFrame({"created_at": conv_time(
#                                         var_in.created_utc)}, index = [0])
#         tmp_dict = pd.concat([tmp_dict, t])

#         tmp_time = tmp_dict.created_at[0]
#     except:
#         print ("ERROR")
#         pass
#     tmp_dict['msg_id']= str(var_in.id)
#     tmp_dict['author']=  str(var_in.author)
#     tmp_dict['body'] = var_in.body
#     tmp_dict['datetime'] = tmp_time

#     return tmp_dict
def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
    {'created_at': var},ignore_index=True)
    
    tmp_df.created_at = pd.to_datetime(
    tmp_df.created_at, unit='s').dt.tz_localize(
    'utc').dt.tz_convert('US/Eastern')
        
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
        var_in.created_utc)},
        ignore_index=True)
        tmp_time = tmp_dict.created_at[0]
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
    'author': str(var_in.author),
    'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict

file_path = '/Users/uumin/Documents/QMSS/f23/NLP/hw/hw4/'
model_a = read_pickle(file_path, 'my_model_a')
vec_a = read_pickle(file_path, 'vectorizer_')
pca_a = read_pickle(file_path, 'pca_a')
for comment in reddit.subreddit(subreddit_channel).stream.comments():
    
    tmp_df = get_reddit_data(comment)

    tmp_df['body_clean'] = clean_txt(tmp_df['body'])
    tmp_df['body_sw'] =  rem_sw(tmp_df['body_clean'])
    tmp_df['body_sw_stem'] = stem_fun(tmp_df['body_sw'])
    # tmp_df['body_clean'] = tmp_df['body'].apply(clean_txt)
    # tmp_df['body_sw'] = tmp_df['body_clean'].apply(rem_sw)
    # tmp_df['body_sw_stem'] = tmp_df['body_sw'].apply(stem_fun)
    
    tmp_df['body_xform'] =  vec_a.transform([tmp_df['body_sw_stem']]).toarray()
    # tmp_df['body_xform'] = pd.DataFrame(vec_a.transform(tmp_df['body_sw_stem']).toarray(), columns=vec_a.get_feature_names_out())
    tmp_df['body_pca'] =  pca_a.transform(tmp_df['body_xform'])

    tmp_df['label_prediction'] = model_a.predict(tmp_df['body_pca'])[0]
    tmp_df['score'] = model_a.predict_proba(tmp_df['body_pca'])
    
    print (tmp_df["body"])
    print('-->Corpus belongs to ', tmp_df['label_prediction'], 'with a likelihood of',
          max(tmp_df["score"][0]))
    # print ("Class Prediction: ", tmp_df["label_prediction"])
    # print ("Score: ", tmp_df["score"][0].max())
