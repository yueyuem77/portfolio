# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:25:09 2023

@author: pathouli
"""

def jd(str_in_a, str_in_b):
    set_a = set(str_in_a.split())
    set_b = set(str_in_b.split())
    the_int = set_a.intersection(set_b)
    the_union = set_a.union(set_b)
    jd_t = len(the_int) / len(the_union)
    return jd_t

def clean_txt(str_in):
    import re
    tmp_clean_t = re.sub("[^A-Za-z']+", " ", str_in
                         ).lower().strip()
    return tmp_clean_t

def file_reader(path_in):
    f = open(path_in, "r", encoding="UTF-8")
    #readlines, readline, read
    tmp = f.read()
    tmp = clean_txt(tmp)
    f.close()    
    return tmp

def file_walker(data_path_in):
    import os
    import pandas as pd
    t_data = pd.DataFrame()
    for root, dirs, files in os.walk(data_path_in, topdown=False):
       for name in files:
           path_t = root + "/" + name
           try:
               txt_t = file_reader(path_t)
               if len(txt_t) > 0:
                   label_t = root.split("/")[-1:][0]
                   tmp_data = pd.DataFrame(
                       {"body": txt_t, "label": label_t}, index=[0])
                   t_data = pd.concat([t_data, tmp_data], ignore_index=True)
           except:
               print (path_t)
               pass
    return t_data

def fun_wrd_cnt(pd_in, col_name_in):
    import collections
    fun_word = dict()
    for topic in pd_in.label.unique():
        wrd_fun = pd_in[pd_in.label == topic]
        str_cat = wrd_fun[col_name_in].str.cat(sep=" ")
        fun_word[topic] = collections.Counter(str_cat.split())
    return fun_word

def rem_sw(str_in):
    import nltk
    sw = nltk.corpus.stopwords.words('english')    
    sent = [word for word in str_in.lower(
        ).split() if word not in sw]
    sent = ' '.join(sent)
    return sent

def stem_fun(str_in):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()    
    sent = [ps.stem(word) for word in str_in.lower(
        ).split()]
    sent = ' '.join(sent)
    return sent

def token_count(str_in):
    tmp = str_in.lower().split()
    return len(tmp)

def token_count_unique(str_in):
    tmp = set(str_in.lower().split())
    return len(tmp)

def read_pickle(path_in, file_name):
    import pickle
    tmp_o = pickle.load(
        open(path_in + file_name + ".pk", 'rb'))
    return tmp_o

def write_pickle(obj_in, path_in, file_name):
    #https://docs.python.org/3/library/pickle.html
    import pickle
    # dump information to that file
    pickle.dump(obj_in, open(
        path_in + file_name + '.pk', 'wb'))

def init():
    #https://pypi.org/project/torpy/
    #pip install torpy
    from torpy import TorClient
    hostname = 'ifconfig.me'  # It's possible use onion hostname here as well
    with TorClient() as tor:
        # Choose random guard node and create 3-hops circuit
        with tor.create_circuit(3) as circuit:
            # Create tor stream to host
            with circuit.create_stream((hostname, 80)) as stream:
                # Now we can communicate with host
                stream.send(b'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % hostname.encode())
                recv = stream.recv(1024)
    return 0

def my_scraper(tmp_url_in):
    #https://pypi.org/project/beautifulsoup4/
    #pip install beautifulsoup4
    from bs4 import BeautifulSoup
    import requests
    import re
    import time
    tmp_text = ''
    try:
        content = requests.get(tmp_url_in, timeout=10)
        soup = BeautifulSoup(content.text, 'html.parser')

        tmp_text = soup.findAll('p') 

        tmp_text = [word.text for word in tmp_text]
        tmp_text = ' '.join(tmp_text)
        tmp_text = re.sub('\W+', ' ', re.sub('xa0', ' ', tmp_text))
    except:
        print("Connection refused by the server..")
        print("Let me sleep for 5 seconds")
        print("ZZzzzz...")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        pass
    return tmp_text

def fetch_urls(query_tmp, cnt):
    #now lets use the following function that returns
    #URLs from an arbitrary regex crawl form google
    #pip install pyyaml ua-parser user-agents fake-useragent
    import requests
    from bs4 import BeautifulSoup
    import re 
    headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    query = '+'.join(query_tmp.split())
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(cnt)
    print (google_url)
    response = requests.get(google_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'egMi0 kCrYT'})

    links = []
    links_t = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            links_t.append(r.a.get('href'))
            for link in links_t:
                x = re.split("&ved", (re.split("url=", link)[1]))
                if x[0] not in links:
                    links.append(x[0])
            if link != '':# and title != '' and description != '': 
                links.append(link['href'])
        # Next loop if one element is not present
        except:
            continue  
    return links
 
def write_crawl_results(my_query, the_cnt_in):
    #let use fetch_urls to get URLs then pass to the my_scraper function 
    import re
    import pandas as pd 

    tmp_pd = pd.DataFrame()       
    for q_blah in my_query:
        #init()
        the_urls_list = fetch_urls(q_blah, the_cnt_in)

        for word in the_urls_list:
            tmp_txt = my_scraper(word)
            if len(tmp_txt) != 0:
                try:
                    tmp_txt = clean_txt(tmp_txt)
                    t = pd.DataFrame(
                        {'body': tmp_txt, 'label': re.sub(
                            ' ', '_', q_blah)}, index=[0])
                    tmp_pd = pd.concat([tmp_pd, t], ignore_index=True)
                    print (word)
                except:
                    pass
    return tmp_pd

def vec_fun(df_in, path_in, name_in, m_in, n_in, label_in):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import pandas as pd
    if name_in == "tfidf":
        cv = TfidfVectorizer(ngram_range=(m_in, n_in))
    elif name_in == "vec":
        cv = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        print ("Hey pick a valid transformer vec or tfidf")
    xform_data_t = pd.DataFrame(cv.fit_transform(df_in).toarray()) #be careful dense matrix takes up space
    xform_data_t.columns = cv.get_feature_names_out()
    xform_data_t.index = label_in
    write_pickle(cv, path_in, name_in)
    return xform_data_t

def extract_embeddings_pre(df_in, out_path_i, name_in):
    #https://code.google.com/archive/p/word2vec/
    #https://pypi.org/project/gensim/
    #pip install gensim
    #name_in = 'models/word2vec_sample/pruned.word2vec.txt'
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    word2vec_sample = str(find(name_in))
    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False)
    # word_dict = my_model.key_to_index
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb" ))
    return tmp_data, my_model_t

def domain_train(df_in, path_in, name_in):
    #domain specific
    import pandas as pd
    import gensim
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        for word in var:
            try:
                tmp_arr.append(list(model.wv.get_vector(word)))
            except:
                pass
        tmp_arr
        return np.mean(np.array(tmp_arr), axis=0)
    model = gensim.models.Word2Vec(df_in.str.split())
    model.save(path_in + 'body.embedding')
    #call up the model
    #load_model = gensim.models.Word2Vec.load('body.embedding')
    model.wv.similarity('fish','river')
    tmp_data = pd.DataFrame(df_in.str.split().apply(get_score))
    return tmp_data, model

def chi_fun(df_in, label_in, k_in, path_out, name_in):
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(
        df_in, label_in))
    feat_index = feat_sel.get_support(indices=True)
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, path_out, name_in)
    return dim_data

def cos_fun(df_in_a, df_in_b, label_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    cos_sim = pd.DataFrame(cosine_similarity(df_in_a, df_in_b))
    cos_sim.index = label_in
    cos_sim.columns = label_in
    return cos_sim

def model_fun(df_in, label_in, sel_in, t_in, o_in):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd
    if sel_in == "rf":
        model = RandomForestClassifier(random_state=123)
    elif sel_in == "nb":
        model = GaussianNB()
    elif sel_in == "dt":
        model = DecisionTreeClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=t_in, random_state=42)
    
    model.fit(X_train, y_train)
    write_pickle(model, o_in, sel_in)
    y_pred = model.predict(X_test)
    y_pred_proba = pd.DataFrame(model.predict_proba(X_test))
    y_pred_proba.columns = model.classes_
    try:
        fi = pd.DataFrame(model.feature_importances_)
        fi.index = model.feature_names_in_
        fi.columns = ["score"]
        num_fi = fi[fi.score != 0]
        fi.to_csv(o_in + "fi.csv", index=True)
    except:
        print (sel_in, "does NOT support feature importance")
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "F1", None]
    print (metrics)
    return model

def pca_fun(df_in, exp_var, path_in, name_in):
    from sklearn.decomposition import PCA
    pca_fun = PCA(n_components=exp_var)
    pca_data = pca_fun.fit_transform(df_in)
    exp_var = sum(pca_fun.explained_variance_ratio_)
    print ("exp var", exp_var)
    write_pickle(pca_fun, path_in, name_in)
    return pca_data

def grid_fun(df_in, lab_in, ts, grid_in, cv_i, name_in, o_path):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=ts, random_state=42)
    
    model = RandomForestClassifier(random_state=123)
    cv = GridSearchCV(model, grid_in, cv=cv_i)
    cv.fit(X_train, y_train)
    
    print ("best perf", cv.best_score_)
    print ("best params", cv.best_params_)
    
    model = RandomForestClassifier(**cv.best_params_, random_state=123)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, lab_in, test_size=ts, random_state=42)
    
    model.fit(X_train, y_train)
    write_pickle(model, o_path, name_in)
    y_pred = model.predict(X_test)
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "F1", None]
    return model