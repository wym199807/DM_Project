import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
pd.set_option('display.max_columns',None)

age_train = pd.read_csv("age_train.csv", names=['uid','age_group'])
age_test = pd.read_csv("age_test.csv", names=['uid'])
user_basic_info = pd.read_csv("user_basic_info.csv", names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','fontSize','ct','carrier','os'])
user_behavior_info = pd.read_csv("user_behavior_info.csv", names=['uid','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'])
user_app_actived = pd.read_csv("user_app_actived.csv", names=['uid','appId'])
app_info = pd.read_csv("app_info.csv", names=['appId', 'category'])

class2id = {}
id2class = {}

def mergeBasicTables(baseTable):
    
    resTable = baseTable.merge(user_basic_info, how='left', on='uid', suffixes=('_base0', '_ubaf'))
    resTable = resTable.merge(user_behavior_info, how='left', on='uid', suffixes=('_base1', '_ubef'))
    
    cat_columns = ['city','prodName','color','carrier','os','ct']
    for c in cat_columns:
        resTable[c] = resTable[c].apply(lambda x: x if type(x)==str else str(x))
        
        sort_temp = sorted(list(set(resTable[c])))  
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        id2class['id2'+c] = dict(zip(range(1,len(sort_temp)+1), sort_temp))
        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])
        
    return resTable

train_basic = mergeBasicTables(age_train)
test_basic = mergeBasicTables(age_test)

train_basic.head()

train_basic['ramCapacity'] = train_basic['ramCapacity'].fillna(round(user_basic_info['ramCapacity'].mean()))
test_basic['ramCapacity'] = test_basic['ramCapacity'].fillna(round(user_basic_info['ramCapacity'].mean()))

train_basic['ramLeftRation'] = train_basic['ramLeftRation'].fillna(round(user_basic_info['ramLeftRation'].mean()))
test_basic['ramLeftRation'] = test_basic['ramLeftRation'].fillna(round(user_basic_info['ramLeftRation'].mean(), 2))

train_basic['romCapacity'] = train_basic['romCapacity'].fillna(round(user_basic_info['romCapacity'].mean()))
test_basic['romCapacity'] = test_basic['romCapacity'].fillna(round(user_basic_info['romCapacity'].mean()))

train_basic['romLeftRation'] = train_basic['romLeftRation'].fillna(round(user_basic_info['romLeftRation'].mean(), 2))
test_basic['romLeftRation'] = test_basic['romLeftRation'].fillna(round(user_basic_info['romLeftRation'].mean(), 2))

train_basic['fontSize'] = train_basic['fontSize'].fillna(round(user_basic_info['fontSize'].mean(), 2))
test_basic['fontSize'] = test_basic['fontSize'].fillna(round(user_basic_info['fontSize'].mean(), 2))

train_app = train_basic.merge(user_app_actived, how='left', on='uid')[['uid', 'appId']]
test_app = test_basic.merge(user_app_actived, how='left', on='uid')[['uid', 'appId']]

vectorizer = CountVectorizer(min_df=1, max_df=0.7, tokenizer=lambda x:x.split('#'))
train_app_counts = vectorizer.fit_transform(train_app['appId'])
test_app_counts = vectorizer.transform(test_app['appId'])

train_app_counts

NB = MultinomialNB()
NB.fit(train_app_counts, train_basic['age_group'])
NB_preds = NB.predict(test_app_counts)

df = pd.DataFrame({'id':test_basic['uid'],'label':NB_preds})
df.to_csv('submission.csv',index=False)
