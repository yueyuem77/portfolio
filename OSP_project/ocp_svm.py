# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import seaborn as sns
import matplotlib.pyplot as plt

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

#Utility Function
from tool.utils import *

#Path
data_path = '/Users/uumin/Documents/QMSS/f23/NLP/NLP_OCP/syllabus_5k_clean.csv'
tax_path = '/Users/uumin/Documents/QMSS/f23/NLP/NLP_OCP/bloomtax/'
out_path = '/Users/uumin/Documents/QMSS/f23/NLP/NLP_OCP/output/'

#Preprocessing of the data
df = pd.read_csv(data_path,nrows=1000)
df.head()

# #combining all learning ourcomes columns into 1
# learnoutcomes_columns = [f'extracted_sections/learning_outcomes/{i}/text' for i in range(1, 34)]

# # Concatenating the columns into one
# df['learning_outcomes'] = df.apply(lambda row: ' '.join([str(row[col]) for col in learnoutcomes_columns if pd.notna(row[col])]), axis=1)

target_columns = ['_id','field','language','Learning_Outcome']
new_df = df.copy()
new_df = new_df[target_columns]

new_column_names = {'_id':'id','field':'field_name','language':'language','Learning_Outcome':'learning_outcomes'}
new_df.rename(columns=new_column_names, inplace=True)

# new_df['learning_outcomes'].replace('', np.nan, inplace=True)
new_df = new_df.dropna(subset = ['learning_outcomes'])



#%%

#save the data to a new csv file for later reference
new_df.to_csv('clean_workng_data.csv', index=False)

#%%
#clean the data using the clean_txt function from utils
new_df['body'] = new_df.learning_outcomes.apply(clean_txt)

#%%
#Load the label: Tax data
tax_cat = ['analyze', 'apply','create', 'evaluate' ,'remember', 'understand']
dataframes = {}
for cat in tax_cat:
    cat_path = tax_path + cat +'.csv'
    dataframes[cat] = pd.read_csv(cat_path)

##Create df for each category
analyze_df = dataframes['analyze']
apply_df = dataframes['apply']
create_df = dataframes['create']
evaluate_df = dataframes['evaluate']
remember_df = dataframes['remember']
understand_df = dataframes['understand']

##Create a Word-Weight Dictionary
analyze_dict = analyze_df.set_index('word')['weight'].to_dict()
apply_dict = apply_df.set_index('word')['weight'].to_dict()
create_dict = create_df.set_index('word')['weight'].to_dict()
evaluate_dict = evaluate_df.set_index('word')['weight'].to_dict()
remember_dict = remember_df.set_index('word')['weight'].to_dict()
understand_dict = understand_df.set_index('word')['weight'].to_dict()


##Calculate Category Score for Each Syllabus
new_df['analyze_score'] = new_df['body'].apply(lambda x: sum(analyze_dict.get(word, 0) for word in x.split()))
new_df['apply_score'] = new_df['body'].apply(lambda x: sum(apply_dict.get(word, 0) for word in x.split()))
new_df['create_score'] = new_df['body'].apply(lambda x: sum(create_dict.get(word, 0) for word in x.split()))
new_df['evaluate_score'] = new_df['body'].apply(lambda x: sum(evaluate_dict.get(word, 0) for word in x.split()))
new_df['remember_score'] = new_df['body'].apply(lambda x: sum(remember_dict.get(word, 0) for word in x.split()))
new_df['understand_score'] = new_df['body'].apply(lambda x: sum(understand_dict.get(word, 0) for word in x.split()))

category_columns = ['analyze_score', 'apply_score', 'create_score', 'evaluate_score', 'remember_score', 'understand_score']
new_df['predicted_category'] = new_df[category_columns].idxmax(axis=1)
new_df['predicted_category'] = new_df['predicted_category'].str.replace('_score', '')
#%%
#Vectorization
cv = CountVectorizer(ngram_range=(1, 1))
xform_data = pd.DataFrame(cv.fit_transform(new_df.body).toarray())
xform_data.columns = cv.get_feature_names_out()
xform_data.index = new_df.predicted_category

write_pickle(cv, out_path, 'xform_cv')
#%%
# Model Training
#Split the train test set


features = pd.concat([xform_data.reset_index(drop=True), new_df[category_columns].reset_index(drop = True)], axis=1)

target = new_df['predicted_category']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=123)

clf = RandomForestClassifier(random_state=123)

clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
# Accuracy, Precision, Recall, and F1 Score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))

# Detailed classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


new_df['model_category'] = clf.predict(features)
#%%
#save the model in pickle
write_pickle(clf, out_path, 'rf_model')










