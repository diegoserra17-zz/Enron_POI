#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier #incluido depois para teste****************
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
###Escolhido o bônus no lugar de salário.
features_list = ['poi','bonus','exercised_stock_options','fraction_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset.
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers.
data_dict.pop('TOTAL',0)

### Vamos remover os pontos de dados que causam distorsão.
data_dict.pop('LOCKHART EUGENE E',0)


##Devido a quantidade de valores ausentes, vamos eliminar os "loan_advances".
for name in data_dict:
    data_dict[name].pop('loan_advances',0)

### Task 3: Create new feature(s)

def compFraction( poi_messages, all_messages ):
   ### Se poi_messages e all_messagens tiverem valores, 
   ### realizar a fração de poi_messages/all_messages e retorná-las.
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages
    return fraction


for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compFraction( from_poi_to_this_person, to_messages )
    
    data_dict[name]["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compFraction( from_this_person_to_poi, from_messages )

    data_dict[name]["fraction_to_poi"] = fraction_to_poi    


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Fonte do Classificador Utilizado:
### (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.
### StratifiedShuffleSplit.html)
### Módulo funciona apenas até a versão 0.18, sendo descontinuado na versão 0.20.
### (Aviso de descontinuação)
from sklearn.cross_validation import StratifiedShuffleSplit

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### 
X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
for train_index, test_index in sss:
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test = y[train_index], y[test_index]


# Example starting point. Try investigating other evaluation techniques!
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)