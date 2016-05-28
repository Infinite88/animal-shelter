
# coding: utf-8

# In[278]:

from __future__ import division
from __future__ import absolute_import
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import re

get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings(u'ignore')


# In[279]:

# function used to import data into dataframe and get info
def animalDF(file):
    df = pd.read_csv(file)
    print df.info()
    return df


# In[280]:

animals = animalDF(u'files/train.csv')
test = animalDF(u'files/test.csv')


# In[281]:

# Determining what features had missing data
animals.isnull().sum()


# In[282]:

print animals[u'Name'].value_counts()
print animals[u'OutcomeSubtype'].value_counts()
print animals[u'SexuponOutcome'].value_counts()
print animals[u'AgeuponOutcome'].value_counts()


# In[283]:

print animals[u'Color'].nunique()
print animals[u'Name'].nunique()
print animals[u'AgeuponOutcome'].nunique()
print animals[u'AnimalType'].nunique()
print animals[u'SexuponOutcome'].nunique()


# In[284]:

animals.head()


# In[285]:

# Splitting SexuponOutcome feature into sex and neutered features.
def sexSplit(data):
    sex = unicode(data)
    if u'Male' in sex:
        return u'Male'
    else:
        return u'Female'
    
def intactSplit(data):
    intact = unicode(data)
    if u'Neutered' in intact or u'Spayed' in intact:
        return u'Neutered'
    elif u'Intact' in intact:
        return u'Intact'
    else:
        return u"Unknown"
    
animals[u'Sex'] = animals.SexuponOutcome.apply(sexSplit)
animals[u'Neutered'] = animals.SexuponOutcome.apply(intactSplit)
test[u'Sex'] = test.SexuponOutcome.apply(sexSplit)
test[u'Neutered'] = test.SexuponOutcome.apply(intactSplit)


# In[286]:

# Dropping old feature as well as Outcomesubtype, so that each dataset matches
animals = animals.drop([u'SexuponOutcome', u'OutcomeSubtype'], axis=1)
test = test.drop(u'SexuponOutcome', axis=1)


# In[287]:

# Change Breed values to either Hybrid or Purebred
def get_mix(data):
    mix = unicode(data)
    if u'Mix' in mix or u'/' in mix:
        return u'Hybrid'
    else:
        return u'Purebred'
animals[u'Breed'] = animals.Breed.apply(get_mix)
test[u'Breed'] = test.Breed.apply(get_mix)


# In[288]:

animals[u'AgeuponOutcome'] = animals[u'AgeuponOutcome'].fillna(u'1 year')
test[u'AgeuponOutcome'] = test[u'AgeuponOutcome'].fillna(u'1 year')


# In[289]:

# Convert Color values into simple colors
def color(data):
    x = unicode(data)
    split = re.split(u'/| ', x)
    return split[0]
    
animals[u'Color'] = animals.Color.apply(color)
test[u'Color'] = test.Color.apply(color)


# In[290]:

animals.head()


# In[291]:

# Convert Name values to whether an animal has a name or not
def hasName(data):
    x = unicode(data)
    if u'nan' in x:
        return u'No Name'
    else:
        return u'Has Name'

animals[u'Name'] = animals.Name.apply(hasName)
test[u'Name'] = test.Name.apply(hasName)


# In[292]:

animals[u'Name'].value_counts()


# In[293]:

# Convert age in age in weeks
def convert_AgeuponOutcome_to_weeks(df):
    result = {}
    for k in df[u'AgeuponOutcome'].unique():
        if type(k) != type(u""):
            result[k] = -1
        else:
            v1, v2 = k.split()
            if v2 in [u"year", u"years"]:
                result[k] = int(v1) * 52
            elif v2 in [u"month", u"months"]:
                result[k] = int(v1) * 4.5
            elif v2 in [u"week", u"weeks"]:
                result[k] = int(v1)
            elif v2 in [u"day", u"days"]:
                result[k] = int(v1) / 7
                
    df[u'AgeuponOutcome'] = df[u'AgeuponOutcome'].map(result).astype(int)
                
    return df

animals = convert_AgeuponOutcome_to_weeks(animals)
test = convert_AgeuponOutcome_to_weeks(test)


# In[294]:

# Convert DateTime into five features Year,Month, Day, Hour, and Minute
def fix_date_time(df):
    def extract_field(_df, start, stop):
        return _df[u'DateTime'].map(lambda dt: int(dt[start:stop]))
    df[u'Year'] = extract_field(df,0,4)
    df[u'Month'] = extract_field(df,5,7)
    df[u'Day'] = extract_field(df,8,10)
    df[u'Hour'] = extract_field(df,11,13)
    df[u'Minute'] = extract_field(df,14,16)
    
    return df.drop([u'DateTime'], axis = 1)

animals = fix_date_time(animals)
test = fix_date_time(test)


# In[295]:

animals.info()


# In[296]:

animals.head()


# In[297]:

test.head()


# In[298]:

animals.to_csv(u'files/cleanTrain.csv', index=False)
test.to_csv(u'files/cleanTest.csv', index=False)


# In[299]:

current_palette = sns.color_palette(u"RdBu", n_colors=9)
g = sns.factorplot(data=animals, x=u'OutcomeType', hue=u'AnimalType', kind=u'count', size=6, palette=current_palette)
plt.subplots_adjust(top=.94)
plt.xlabel(u'Outcome for Animal', fontsize=18)
plt.ylabel(u'Number of Animals', fontsize=18)
g.fig.suptitle(u'Outcome for Type of Animal', fontsize=20)
g.savefig(u'images/outcome_animal.png')


# In[300]:

g = sns.factorplot(data=animals, x=u'OutcomeType', hue=u'Neutered', kind=u'count', size=6, palette=u'pastel')
plt.subplots_adjust(top=.94)
plt.xlabel(u'Outcome for Animal', fontsize=18)
plt.ylabel(u'Number of Neutered Animals', fontsize=18)
g.fig.suptitle(u'Outcome for Neutered Animals', fontsize=20)
g.savefig(u'images/outcome_neutered.png')


# In[301]:

g = sns.factorplot(data=animals, x=u'AnimalType',hue=u'Breed', kind=u'count', size=6)
plt.subplots_adjust(top=.94)
plt.xlabel(u'Type of Animal', fontsize=18)
plt.ylabel(u'Number of Breed Type', fontsize=18)
g.fig.suptitle(u'Animal Breed Type', fontsize=20)
g.savefig(u'images/animal_breed.png')


# In[302]:

g = sns.factorplot(data=animals, x=u'AnimalType', kind=u'count', palette=u'Set3', size=6)
plt.subplots_adjust(top=.94)
plt.xlabel(u'Type of Animal', fontsize=18)
plt.ylabel(u'Number of Animals', fontsize=18)
g.fig.suptitle(u'Number of Animals by Type', fontsize=20)
g.savefig(u'images/animal_type.png')

