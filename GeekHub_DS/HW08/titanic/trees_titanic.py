#!/usr/bin/env python
# coding: utf-8

# # <center>Тема 3. Обучение с учителем. Методы классификации
# ## <center>Практика. Дерево решений в задаче предсказания выживания пассажиров "Титаника". Решение

# In[37]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn import tree
from IPython.display import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# используем .dot формат для визуализации дерева
from ipywidgets import Image

get_ipython().system('pip install pydotplus')
from sklearn.tree import export_graphviz


# **Функция для формирования csv-файла посылки на Kaggle:**

# In[38]:


def write_to_submission_file(predicted_labels, out_file, train_num=891,
                    target='Survived', index_label="PassengerId"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(train_num + 1,
                                                  train_num + 1 +
                                                  predicted_labels.shape[0]),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# **Считываем обучающую и тестовую выборки.**

# In[39]:


train_df = pd.read_csv("titanic_train.csv") 
test_df = pd.read_csv("titanic_test.csv") 
train_df = pd.read_csv("titanic_train.csv") 
#test_df 32%, train_df 68%


# In[40]:


train_df


# In[41]:


y = train_df['Survived']
print(y)


# In[42]:


train_df.head()


# In[43]:


train_df.describe(include='all')


# In[44]:


test_df.describe(include='all')


# **Заполним пропуски медианными значениями.**

# In[45]:


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna('S', inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)


# **Кодируем категориальные признаки `Pclass`, `Sex`, `SibSp`, `Parch` и `Embarked` с помощью техники One-Hot-Encoding.**

# In[46]:


train_df = pd.concat([train_df, pd.get_dummies(train_df['Pclass'], 
                                               prefix="PClass"),
                      pd.get_dummies(train_df['Sex'], prefix="Sex"),
                      pd.get_dummies(train_df['SibSp'], prefix="SibSp"),
                      pd.get_dummies(train_df['Parch'], prefix="Parch"),
                     pd.get_dummies(train_df['Embarked'], prefix="Embarked")],
                     axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['Pclass'], 
                                             prefix="PClass"),
                      pd.get_dummies(test_df['Sex'], prefix="Sex"),
                      pd.get_dummies(test_df['SibSp'], prefix="SibSp"),
                      pd.get_dummies(test_df['Parch'], prefix="Parch"),
                    pd.get_dummies(test_df['Embarked'], prefix="Embarked")],
                     axis=1)


# In[47]:


train_df.drop(['Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 
               'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 
              axis=1, inplace=True)
test_df.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 
             axis=1, inplace=True)


# In[48]:


train_df


# **В тестовой выборке появляется новое значение Parch = 9, которого нет в обучающей выборке. Проигнорируем его.**

# In[49]:


train_df.shape, test_df.shape


# In[50]:


set(test_df.columns) - set(train_df.columns)


# In[51]:


test_df.drop(['Parch_9'], axis=1, inplace=True)


# In[52]:


train_df.head()


# In[53]:


test_df.head()


# ## 1. Дерево решений без настройки параметров 

# **Обучите на имеющейся выборке дерево решений (`DecisionTreeClassifier`) максимальной глубины 2. Используйте параметр `random_state=17` для воспроизводимости результатов.**

# In[54]:


# Ваш код здесь
get_ipython().system('pip install pydot')


# In[55]:


X = train_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,
                                                          random_state=17)
tree = DecisionTreeClassifier(max_depth=2, random_state=17)
tree.fit(X_train, y_train)


# **Сделайте с помощью полученной модели прогноз для тестовой выборки **

# In[56]:


tree_pred = tree.predict(X_test)
accuracy_score(y_test, tree_pred)


# **Сформируйте файл посылки и отправьте на Kaggle**

# In[57]:


# Ваш код здесь
get_ipython().system('jupyter nbconvert --to script trees_titanic.ipynb')


# <font color='red'>Вопрос 1. </font> Каков результат первой посылки (дерево решений без настройки параметров) в публичном рейтинге соревнования Titanic?
# - 0.746
# - 0.756
# - 0.766
# - 0.776

# In[58]:


0.756


# **Отобразите дерево с помощью `export_graphviz` и `dot`.**

# In[59]:


export_graphviz(tree, feature_names=['X', 'y'],
                 out_file='/small_tree.dot', filled=True)
get_ipython().system("dot -Tpng 'churn_tree.dot' -o 'churn_tree.png'")

(graph,) = pydot.graph_from_dot_file('/small_tree.dot')
graph.write_png('/small_tree.png')

from IPython.core.display import Image, display
display(Image('/small_tree.png', unconfined=True))


# In[60]:


# Ваш код здесь
from IPython.display import Image
import pydotplus
model = tree.ExtraTreeClassifier(max_depth=2, random_state=17)
model.fit(X_train, y_train)

#Use http://webgraphviz.com to visualize the graph of this file
with open("tree_classifier_depth2.txt", "w") as f:
    f = tree.export_graphviz(model, out_file=f)
#!dot -Tpng '../img/f.dot' -o 'TREE1.png'    


# ![tree]("TREE1.png") 
# 

# In[62]:


from IPython.display import Image 
from IPython.core.display import HTML 


# 
# ![Image of tree](http://localhost:8888/view/lesson08/titanic/TREE1.png) 

# <font color='red'>Вопрос 2. </font> Сколько признаков задействуются при прогнозе деревом решений глубины 2?
# - 2
# - 3
# - 4
# - 5

# In[63]:


3


# ## 2. Дерево решений с настройкой параметров 

# **Обучите на имеющейся выборке дерево решений (`DecisionTreeClassifier`). Также укажите `random_state=17`. Максимальную глубину и минимальное число элементов в листе настройте на 5-кратной кросс-валидации с помощью `GridSearchCV`.**

# In[64]:


# tree params for grid search
tree_params = {'max_depth': list(range(1, 5)), 
               'min_samples_leaf': list(range(1, 5))}

# Ваш код здесь
tree_grid = GridSearchCV(tree, tree_params,
                         cv=5, n_jobs=-1,
                        verbose=True)


# <font color='red'>Вопрос 3. </font> Каковы лучшие параметры дерева, настроенные на кросс-валидации с помощью `GridSearchCV`?
# - max_depth=2, min_samples_leaf=1
# - max_depth=2, min_samples_leaf=4
# - max_depth=3, min_samples_leaf=2
# - max_depth=3, min_samples_leaf=3

# In[65]:


tree_grid.fit(X_train, y_train)
tree_grid.best_params_, 


# <font color='red'>Вопрос 4. </font> Какой получилась средняя доля верных ответов на кросс-валидации для дерева решений с лучшим сочетанием гиперпараметров `max_depth` и `min_samples_leaf`?
# - 0.77
# - 0.79
# - 0.81
# - 0.83

# In[66]:


# tree params for grid search
tree_grid.best_score_


# **Сделайте с помощью полученной модели прогноз для тестовой выборки.**

# In[67]:


# Ваш код здесь
np.mean(cross_val_score(RandomForestClassifier(random_state=17), X_train, y_train, cv=5))
rf = RandomForestClassifier(random_state=17, n_jobs=-1).fit(X_train, y_train)
accuracy_score(y_test, rf.predict(X_test))


# **Сформируйте файл посылки и отправьте на Kaggle.**

# In[68]:


# Ваш код здесь
from IPython.display import Image 
from IPython.core.display import HTML 
Image(url= "http://my_site.com/my_picture.jpg") 


# <font color='red'>Вопрос 5. </font> Каков результат второй посылки (дерево решений с настройкой гиперпараметров) в публичном рейтинге соревнования Titanic?
# - 0.7499
# - 0.7599
# - 0.7699
# - 0.7799

# In[69]:


accuracy_score(y_test, rf.predict(X_test))


# In[74]:


export_graphviz(tree_grid.best_estimator_, feature_names=X.columns, 
    out_file='churn_tree-best.dot', filled=True)
get_ipython().system("dot -Tpng 'churn_tree-best.dot' -o 'churn_tree-best.png'")


# In[76]:


get_ipython().system('ls -l *.png')


# <img src='churn_tree-best.png'>

# ## Ссылки:

#  - <a href="https://www.kaggle.com/c/titanic">Соревнование</a> Kaggle "Titanic: Machine Learning from Disaster"
#  - <a href="https://www.dataquest.io/mission/74/getting-started-with-kaggle/">Тьюториал</a> Dataquest по задаче Kaggle "Titanic: Machine Learning from Disaster"
