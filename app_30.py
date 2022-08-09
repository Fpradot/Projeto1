#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # Tarefa - Agrupamento hierárquico

# Neste exercício vamos usar a base [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://doi.org/10.1007/s00521-018-3523-0).
# 
# A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" 
# 
# Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

# |Variavel                |Descrição          | 
# |------------------------|:-------------------| 
# |Administrative          | Quantidade de acessos em páginas administrativas| 
# |Administrative_Duration | Tempo de acesso em páginas administrativas | 
# |Informational           | Quantidade de acessos em páginas informativas  | 
# |Informational_Duration  | Tempo de acesso em páginas informativas  | 
# |ProductRelated          | Quantidade de acessos em páginas de produtos | 
# |ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
# |BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
# |ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
# |PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
# |SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
# |Month                   | Mês  | 
# |OperatingSystems        | Sistema operacional do visitante | 
# |Browser                 | Browser do visitante | 
# |Region                  | Região | 
# |TrafficType             | Tipo de tráfego                  | 
# |VisitorType             | Tipo de visitante: novo ou recorrente | 
# |Weekend                 | Indica final de semana | 
# |Revenue                 | Indica se houve compra ou não |
# 
# \* variávels calculadas pelo google analytics

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from gower import gower_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


# In[30]:


df = pd.read_csv('online_shoppers_intention.csv')


# In[31]:


df.head()


# In[32]:


df.Revenue.value_counts(dropna=False)


# ## Análise descritiva
# 
# Faça uma análise descritiva das variáveis do escopo.
# 
# - Verifique a distribuição dessas variáveis
# - Veja se há valores *missing* e caso haja, decida o que fazer
# - Faça mais algum tratamento nas variáveis caso ache pertinente

# In[33]:


df.info()


# In[34]:


df.nunique(axis=0)


# In[35]:


df['SpecialDay'].unique()


# In[36]:


df['SpecialDay'].value_counts()


# In[37]:


df['Month'].unique()


# In[38]:


df['Month'].value_counts()


# In[39]:


df['Weekend'].value_counts()


# In[40]:


fig, axis = plt.subplots(4, 2, figsize=(20,20))

sns.histplot(data=df, x = "Administrative", discrete=True, ax=axis[0,0])
axis[0, 0].set_title("Administrative Count")

sns.histplot(data=df, x = "Administrative_Duration", ax=axis[0,1])
axis[0, 1].set_title("Administrative_Duration Count")

sns.histplot(data=df, x = "Informational", discrete=True, ax=axis[1,0])
axis[1, 0].set_title("Informational Count")

sns.histplot(data=df, x = "Informational_Duration", bins=50, ax=axis[1,1])
axis[1, 1].set_title("Informational_Duration Count")

sns.histplot(data=df, x = "ProductRelated", discrete=True, ax=axis[2,0])
axis[2, 0].set_title("ProductRelated Count")

sns.histplot(data=df, x = "ProductRelated_Duration", ax=axis[2,1])
axis[2, 1].set_title("ProductRelated_Duration Count")

sns.histplot(data=df, x = "SpecialDay", ax=axis[3,0])
axis[3, 0].set_title("SpecialDay Count")

sns.histplot(data=df, x = "Month", ax=axis[3,1])
axis[3, 1].set_title("Month Count")


# In[41]:


sns.countplot(data=df, x = 'Weekend').set_title("Weekend Count")


# In[ ]:





# ## Variáveis de agrupamento
# 
# Liste as variáveis que você vai querer utilizar. Essa é uma atividade importante do projeto, e tipicamente não a recebemos pronta. Não há resposta pronta ou correta, mas apenas critérios e a sua decisão. Os critérios são os seguintes:
# 
# - Selecione para o agrupamento variáveis que descrevam o padrão de navegação na sessão.
# - Selecione variáveis que indiquem a característica da data.
# - Não se esqueça de que você vai precisar realizar um tratamento especial para variáveis qualitativas.
# - Trate adequadamente valores faltantes.

# In[42]:


variaveis = ['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
             'SpecialDay', 'Month', 'Weekend']
variaveis_qtd = ['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']
variaveis_cat = ['SpecialDay', 'Month', 'Weekend']


# In[43]:


df_pad = pd.DataFrame(StandardScaler().fit_transform(df[variaveis_qtd]), columns = df[variaveis_qtd].columns)


# In[44]:


df_pad.head()


# In[45]:


df_pad[variaveis_cat] = df[variaveis_cat]


# In[46]:


df2 = pd.get_dummies(df_pad[variaveis].dropna(), columns=variaveis_cat)
df2.head()


# In[47]:


df2.columns.values


# ## Número de grupos
# 
# Nesta atividade vamos adotar uma abordagem bem pragmática e avaliar agrupamentos hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do diretor da empresa. 
# 
# *Atenção*: Cuidado se quiser fazer o dendrograma, pois com muitas observações ele pode ser mais complicado de fazer, e dependendo de como for o comando, ele pode travar o *kernell* do seu python.

# In[48]:


vars_cat = [True if x in {'SpecialDay_0.0', 'SpecialDay_0.2',
       'SpecialDay_0.4', 'SpecialDay_0.6', 'SpecialDay_0.8',
       'SpecialDay_1.0', 'Month_Aug', 'Month_Dec', 'Month_Feb',
       'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov',
       'Month_Oct', 'Month_Sep', 'Weekend_False', 'Weekend_True'} else False for x in df2.columns]


# In[49]:


df2.shape


# In[50]:


distancia_gower = gower_matrix(df2, cat_features=vars_cat)


# In[51]:


gdv = squareform(distancia_gower,force='tovector')


# In[52]:


gdv.shape


# In[53]:


Z = linkage(gdv, method='complete')


# In[54]:


df2['grupos_3'] = fcluster(Z, 3, criterion='maxclust')
df2.grupos_3.value_counts()


# In[55]:


df3 = df.join(df2['grupos_3'], how='left')


# In[56]:


df3['grupos_3'].replace({1:"grupo_1", 3:"grupo_3", 2:"grupo_2"}, inplace=True)


# In[57]:


sns.boxplot(data=df3, y='grupos_3', x='BounceRates')


# In[58]:


pd.crosstab(df3.Revenue, df3.grupos_3)


# In[59]:


df2['grupos_4'] = fcluster(Z, 4, criterion='maxclust')
df2.grupos_4.value_counts()


# In[60]:


df3 = df.join(df2['grupos_4'], how='left')
df3['grupos_4'].replace({1:"grupo_1", 3:"grupo_3", 2:"grupo_2", 4:"grupo_4"}, inplace=True)


# In[61]:


sns.boxplot(data=df3, y='grupos_4', x='BounceRates')


# In[62]:


pd.crosstab(df3.Revenue, df3.grupos_4)


# ## Avaliação dos grupos
# 
# Construa os agrupamentos com a técnica adequada que vimos em aula. Não se esqueça de tratar variáveis qualitativas, padronizar escalas das quantitativas, tratar valores faltantes e utilizar a distância correta.
# 
# Faça uma análise descritiva para pelo menos duas soluções de agrupamentos (duas quantidades diferentes de grupos) sugeridas no item anterior, utilizando as variáveis que estão no escopo do agrupamento.
# - Com base nesta análise e nas análises anteriores, decida pelo agrupamento final. 
# - Se puder, sugira nomes para os grupos.

# In[63]:


df2['grupos_6'] = fcluster(Z, 6, criterion='maxclust')
df2.grupos_6.value_counts()


# In[64]:


df3 = df.join(df2['grupos_6'], how='left')
df3['grupos_6'].replace({1:"grupo_1", 3:"grupo_3", 2:"grupo_2", 4:"grupo_4", 5:"grupo_5", 6:"grupo_6"}, inplace=True)


# In[65]:


sns.boxplot(data=df3, y='grupos_6', x='BounceRates')


# In[66]:


df2['grupos_2'] = fcluster(Z, 2, criterion='maxclust')
df2.grupos_2.value_counts()


# In[67]:


df3 = df.join(df2['grupos_2'], how='left')
df3['grupos_2'].replace({1:"grupo_1", 2:"grupo_2"}, inplace=True)


# In[68]:


sns.boxplot(data=df3, y='grupos_2', x='BounceRates')


# In[69]:


pd.crosstab(df3.Revenue, df3.grupos_2)


# In[70]:


df2['grupos_3'] = fcluster(Z, 3, criterion='maxclust')
df3 = df.join(df2['grupos_3'], how='left')
df3['grupos_3'].replace({1:"grupo_1", 3:"grupo_3", 2:"grupo_2"}, inplace=True)


# In[71]:


pd.crosstab(df3.Revenue, df3.grupos_3, normalize='columns')


# In[72]:


sns.pairplot(df2[['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'grupos_3']], hue='grupos_3')


# Analisando os resultados aparentemente 3 grupos é um número bom, ao aumentarmos o número de grupos o algoritmo cria grupos muito pequenos que não influenciam muito em decisões, muito provavelmente agrupando outliers. Escolhendo o melhor agrupamento como sendo o de 3 grupos analisaremos mais profundamente o mesmo.

# In[73]:


fig, axis = plt.subplots(3, 1, figsize=(15,15))

sns.countplot(data=df3, x = "SpecialDay", hue='grupos_3', ax=axis[0])

sns.countplot(data=df3, x = "Month", hue='grupos_3', ax=axis[1])

sns.countplot(data=df3, x = "Weekend", hue='grupos_3', ax=axis[2])


# - Observando o pairplot observamos que os grupos não foram separados por nenhuma das informações ali presentes, pois não percebemos nenhum padrão, na verdade o que parece ter maior peso é a variável Weekend, o que sugere que talvez a análise de 2 grupos apenas já seja útil, pois os grupos 2 e 3 parecem ser bem dificeis de se distinguir.

# In[74]:


df2['grupos_2'] = fcluster(Z, 2, criterion='maxclust')
df3 = df.join(df2['grupos_2'], how='left')
df3['grupos_2'].replace({1:"grupo_1", 2:"grupo_2"}, inplace=True)


# In[75]:


pd.crosstab(df3.Revenue, df3.grupos_2, normalize='columns')


# In[76]:



fig, axis = plt.subplots(3, 1, figsize=(15,15))

sns.countplot(data=df3, x = "SpecialDay", hue='grupos_2', ax=axis[0])

sns.countplot(data=df3, x = "Month", hue='grupos_2', ax=axis[1])

sns.countplot(data=df3, x = "Weekend", hue='grupos_2', ax=axis[2])


# In[ ]:





# ## Avaliação de resultados
# 
# Avalie os grupos obtidos com relação às variáveis fora do escopo da análise (minimamente *bounce rate* e *revenue*). 
# - Qual grupo possui clientes mais propensos à compra?

# - Na análise de 2 grupos o grupo mais propenso a compra é o grupo_1 (maior porcentagem de Revenue = True e menores valores de Bounce Rate) que indica clientes que acessaram páginas aos finais de semana. (Na análise de 3 grupos chegamos a uma conclusão semelhante, o grupo de clientes que acessaram páginas aos finais de semana são mais propensos a compra).

# In[ ]:




