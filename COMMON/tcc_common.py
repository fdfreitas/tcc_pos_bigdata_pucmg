#!/usr/bin/env python
# coding: utf-8

# Trabalho de Conclusao de Curso
# Pos-graduação em Ciencia de Dados e Big Data (2020) - PUC Minas
# Fabio Daros de Freitas

#
#  Common use artifacts and Jupyter Notebooks stuff
#

import builtins as __builtin__
import time
from datetime import datetime, timedelta
from IPython.display import Markdown, display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
print(f'Pandas version......: {pd.__version__}')

import numpy as np
print(f'Numpy version.......: {np.__version__}')

import sklearn as sk
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, LeaveOneOut
print(f'Scikit-learn version: {sk.__version__}')

import seaborn as sns
print(f'Seaborn version.....: {sns.__version__}')
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import statsmodels as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm_api
print(f'Stats models version: {sm.__version__}')

import matplotlib as mpl
print(f'Matplotlib version..: {mpl.__version__}')
import matplotlib.pyplot as plt 
plt.rc("font", size=14)


#-------------------------------------------------------------------------------
# Classes and general functions
#-------------------------------------------------------------------------------

class Dict(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

class ETime():
    def __init__(self):
        self.start = datetime.now()
    def check(self):
        e = datetime.now() - self.start
        print(f'Elapsed time: {e}')      
      
def print(*args, **kwargs):
    '''
    print() override to treat behavorial kw
    '''
    verbose = kwargs.pop('verbose', True)
    if not verbose:
        return
    ret = __builtin__.print(*args, **kwargs)
    return ret   
    
def print_md(string, verbose=True):
    if verbose:
        display(Markdown(string)) 
    
    
#-------------------------------------------------------------------------------
# Pandas dataframe functions
#-------------------------------------------------------------------------------

def df_display(df, name=None, rows=2):
    if name is not None:
        print(f'{name}:')
    max_rows = pd.options.display.max_rows
    print(df.shape)
    if hasattr(df, 'columns'):
        print(df.columns)    
    print(df.dtypes)
    pd.options.display.max_rows = rows
    display(df)
    pd.options.display.max_rows = max_rows
    
# Print df as LaTeX table
def df_latex(df, dec=2,  **kwargs):    
    def num(x):
        x = round(x, dec)
        return f'\\num{{{x}}}'        
    
    s  = '\\begin{table}[TABPOS]\n'
    s += '\\centering\n'    
    s += '\\small\n'        
    s += df.to_latex(float_format=num, escape=False, **kwargs)
    s += '\\caption{}\n'    
    s += '\\label{}\n'
    s += '\\end{table}\n'    
    s = s.replace('_', '\\_')
    s = s.replace('\\\\', '\\\\%')    
    print(s)
        
    
#
# dtype functions
#
def df_metric_dtypes(): 
    #np.number
    return ['float16', 'float32', 'float64']

def df_category_dtypes(): 
    return ['int32', 'int64']

#
# Counting functions
#
def df_count_rows_duplicated(df):     
    if 'cnpj8' in df.columns and 'ano_mes' in  df.columns:
        return df.duplicated(['cnpj8','ano_mes']).sum()
    return df.duplicated().sum()

def df_count_rows_nan(df): 
    return df.shape[0] - df.dropna().shape[0]

def df_count_rows_metric_zero(df): 
    zero = 0
    a = df.select_dtypes(include=df_metric_dtypes())
    if not a.empty:
        b = df[a.eq(0).any(axis=1)]
        if not b.empty:
            zero = b.shape[0]
    return zero
def df_count_rows_metric_negative(df): 
    neg = 0
    a = df.select_dtypes(include=df_metric_dtypes())
    if not a.empty:
        b = df[a.lt(0).any(axis=1)]
        if not b.empty:
            neg = b.shape[0]
    return neg

def df_count_columns_binary(df, column): 
    s =0
    n = 0
    if column in df:        
        s = len(df[df[column] == 1])
        n = len(df[df[column] == 0])        
    return s, n

#
# Rows functions
#

#
# Columns functions
#

def df_metric_dtypes(): 
    #np.number
    return ['float16', 'float32', 'float64']

def df_category_dtypes(): 
    return ['int32', 'int64']


def df_columns_metric(df): 
    a = df.select_dtypes(include=df_metric_dtypes())
    return a.columns

def df_columns_metric_zero(df): 
    cols = []
    a = df.select_dtypes(include=df_metric_dtypes())
    for c in a.columns:
        n = a[c].eq(0).sum(axis=0)
        if n > 0:
            cols.append(f'{c}({n}) ')
    return cols

#
# Profile functions
#
def df_profile(df, name=''):
    print_md(f'**Sumário do DataFrame:** {name}')    
    l      = df.shape[0]
    c      = df.shape[1]
    ldup   = df_count_rows_duplicated(df)
    lnan   = df_count_rows_nan(df)    
    cnpj8  = len(df['cnpj8'].unique()) if 'cnpj8' in df else 0
    periodo=  f'{df["ano_mes"].min()} a {df["ano_mes"].max()}' if 'ano_mes' in df else '-'    
    neg   = df_count_rows_metric_negative(df)
    zero = df_count_rows_metric_zero(df)
    zcols = df_columns_metric_zero(df)    
    maed_s, maed_n = df_count_columns_binary(df, 'maed_dctf_prox_mes') 
    
    print(f'-Linhas({l}): duplicadas={ldup} nulas={lnan} metricas zero={zero} metricas negativo={neg}')
    print(f'-Colunas({c}): colunas com alguma metrica zero={"".join(zcols)}')    
    print(f'-cnpj8 únicos={cnpj8} período={periodo}')
    tt = maed_s + maed_n
    if tt == 0:
        s = '-'
        n = '-'
        r = '-'
    else:
        s = f'{maed_s} ({100*maed_s/tt:.2f}%)'
        n = f'{maed_n} ({100*maed_n/tt:.2f}%)' 
        r = f'1:{int(1/(maed_s/maed_n))}' if maed_n > 0 else f'1:-'
    print(f'-maed_dctf_prox_mes: Sim[1]={s}  Não[0]={n}  ratio(S:N)={r})')    
    df_profile_maed(df)
    print(df.dtypes)
    print_md('---')
    
    
# MAED specific profile
def df_profile_maed(df):
    if not ('maed_dctf' in df and 'maed_dctf_prox_mes' in df):
        return    
    f00 = (df["maed_dctf"] == 0) & (df["maed_dctf_prox_mes"] == 0)
    f01 = (df["maed_dctf"] == 0) & (df["maed_dctf_prox_mes"] == 1)
    f10 = (df["maed_dctf"] == 1) & (df["maed_dctf_prox_mes"] == 0)
    f11 = (df["maed_dctf"] == 1) & (df["maed_dctf_prox_mes"] == 1)
    f0x = (df["maed_dctf"] == 0)    
    f1x = (df["maed_dctf"] == 1)        
    fx0 = (df["maed_dctf_prox_mes"] == 0)    
    fx1 = (df["maed_dctf_prox_mes"] == 1)        
    
    m00  =  len(df[f00])
    m01  =  len(df[f01])
    m10  =  len(df[f10])
    m11  =  len(df[f11])
    m0x  =  len(df[f0x])    
    m1x  =  len(df[f1x])        
    mx0  =  len(df[fx0])    
    mx1  =  len(df[fx1])        
    
    print('-Profile MAED: m_ij (i==maed_dctf, j=maed_dctf_prox_mes)')
    print(f'\tm_00={m00:6d}\tm_01={m01:6d}\t\tm_0x={m0x:6d}')
    print(f'\tm_10={m10:6d}\tm_11={m11:6d}\t\tm_1x={m1x:6d}')
    print('')
    print(f'\tm_x0={mx0:6d}\tm_x1={mx1:6d}')    
    
    print('')
    return f00, f01, f10, f11

def df_profile_maed_plot(df, width=940, height=1024, dpi=100, log=False, fname=None):
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    #fig.suptitle(f'Ocorrências maed_dcft', fontsize=16)        
    plt.subplots_adjust(left=0.1, bottom=-0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)
    #plt.xticks(fontsize=10)  # for xticks
    #plt.yticks(fontsize=10) # for yticks    
        
    axes = fig.subplots(2, 1, sharex=False, gridspec_kw={'height_ratios': [1, 1]})
    ax1, ax2 = axes

    ax = ax1
    if log:
        ax.set_yscale('log')
    ax.set_title(f'Ocorrências maed_dcft', fontsize=16)                
    ax.set_xlabel('maed_dctf')               
    ax.set_ylabel('contagem', fontsize=14)                       
    df['maed_dctf'].value_counts().plot(ax=ax, kind='bar', fontsize=14, rot=0) # , color=['blue', 'red'])
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.0001, p.get_height() * 1.0001))    

    ax = ax2
    if log:
        ax.set_yscale('log')
    ax.set_title(f'Ocorrências maed_dctf_prox_mes', fontsize=16)                
    ax.set_xlabel('maed_dctf_prox_mes')           
    ax.set_ylabel('contagem', fontsize=14)                       
    df['maed_dctf_prox_mes'].value_counts().plot(ax=ax, kind='bar', fontsize=14, rot=0) # , color=['blue', 'red'])
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.0001, p.get_height() * 1.0001))
    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')        

#
# Changing and augmentation functions
#

def df_clean(df, name='', drop_metrica_negativo=True, drop_metrica_zero=False, fill_zeros=False):
    '''Remove negativos e zerados apenas das colunas das metricas float'''
    print(f'df_clean: {name} total linhas={df.shape[0]}')    
          
    n = df_count_rows_duplicated(df)
    if n > 0:
        if 'cnpj8' in df.columns and 'ano_mes' in  df.columns:
            print(f'Removendo linhas duplicadas [cnpj8, ano_mes]({n})...')            
            df.drop_duplicates(['cnpj8','ano_mes'],keep= 'last', inplace=True)
        else:
            print(f'Removendo linhas duplicadas({n})...')
            df.drop_duplicates(inplace=True)
            
    n = df_count_rows_nan(df)
    if n > 0:
        print(f'Removendo linhas nulas({n})...')              
        df.dropna(inplace=True)
    
    n = df_count_rows_metric_negative(df)    
    if n > 0 and drop_metrica_negativo:    
        print(f'Removendo linhas com metricas com valores negativos({n})...')
        a = df.select_dtypes(include=df_metric_dtypes())        
        if not a.empty:            
            b = a.lt(0).any(axis=1)
            if not b.empty:
                df.drop(df[b].index, inplace=True)                            
                
    n = df_count_rows_metric_zero(df)
    if n > 0 and drop_metrica_zero:    
        print(f'Removendo linhas com metricas com valores zero({n})...')
        a = df.select_dtypes(include=df_metric_dtypes())        
        if not a.empty:            
            b = a.eq(0).any(axis=1)
            if not b.empty:
                df = df.drop(df[b].index, inplace=True)      
          
    n = df_count_rows_metric_zero(df)
    if n > 0 and fill_zeros:    
        print(f'Preenchendo metricas zero({n})...')          
        a = df.select_dtypes(include=df_metric_dtypes())        
        cols = a.columns
        df[cols] = df[cols].replace(0, df[cols].mean(skipna=True, axis=0))  
          
    print(f'total linhas final={df.shape[0]}')                    
    return df

def df_augment(df, remove_colinear=True):
    '''Augment dataframe columns with:
    - rbc : receita bruta calculada
    - ct  : carga tributaria
    - rec : receitas
    - desp: despesas
    - maed_dctf_total : quantidade de vezes que o contribuinte recebeu MAED DCTF
    
    Remove colinear columns:
    deb, vndas, crd, cpras
    
    '''
    df = df.copy()
    df['rbc'] = df.apply(df_column_apply_rbc, axis=1)
    df['ct']  = df.apply(df_column_apply_ct, axis=1)    
    #df['rec']  = df.deb + df.vndas
    #df['desp'] = df.crd + df.cpras

    df = df.sort_values(['cnpj8', 'ano_mes'])
    df['maed_dctf_total'] = (df['maed_dctf']).groupby(df['cnpj8']).cumsum()
    #df = df.astype({'maed_dctf_total': np.float64})

    # Final dataset columns: must have target at last column
    cols = ['cnpj8',
            'ano_mes',
            'dctf',
            'pis_cfs',            
            'cprb',
            #'rb',
            'crd',
            'deb',
            'cpras',
            'vndas',
            'rbc',
            'ct',
            #'rec',
            #'desp',
            'maed_dctf',
            'maed_dctf_total',
            'maed_dctf_prox_mes']

    if remove_colinear:
        #cols.remove('rb')        
        #cols.remove('deb')
        #cols.remove('crd')
        #cols.remove('cpras')        
        #cols.remove('vndas') 
        pass

    df = df[cols]
    df = df_clean(df) # protect nan    
    # debug
    df_display(df)
    df.describe()
    #df.maed_dctf_prox_mes.describe()
    return df


# Columns changing and augmentation functions

def df_column_create_ano_mes(df):
    df['ano_mes'] = df['ano'] + '-' + df['mes']
    df.drop(['ano', 'mes'], axis=1, inplace=True)
                 
# apply func para criacao da coluna maed_dctf
def df_column_apply_maed_dctf(maed_dict, row):    
    cnpj8 = row.cnpj8
    ano_mes = row.ano_mes
    key = (cnpj8, ano_mes)    
    return 1 if key in maed_dict else 0

# apply func para criacao da coluna maed_dctf_prox_mes
def df_maed_dict(df):
    df = df.set_index(['cnpj8', 'ano_mes'], inplace=False)
    return df.T.to_dict()

def df_column_apply_maed_dctf_prox_mes(maed_dict, row):    
    cnpj8 = row.cnpj8
    ano_mes = ano_mes_next(row.ano_mes)
    key = (cnpj8, ano_mes)    
    return 1 if key in maed_dict else 0

def df_column_apply_rbc(row):      
    '''Maior valor entre EFD Reeceita Bruta e NFe Vendas, quando ambos > 0, ou valor Efinancera Debitos'''
    v = max(row.rb, row.vndas) if row.rb > 0.0 or row.vndas > 0.0 else row.deb
    return v if v >= 0 else np.nan

def df_column_apply_ct(row):      
    '''ct = dctf / rbc'''
    v = row.dctf / row.rbc if row.rbc > 0 else 0
    return v


# Auxiliary functions

# returns next ano_mes from current ano_mes
def ano_mes_next(am):
    aml = am.split('-')
    a = int(aml[0])
    m = int(aml[1])
    a = a+1 if m == 12 else a
    m = 1  if m == 12 else m+1
    return(f'{a}-{m:02d}')     

# returns previous ano_mes from current ano_mes
def ano_mes_previous(am):
    aml = am.split('-')
    a = int(aml[0])
    m = int(aml[1])
    a = a-1 if m == 1 else a
    m = 12  if m == 1 else m-1
    return(f'{a}-{m:02d}')     


#-------------------------------------------------------------------------------
# Training support functions
#-------------------------------------------------------------------------------

# Select inputs and outputs for model training applying preprocessing
# TO BE DEPRECATED because using model_cross_validation()
def train_test_make_sets(df, name='', pre_proc='', test_size=0.25):
    print_md(f'**Cria conjuntos treino-teste:** {name}')    

    # Dropa colunas que nao sao metricas nem categorias
    inp = df.select_dtypes(include=df_category_dtypes() + df_metric_dtypes()).drop('maed_dctf_prox_mes', axis=1).copy()    
        
    # pre-processamento
    if pre_proc == 'none':
        pass
    elif pre_proc == 'norm_min_max':
        cols = df.select_dtypes(include=df_metric_dtypes()).columns       
        inp[cols] = (inp[cols]-inp[cols].min())/(inp[cols].max()-inp[cols].min())
        
    elif pre_proc == 'norm_standard':
        cols = df.select_dtypes(include=df_metric_dtypes()).columns
        inp[cols] = (inp[cols]-inp[cols].mean())/inp[cols].std()        
    else:
        print_md(f'**ERRO** parametro pre_proc({pre_proc}) inválido')
    
    tgt = df['maed_dctf_prox_mes'].copy()

    # Cria os conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(inp, tgt, test_size=0.205, random_state=0)

    y_train = y_train.astype('int32')
    y_test  = y_test.astype('int32')    
    
    # printa estatisticas dos sets
    num_maed_s = len(y_train == 1)
    num_maed_n = len(y_train == 0)
    num_data   = num_maed_s + num_maed_n
    per_maed_s = 100. * num_maed_s/num_data
    per_maed_n = 100. * num_maed_n/num_data
    ratio_maed_sn = num_maed_s/num_maed_n
    print(f'Preprocessamento: {pre_proc}')    
    print(f'Linhas do treino: {num_maed_s:7d}')
    print(f'MAED=1          : {num_maed_s:7d} ({per_maed_s:5.2f}%)')
    print(f'MAED=0          : {num_maed_n:7d} ({per_maed_n:5.2f}%)')
    print(f'Ratio MAED S/N  : {ratio_maed_sn:.2f}')
    print('x_train:')
    display(x_train.describe())
    df_display(x_train)
    print('y_train:')    
    display(y_train.describe())
    df_display(y_train)    
    print('')
    num_maed_s = len(y_test == 1)
    num_maed_n = len(y_test  == 0)
    num_data   = num_maed_s + num_maed_n
    per_maed_s = 100. * num_maed_s/num_data
    per_maed_n = 100. * num_maed_n/num_data
    ratio_maed_sn = num_maed_s/num_maed_n
    print(f'Linhas do teste : {num_maed_s:7d}')
    print(f'MAED=1          : {num_maed_s:7d} ({per_maed_s:5.2f}%)')
    print(f'MAED=0          : {num_maed_n:7d} ({per_maed_n:5.2f}%)')
    print(f'Ratio MAED S/N  : {ratio_maed_sn:.2f}')
    print('x_test:')
    display(x_test.describe())
    df_display(x_test)    
    print('y_test:')
    display(y_test.describe())
    df_display(y_test)    
    print_md('---')
    return x_train, x_test, y_train, y_test

# Get inputs and target from dataframe
def dataset_input_and_target(df):
    inp = df.select_dtypes(include=df_category_dtypes() + df_metric_dtypes()).drop('maed_dctf_prox_mes', axis=1).copy()
    tgt = df['maed_dctf_prox_mes'].copy()
    return inp, tgt

# Model cross validation 
def model_cross_validation(model, df, K, fold_data, train_model, **kwargs):
    '''
    Client application must implement the following functions:

    - df may be a datafraee or already hold the folds dict {'trn':[folds]. 'tst':[folds}}

    - fold_data(df, trn_ids, tst_ids)
    receives current fold training and testing indices of df dataframe,
    and returns trn_data and tst_data (in suitable format) to be used by train_model()
    
    - train_model(model, trn_data, tst_data, fold=fold, **kwargs)
    receives model current fold trn_data and tst_data, and returns 
    trn_mts and tst_mts fold metrics PredResult objects for aggregating
    '''
    
    cv_type = f'{K}-fold' if K > 0 else 'leave-one-out'
    print_md(f'**Model {cv_type} Cross Validation:**')
        
    # df measures
    if isinstance(df, dict):     # df may hold the folds dict {'trn':[folds]. 'tst':[folds}}
        #rows_df = [sum(len(i)) for i in df.trn]
        #rows_df = [sum(len(i)) for i in df.trn]        
        rows_df = len(df.trn[0]) + len(df.tst[0]) # get full data number of rows
        rows = list(range(rows_df)) # generate a list of indexes to folds.split()
    else:
        rows_df = df.shape[0] # get full data number of rows
        rows = df.index.tolist()    # generate a list of indexes to folds.split()
        
    metrics  = []
    
    # Define the K-fold Cross Validator
    if K == 0:
        folds = LeaveOneOut()
    else:
        folds = KFold(n_splits=K, shuffle=True, random_state=10)

    # Cross Validation model evaluation
    trn_metrics = PredResult(name=f'Folds trn', index_name='fold')
    tst_metrics = PredResult(name=f'Folds tst', index_name='fold')
    
    for fold, (trn_ids, tst_ids) in enumerate(folds.split(rows)):

        print_md('\n---\n')
        print_md(f'**Fold {fold+1}**')

        rows_trn, rows_tst = len(trn_ids), len(tst_ids)        
        
        
        print(f'trn_ids={rows_trn} ({100*rows_trn/rows_df:.2f}%)', end=' ')
        print(f'tst_ids={rows_tst} ({100*rows_tst/rows_df:.2f}%)')              
        
        trn_data, tst_data = fold_data(df, trn_ids, tst_ids, fold=fold, **kwargs)  
        
        trn_mts, tst_mts = train_model(model, trn_data, tst_data, fold=fold, **kwargs)
        
        print(f'Fold {fold+1} performance and last test set prediction result:')
        trn_mts.display_metrics()
        tst_mts.prediction_summary()

        # aggreagte fold result
        trn_metrics.latex = trn_mts.latex
        trn_metrics.append(trn_mts.get(row_id=fold+1))
        
        tst_metrics.latex = tst_mts.latex        
        tst_metrics.append(tst_mts.get(row_id=fold+1))        
        
    print_md('\n---\n')        
    print_md(f'**Folds overall performance:**')
    trn_metrics.describe_metrics()
    #display(trn_metrics.get_history())    
    trn_metrics.display_metrics_history()    
    tst_metrics.describe_metrics()
    #display(tst_metrics.get_history()) 
    tst_metrics.display_metrics_history()

#-------------------------------------------------------------------------------
# Predictions evalaution classes and functions
#-------------------------------------------------------------------------------

def pred_discretize(y):
    y[y >  0.5] = 1
    y[y <= 0.5] = 0
    return y


class PredResult():

    #metrics_names = {  'false_positive_rate'       : 'False Positive Rate (Type I error) - FPR',
                       #'false_negative_rate'       : 'False Negative Rate (Type II error)- FNR',
                       #'true_negative_rate'        : 'True Negative Rate (Specificity): TNR', 
                       #'negative_predictive_value' : 'Negative Predictie Value - NPV',
                       #'false_discovery_rate'      : 'False Discovery Rate - FDR',
                       #'recall'                    : 'True Positive Rate (Recall/Sensibility) - TPR/Recall',
                       #'precision'                 : 'Positive Predictive Value (Precision) - PPV/Precision',
                       #'accuracy'                  : 'Accuracy'}
    
    metrics_names = {  'FPR'       : 'False Positive Rate (Type I error) - FPR',
                       'FNR'       : 'False Negative Rate (Type II error)- FNR',
                       'TNR'       : 'True Negative Rate (Specificity): TNR', 
                       'NPV'       : 'Negative Predictive Value - NPV',
                       'FDR'       : 'False Discovery Rate - FDR',
                       'Recall'    : 'True Positive Rate (Recall/Sensibility) - TPR/Recall',
                       'Precision' : 'Positive Predictive Value (Precision) - PPV/Precision',
                       'Accuracy'  : 'Accuracy'}
    
    def __init__(self, name='', index_name='', latex=False):
    
        self.name      = name
        self.pp        = f'{self.name} - ' if self.name != '' else '' # print prtefix
        self.latex     = latex # Output describe and history also a LaTeX version
    
        # initialize metrics dataframe        
        c = Dict((m, float()) for m in self.metrics_names.keys())
        self.df_metrics = pd.DataFrame(c, index=[])
        if index_name != '':
            self.df_metrics.index.name = index_name
        self.y_true = None
        self.y_pred = None
        
    def update(self, y_true, y_pred, row_id=None):
        
        if y_true is None or y_pred is None:
            return
        
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # metrics calculation
        
        # Confusion matrix: (C_ij: observations in group i predicted as group j)
        cm = confusion_matrix(self.y_true, self.y_pred) 
        
        #print(f'cm[0][0] = {cm[0][0]}')
        #print(f'cm[0][1] = {cm[0][1]}')        
        #print(f'cm[1][0] = {cm[1][0]}')
        #print(f'cm[1][1] = {cm[1][1]}')        
        
        cm_res = cm.ravel()
        
        # not binary classification result...
        if len(cm_res) != 4:
            return
        tn, fp, fn, tp = cm_res
        row = dict()  # Pandas dislike Dict()...
    
        def div(a, b):            
            return a/b if b != 0 else 0
    
        #if name is not None:
        #row['name'] = name
    
        #row['false_positive_rate']       = div( fp,   fp + tn )
        #row['false_negative_rate']       = div( fn,   tp + fn )
        #row['true_negative_rate']        = div( tn,   tn + fp )
        #row['negative_predictive_value'] = div( tn,   tn + fn )
        #row['false_discovery_rate']      = div( fp,   tp + fp )
        #row['recall']                    = div( tp,   tp + fn )
        #row['precision']                 = div( tp,   tp + fp )
        #row['accuracy']                  = div( tp + tn,   tp + fp + fn + tn )
        
        
        row['FPR']       = div( fp,   fp + tn )
        row['FNR']       = div( fn,   tp + fn )
        row['TNR']       = div( tn,   tn + fp )
        row['NPV']       = div( tn,   tn + fn )
        row['FDR']       = div( fp,   tp + fp )
        row['Recall']    = div( tp,   tp + fn )
        row['Precision'] = div( tp,   tp + fp )
        row['Accuracy']  = div( tp + tn,   tp + fp + fn + tn )
                
        if row_id is None:
            self.df_metrics = self.df_metrics.append(row, ignore_index=True)
        else:
            ser = pd.Series(row)
            ser.name = row_id
            self.df_metrics = self.df_metrics.append(ser, ignore_index=False)            

    def append(self, df):
        self.df_metrics = self.df_metrics.append(df, ignore_index=False)

    def get(self, func=np.mean, row_id=None):          
        ser = self.df_metrics.apply(func, axis=0)
        if row_id is not None:
           ser.name = row_id 
        return ser

    def get_history(self, iloc=None):
        if iloc == None:
            return self.df_metrics if len(self.df_metrics) >  0 else None
            
        return self.df_metrics.iloc[iloc] if len(self.df_metrics) >  0 else None

    def display_metrics_history(self):
        print(f'{self.pp}Metrics history:')
        display(self.df_metrics)
        if self.latex:
            df_latex(self.df_metrics)
        

    def display_metrics(self, func=np.mean, T=True):
        
        ag = {np.mean : 'avg',
              np.sum  : 'sum',
              np.min  : 'min',              
              np.max  : 'max'}                
        
        print_md(f'**{self.pp}Metrics values ({ag.get(func, "(agg. function not found)")}) values:**')
        df = pd.DataFrame(self.get(func), columns = ['value'])
        if T:
            display(df.T)
        else:
            display(df)

    def describe_metrics(self):
        print(f'{self.pp}Metrics summary:')        
        display(self.df_metrics.describe())
        if self.latex:
            df_latex(self.df_metrics.describe())
        
        
    def explain_metrics(self):
        print(f'{self.pp}Metrics descriptions:')
        for m in self.metrics_names.items():
            print(f'{m[0]:30s} :  {m[1]}')

    def prediction_summary(self):
        
        self.display_metrics(T=True)
        
        if self.y_true is None or self.y_pred is None:
            return
                    
        print_md('**Result of the last updated prediction:**')
        y = self.y_true
        s = len(y[y == 1])
        n = len(y[y == 0])
        r = s/n if n > 0 else 0
        t = len(y)
        print(f'TRUE: Sim[1]={s:6d}  Não[0]={n:6d} ratio(S/N)={r:.2f} Total={t:6d}')
        
        y = self.y_pred    
        s = len(y[y == 1])
        n = len(y[y == 0])
        r = s/n if n > 0 else 0
        t = len(y)
        print(f'PRED: Sim[1]={s:6d}  Não[0]={n:6d} ratio(S/N)={r:.2f} Total={t:6d}')
                        
        cm = confusion_matrix(self.y_true, self.y_pred)

        print('prediction_summary: cm')
        accuracy = cm.diagonal().sum()/cm.sum()
        
        print_md(f'Accuracy: {accuracy:.2f}')    
            
        print_md('**Prediction Summary:**')            
            
        #By definition a confusion matrix C is such that C_ij is equal to the number of observations known to be in group i and predicted to be in group j .
        #Thus in binary classification, the count of true negatives is C_00, false negatives is C_10, true positives is C_11  and false positives is C_01.
        #print('Confusion matrix: (C_ij: observations in group i predicted as group j)')
        #print(cm)
        print('Confusion matrix: (literature format)')        
        tn, fp, fn, tp = cm.ravel()
        d = { 1 : [fn, tp], 0 : [tn, fp] }   # create conf mat. with inverted rows        
        dfcm = pd.DataFrame(d)
        dfcm = dfcm.iloc[::-1]               # revert rows
        print(dfcm)
        
        if self.latex:
            df_latex(dfcm)
            
        print(classification_report(self.y_true, self.y_pred))


# eof
