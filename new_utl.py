import pandas as pd
import pickle
import numpy as np
import datetime

'''
Quck tips:

Extract columns X,Y from a data frame D into a new
data frame D':

D' = D[[X,Y]]

where X and Y are column labels.

Extract the column of index i from a dataframe D:

S = D.iloc[:,i]

the resulting columns has type 'series'. The
entries of S can be indexed as well (i.e., S[<int>]).

Extract from a dataframe D a sub-dataframe composed
by the columns i to j and by the rows m to n,
where i<j and m<n. Put the result in a new dataframe
D'.

D' = D.iloc[m:n+1,i:j+1]


'''
# Add a brief description to each column label
# i.e. the meaning of each label
# everything is stored in a dictionary
class DataDescription:

    desc = 'description'
    dom = 'domain'
    nans = 'nan_percentage'
    col_idx = 'column_index'

    def __init__(self,file=""):
        if file == "":
            self.__data = {}
        else:
            infile = open(file,'rb')
            self.load_data(file)
            infile.close()
        return

    def get_everything(self):
        return self.__data

    # in case of stored data
    def store_data(self,file):
        outfile = open(file,"wb")
        pickle.dump(self.__data,outfile)
        outfile.close()
        return

    def load_data(self,file):
        infile = open(file,"rb")
        self.__data = pickle.load(infile)
        infile.close()
        return

    def get_description(self,label):
        ret = label + ": " + self.__data[label][self.desc]
        return ret

    def get_info(self,label):
        return self.__data[label]

    def set_description(self,label,description):
        self.__data[label] = {self.desc:description,self.dom:None}

    def set_domain(self,label,domain):
        try:
            self.__data[label][self.dom] = domain
        except:
            self.__data[label] = {self.desc:"",self.dom:domain}

    def set_nans(self,label,nan_p):
        try:
            self.__data[label][self.nans] = nan_p
        except:
            self.__data[label] = {self.desc:"",self.dom:None,self.nans:nan_p}

    def set_colidx(self,label,col_idx):
        try:
            self.__data[label][self.col_idx] = col_idx
        except:
            self.__data[label] = {self.desc:"",self.dom:None,self.nans:None, self.col_idx:col_idx}


def extract_dataset(df,input_columns=[],output_column=-1):
    if len(input_columns) == 0 or output_column==-1:
        return pd.DataFrame() # empty df

    input_columns.insert(len(input_columns),output_column)
    #print(len(input_columns))
    new_df = df.iloc[:,input_columns]
    return new_df

def count_non_zeros(data_frame, nan_as=0): # nan are countes as non-zero
    return data_frame.fillna(nan_as).astype(bool).sum(axis=0).sum()

def count_zeros(data_frame):
    total_entries = data_frame.size
    zeros = total_entries - data_frame.fillna(-1).astype(bool).sum(axis=0).sum()
    return zeros

def count_nans(data_frame):
    total_entries = data_frame.size
    non_nans = data_frame.count().sum()
    nans = total_entries - non_nans
    return nans

def series_info(s):
    print(f"repr: {s}")
    print(f"size: {s.size}")
    print(f"domain: {s.astype('category')}")
    print(f"#nans: {count_nans(s)}")
    return

def my_search(df,val):

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iat[i,j] == 'Unknown':
                print(i," ",j, "->", df.iat[i,j])

    return

# some rules to process the data frame before extracting the rows

# a naive rule: map all nans to 0
# no sideeffect
def rule_map_nan_to_zero(data_frame):
    new_df = data_frame.fillna(0)
    return new_df

# with side effects!
def rule_map_string_to_1(data_frame,col):
    col_entries = data_frame.iloc[:,col].iteritems()
    for idx,s in col_entries:
        if s:
            data_frame.iat[idx,col] = 1
    return data_frame

def rule_map_nan_col_to_value(data_frame,col,value):
    col_series = data_frame[col]
    #col_series = col_series.fillna(value)
    data_frame[col] = col_series.fillna(value)
    return data_frame

def rule_map_mmol_to_g(data_frame,col1,col2,unit):

    for i in range(data_frame[col1].size):
        if data_frame.loc[i,unit] == 'mmol/L':
            # then convert it to g/dl
            data_frame.loc[i,col1] /= 0.6206
            data_frame.loc[i,col2] /= 0.6206
            data_frame.loc[i,unit] = 'g/dl'
    return

def rule_categorize_columns(df):
    for col in df.columns:
        if df[col].dtype != 'O':
            continue
        else:
            df[col] = df[col].astype('category').cat.codes

    return

# TODO

# end rules #################################

def compute_numeric_series(data_frame):
    cols_to_extract = []
    for i,(label,series) in zip(range(data_frame.shape[1]),data_frame.iteritems()):
        if series.dtype == np.int64 or series.dtype == np.float64:
            cols_to_extract.insert(len(cols_to_extract),i)
    return cols_to_extract

def preprocess_dataframe(data_frame):
    description = DataDescription()
    for i,col in zip(range(data_frame.shape[1]),data_frame.columns):
        # print(f"processing column {i}: {col}...")
        series = data_frame[col]
        if series.dtype == np.object:
            string_domain(series,description,col)
        elif series.dtype == np.int64:
            int_domain(series,description,col)
        elif series.dtype == np.float64:
            float_domain(series,description,col)
        else:
            print("ERROR: preprocess_dataframe: not recognized type.")
            return
        # set the percentage of null values for each column
        description.set_nans(col,1-(series.count()/len(series)))
        description.set_colidx(col,i)

    return description

def string_domain(series,description,label):
    if label.lower().find("date")>=0:
        # then it has to be a date field.
        min_date = datetime.date(year=3000,month=1,day=1)
        max_date = datetime.date(year=1000,month=1,day=1)
        for date in series:
            date_repr = repr(date)
            if date_repr == 'nan': continue
            year = int(date[6:])
            month = int(date[3:5])
            day = int(date[0:2])
            current_date = datetime.date(year,month,day)
            if current_date > max_date:
                 max_date = current_date
            if current_date < min_date:
                min_date = current_date
        description.set_domain(label,[min_date,max_date])
        return

    # treat other string fields simple strings:
    description.set_domain(label,"String Field")
    return

def float_domain(series,description,label):
    min = series.min()
    max = series.max()
    description.set_domain(label,[min,max])
    return

def int_domain(series,description,label):
    min = series.min()
    max = series.max()
    description.set_domain(label,[min,max])
    return

# prints the columns label of a csv file
def csv_pretty_print_cols(file_csv):
    df = pd.read_csv(file_csv,sep=";")
    df_pretty_print_cols(df)
    return

# prints the columns labels of a dataframe
def df_pretty_print_col_labels(data_frame):
    df = data_frame
    n_cols = df.shape[1]
    cols = df.columns
    for i,col in zip(range(n_cols),cols):
        print(i," ",col)

def pretty_print_cols(df,key):
    l = all_occurrences(df,key)
    for i,k in l:
        print(f"{i})\t{k}")
    return

# finds the first occurrence of a string containing "key" in 'labels' and
# returns (position,<string containing key>)
# NOTICE: case insensitive
def first_occurrence(labels, key):
    #print("NOTICE: case insensitive")
    for i,l in zip(range(len(labels)),labels):
        l1 = ''.join([y.lower() for y in str(l)]) # lower the case of l
        index = l1.find(key.lower()) # search for key in l1
        if index != -1:
            return (i,l)
    return (-1,"")

# finds all occurrences of a string containing "key" in 'labels' and
# returns (position,<string containing key>)
# NOTICE: case insensitive
def all_occurrences(labels,key):
    #print("NOTICE: case insensitive")
    occs = []
    for i,l in zip(range(len(labels)),labels):
        l1 = ''.join([y.lower() for y in str(l)]) # lower the case of l
        index = l1.find(key.lower()) # search for key in l1
        if index != -1:
            occs.insert(len(occs),(i,l))
    return occs

def csv2df(file):
    df = pd.read_csv(file,sep=';')
    return df

def get_rs_dataframe():
    df = pd.read_pickle('datasets/rs_extended_cat.gzip')
    return df

def get_ps_dataframe():
    df = pd.read_pickle('datasets/ps_cat.gzip')
    return df

def get_ps_dataframe_full():
    ps = pd.read_pickle('datasets/ps.gzip')
    return ps

def get_rs_output_column():
    file = 'datasets/openclinica_rs_ALL_extended.csv'
    df = csv2df(file)
    output_column = 'follow_Status_of_Patient'
    Ys = df[output_column].astype('category').cat.codes.to_numpy()
    return Ys

def get_rs_full_dataframe():
    file = 'datasets/openclinica_rs_ALL_extended.csv'
    df = df = csv2df(file)
    return df

def get_ps_data_to_predict(alive_after=4*365):
    ps_full = get_ps_dataframe_full()
    Ys = compute_alive_after(ps_full,alive_after=alive_after,format_date='ita')
    Xs = get_ps_dataframe()
    return (Xs,Ys)

def get_rs_training_data(alive_after=365*10):
    rs_preprocessed = get_rs_dataframe()
    Xs = rs_preprocessed.to_numpy()
    features = rs_preprocessed.columns
    # get rs full dataframe
    rs_full = get_rs_full_dataframe()
    Ys = compute_alive_after(rs_full,'clinical_Date_of_first_Diagnosis','follow_Date_of_Death',alive_after,format_date='ita')
    return Xs,Ys,features


# some datasets contain italian date format while others contain english data
# formattig. The following data structure can be used to access the y,m,d, fields
# in both formats with a unique syntax.
date_format = {
'ita' : {'y':(lambda s: s[6:]),
            'm':(lambda s: s[3:5]),
            'd':(lambda s: s[0:2]),
            'dummy': '01/01/1500'
           },

'eng' : {'y':(lambda s: s[0:4]),
            'm':(lambda s: s[5:7]),
            'd':(lambda s: s[8:]),
            'dummy':'1500-01-01'
           },
}

def compute_alive_after(df,start='clinical_Date_of_first_Diagnosis',end='follow_Date_of_Death',alive_after=10*365,format_date='eng'):

    print(f'Using format \'{format_date}\' for dates')
    d_format = date_format[format_date]
    Y = []
    delta = alive_after
    # get the input columns
    col_start = df[start]
    col_end = df[end].fillna(d_format['dummy'])
    max_alive = count_nans(df[end])
    assert count_nans(col_start) == 0
    # the end columns may have nans which need to
    # be filled

    # parse each string into a date object
    for s,e in zip(col_start,col_end):
        #print(f'start:{s}, end:{e}')
        # convert s
        s_year = int( d_format['y'](s) )
        s_month = int( d_format['m'](s) )
        s_day = int( d_format['d'](s) )
        s_date = datetime.date(s_year,s_month,s_day)

        # convert e
        e_year = int( d_format['y'](e) )
        e_month = int( d_format['m'](e) )
        e_day = int( d_format['d'](e) )
        e_date = datetime.date(e_year,e_month,e_day)

        # compute the time-delta
        delta_es = (e_date-s_date).days
        if delta_es < 0:
            assert e_date.year == 1500
            #Y.insert(len(Y),'Yes') # still alive
            Y.insert(len(Y),1)
        elif delta_es <= delta:
            #Y.insert(len(Y),'No') # dead after delta
            Y.insert(len(Y),0)
        else:
            #Y.insert(len(Y),'Yes') # alive after delta
            Y.insert(len(Y),1)

    Y = np.array(Y)
    print(f"Number of people alive after {delta} days: {Y.sum()} out of {len(Y)}")
    print(f"Maximum number of alive people in dataset: {max_alive} out of {len(Y)}")
    return Y

# value != nan
def count_occurrences(df,value):
    c = 0
    if len(df.shape) > 1:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if df.iat[i,j]==value:
                    c+=1
    else:
        for i in range(df.size):
            if df.iloc[i] == value:
                c+=1

    return c

### Training procedures ###
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Random Forest
def rf_approach(Xs,Ys,features,classif=True):
    # Training
    print("Random forest training...")
    print(f"input size: {len(Xs)}")
    print(f"number of features considered: {len(features)}")
    training_size = 0.8
    print(f"training on {int(training_size*len(Xs))} elements")
    print(f"testing on {int((1-training_size)*len(Xs))} elements")

    X_train, X_test, Y_train, Y_test=  train_test_split(Xs, Ys, test_size=1-training_size)

    if classif:
        print('Performing classification ...')
        return rf_classifier(X_train, X_test, Y_train, Y_test,features)
    
    print('Performing regression ...')
    return rf_regressor(X_train, X_test, Y_train, Y_test,features)



# performs classification 
def rf_classifier(X_train, X_test, Y_train, Y_test,features):
    classifier = RandomForestClassifier(n_estimators=1000,
    random_state=0, criterion='gini')
    classifier.fit(X_train,Y_train)
    Y_pred =  classifier.predict(X_test)
    
    
    # Evaluation
    print("output:")
    #print("confusion matrix:\n",confusion_matrix(Y_test,Y_pred))
    #print("classification report\n",classification_report(Y_test,Y_pred))
    print("Accuracy score: ", accuracy_score(Y_test, Y_pred))
    
    importance = pd.DataFrame(classifier.feature_importances_,index=features)
    importance = importance.sort_values(by=0,ascending=False)
    
    return classifier, importance

# performs regression 
def rf_regressor(X_train, X_test, Y_train, Y_test,features):

    regressor = RandomForestRegressor(n_estimators=1000,
    random_state=0, criterion='mse',)
    regressor.fit(X_train,Y_train)
    Y_pred = regressor.predict(X_test)

    

    importance = pd.DataFrame(regressor.feature_importances_,index=features)
    importance = importance.sort_values(by=0,ascending=False)

    print("Accuracy score: ", accuracy_score(Y_test, np.array(Y_pred).round() ))
    
    return regressor, importance


def rf_evaluate(Y_test,Y_pred):
    print("output:")
    print("confusion matrix:\n",confusion_matrix(Y_test,Y_pred))
    print("classification report\n",classification_report(Y_test,Y_pred))
    print("Accuracy score: ", accuracy_score(Y_test, Y_pred))
    print("Done")
    return
