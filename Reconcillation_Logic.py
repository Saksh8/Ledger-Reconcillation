#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors


# In[2]:


def find_best_match(ledger_vector,ledger_amount, vendor_vectors, vendor_entries, vendor_amounts, threshold=0.5):
    best_match = None
    best_match_amount = None
    highest_score = 0
    min_amount_difference = float('inf')  # Initialize with infinity
    
    # Compute cosine similarities
    similarities = cosine_similarity(ledger_vector, vendor_vectors)
    
    for i, score in enumerate(similarities[0]):
        if score >= threshold:
            amount_difference = abs(ledger_amount-vendor_amounts[i])  # Calculate amount difference
            if score > highest_score or (score == highest_score and amount_difference < min_amount_difference):
                highest_score = score
                best_match = vendor_entries[i]
                best_match_amount = vendor_amounts[i]
                min_amount_difference = amount_difference
    
    return best_match, best_match_amount, highest_score


# In[3]:


def preprocess_payment(ledger_df,vendor_df):
    vendor_df['Net Amount']=vendor_df['Net Amount'].astype(float)
    ledger_df['Net Amount']=ledger_df['Net Amount'].astype(float)
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
    vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
    ledger_df1=ledger_df[ledger_df['TYPE']=='Payment']
    vendor_df1=vendor_df[vendor_df['TYPE']=='Receipt']
    vendor_df1['Net Amount']=abs(vendor_df1['Net Amount'])
    ledger_df1['Net Amount']=abs(ledger_df1['Net Amount'])
    ledger_df2=ledger_df[ledger_df['TYPE']=='Receipt']
    ledger_df2['Net Amount']=abs(ledger_df2['Net Amount'])
    vendor_df2=vendor_df[vendor_df['TYPE']=='Payment']
    vendor_df2['Net Amount']=abs(vendor_df2['Net Amount'])
    
    
    if(ledger_df1.shape[0]==0):
        ledger_df1.loc[0] = [0]*(ledger_df1.shape[1])
    if(vendor_df1.shape[0]==0):
        vendor_df1.loc[0] = [0]*(vendor_df1.shape[1])
    if(ledger_df2.shape[0]==0):
        ledger_df2.loc[0] = [0]*(ledger_df2.shape[1])
    if(vendor_df2.shape[0]==0):
        vendor_df2.loc[0] = [0]*(vendor_df2.shape[1])
    
    return ledger_df1,vendor_df1,vendor_df2,ledger_df2

def preprocess_tds(ledger_df,vendor_df):
    vendor_df['Net Amount']=vendor_df['Net Amount'].astype(float)
    ledger_df['Net Amount']=ledger_df['Net Amount'].astype(float)
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
    vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
    ledger_df1=ledger_df[ledger_df['TYPE']=='TDS']
    vendor_df1=vendor_df[vendor_df['TYPE']=='TDS']
    vendor_df1['Net Amount']=abs(vendor_df1['Net Amount'])
    ledger_df1['Net Amount']=abs(ledger_df1['Net Amount'])
   
    
    if(ledger_df1.shape[0]==0):
        ledger_df1.loc[0] = [0]*(ledger_df1.shape[1])
    if(vendor_df1.shape[0]==0):
        vendor_df1.loc[0] = [0]*(vendor_df1.shape[1])
   
    
    return ledger_df1,vendor_df1

# In[4]:


def preprocess_invoice(ledger_df,vendor_df):
   vendor_df['Net Amount']=vendor_df['Net Amount'].astype(float)
   ledger_df['Net Amount']=ledger_df['Net Amount'].astype(float)
   ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
   vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
   ledger_df1=ledger_df[(ledger_df['TYPE']=='Purchase') | (ledger_df['TYPE']=='Sales')]
   ledger_df1.reset_index(inplace=True,drop=True)
   vendor_df1=vendor_df[(vendor_df['TYPE']=='Sales') | (vendor_df['TYPE']=='Purchase')]
   vendor_df1.reset_index(inplace=True,drop=True)
   ledger_df2=ledger_df[(ledger_df['TYPE']=='Purchase') | (ledger_df['TYPE']=='Sales')]
   ledger_df2.reset_index(inplace=True,drop=True)
   vendor_df2=vendor_df[(vendor_df['TYPE']=='Sales') | (vendor_df['TYPE']=='Purchase')]
   vendor_df2.reset_index(inplace=True,drop=True)
   ledger_df1['INV'] = ledger_df1['INV'].astype(str).fillna('')
   vendor_df1['INV'] = vendor_df1['INV'].astype(str).fillna('')
   ledger_df2['INV'] = ledger_df2['INV'].astype(str).fillna('')
   vendor_df2['INV'] = vendor_df2['INV'].astype(str).fillna('')
   ledger_df1['INV'] = ledger_df1['INV'].str.replace('\r\n', '')
   vendor_df1['INV'] = vendor_df1['INV'].str.replace('\r\n', '')
   ledger_df2['INV'] = ledger_df2['INV'].str.replace('\r\n', '')
   vendor_df2['INV'] = vendor_df2['INV'].str.replace('\r\n', '')
   ledger_df1['Net Amount'] = abs(ledger_df1['Net Amount'])
   vendor_df1['Net Amount'] = abs(vendor_df1['Net Amount'])
   ledger_df2['Net Amount'] = abs(ledger_df2['Net Amount'])
   vendor_df2['Net Amount'] = abs(vendor_df2['Net Amount'])
   #ledger_df1 = ledger_df1.drop_duplicates(subset=['INV', 'Net Amount']).reset_index(drop=True)
   #ledger_df2 = ledger_df2.drop_duplicates(subset=['INV', 'Net Amount']).reset_index(drop=True)

   return ledger_df1,vendor_df1,ledger_df2,vendor_df2


# In[5]:


def preprocess_notes(ledger_df,vendor_df):
   vendor_df['Net Amount']=vendor_df['Net Amount'].astype(float)
   ledger_df['Net Amount']=ledger_df['Net Amount'].astype(float)
   ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
   vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
   ledger_df1=ledger_df[(ledger_df['TYPE']=='Credit Note') | (ledger_df['TYPE']=='Debit Note')]
   ledger_df1.reset_index(inplace=True,drop=True)
   vendor_df1=vendor_df[(vendor_df['TYPE']=='Debit Note') | (vendor_df['TYPE']=='Credit Note')]
   vendor_df1.reset_index(inplace=True,drop=True)
   ledger_df2=ledger_df[(ledger_df['TYPE']=='Debit Note') | (ledger_df['TYPE']=='Credit Note')]
   ledger_df2.reset_index(inplace=True,drop=True)
   vendor_df2=vendor_df[(vendor_df['TYPE']=='Credit Note') | (vendor_df['TYPE']=='Debit Note')]
   vendor_df2.reset_index(inplace=True,drop=True)
   ledger_df1['INV'] = ledger_df1['INV'].astype(str).fillna('')
   vendor_df1['INV'] = vendor_df1['INV'].astype(str).fillna('')
   ledger_df2['INV'] = ledger_df2['INV'].astype(str).fillna('')
   vendor_df2['INV'] = vendor_df2['INV'].astype(str).fillna('')
   ledger_df1['INV'] = ledger_df1['INV'].str.replace('\r\n', '')
   vendor_df1['INV'] = vendor_df1['INV'].str.replace('\r\n', '')
   ledger_df2['INV'] = ledger_df2['INV'].str.replace('\r\n', '')
   vendor_df2['INV'] = vendor_df2['INV'].str.replace('\r\n', '')
   ledger_df1['Net Amount'] = abs(ledger_df1['Net Amount'])
   vendor_df1['Net Amount'] = abs(vendor_df1['Net Amount'])
   ledger_df2['Net Amount'] = abs(ledger_df2['Net Amount'])
   vendor_df2['Net Amount'] = abs(vendor_df2['Net Amount'])
   #ledger_df1 = ledger_df1.drop_duplicates(subset=['INV', 'Net Amount']).reset_index(drop=True)
   #ledger_df2 = ledger_df2.drop_duplicates(subset=['INV', 'Net Amount']).reset_index(drop=True)
   
   return ledger_df1,vendor_df1,ledger_df2,vendor_df2


# In[6]:


def check_payment_match_date(ledger_df, vendor_df):
    # Grouping by Date and calculating the sum for each dataframe
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
    vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
    
    our_sum = ledger_df.groupby('Date')['Net Amount'].sum().reset_index()
    vendor_sum = vendor_df.groupby('Date')['Net Amount'].sum().reset_index()

    # Merging the sums on Date
    merged = pd.merge(our_sum, vendor_sum, on='Date', how='left', suffixes=('_our', '_vendor'))

    # Creating a dictionary for quick lookup
    date_match_dict = dict(zip(merged['Date'], merged['Net Amount_our'] == merged['Net Amount_vendor']))

    # Adding remarks to the original ledger
    ledger_df['remarks'] = ledger_df['Date'].map(lambda date: 'Payment Matched' if date_match_dict.get(date, False) else 'Payment Mismatch')

    return ledger_df


# In[7]:


def check_payment_match_month(ledger_df, vendor_df):
    # Convert Date columns to datetime in both dataframes
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
    vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
    
    # Extract month and year separately for grouping
    ledger_df['YearMonth'] = ledger_df['Date'].dt.to_period('M')
    vendor_df['YearMonth'] = vendor_df['Date'].dt.to_period('M')

    # Grouping by YearMonth and calculating the sum for each dataframe
    our_sum = ledger_df.groupby('YearMonth')['Net Amount'].sum().reset_index()
    vendor_sum = vendor_df.groupby('YearMonth')['Net Amount'].sum().reset_index()

    # Merging the sums on YearMonth
    merged = pd.merge(our_sum, vendor_sum, on='YearMonth', how='left', suffixes=('_our', '_vendor'))

    # Creating a dictionary for quick lookup
    date_match_dict = dict(zip(merged['YearMonth'],abs(merged['Net Amount_our']-merged['Net Amount_vendor'])<=3))

    # Adding remarks to the original ledger
    ledger_df['remarks'] = ledger_df['YearMonth'].map(lambda ym: 'Payment Matched' if date_match_dict.get(ym, False) else 'Payment Mismatch')

    # Drop the temporary YearMonth column
    ledger_df.drop(columns=['YearMonth'], inplace=True)

    return ledger_df

def check_tds_match_year(ledger_df, vendor_df):
    # Convert Date columns to datetime in both dataframes
    ledger_df['Date'] = pd.to_datetime(ledger_df['Date'])
    vendor_df['Date'] = pd.to_datetime(vendor_df['Date'])
    
    # Extract year for grouping
    ledger_df['Year'] = ledger_df['Date'].dt.year
    vendor_df['Year'] = vendor_df['Date'].dt.year

    # Grouping by Year and calculating the sum for each dataframe
    our_sum = ledger_df.groupby('Year')['Net Amount'].sum().reset_index()
    vendor_sum = vendor_df.groupby('Year')['Net Amount'].sum().reset_index()

    # Merging the sums on Year
    merged = pd.merge(our_sum, vendor_sum, on='Year', how='left', suffixes=('_our', '_vendor'))

    # Creating a dictionary for quick lookup
    date_match_dict = dict(zip(merged['Year'], abs(merged['Net Amount_our'] - merged['Net Amount_vendor']) <= 3))

    # Adding remarks to the original ledger
    ledger_df['remarks'] = ledger_df['Year'].map(lambda yr: 'TDS Matched' if date_match_dict.get(yr, False) else 'TDS Mismatch')

    # Drop the temporary Year column
    ledger_df.drop(columns=['Year'], inplace=True)

    return ledger_df



# In[8]:


def determine_remark(row):
    if (row['TYPE']=='Credit Note') or (row['TYPE']=='Debit Note'):
        if row['match_score']==1.000000:
            if math.floor(row['amount_difference']) <= 3.000000:
                if row['TYPE']=='Credit Note':
                    return 'CN Matched'
                else:
                    return 'DN Matched'
            else:
                return 'CN/DN Mismatch'
        else:
            if row['TYPE']=='Credit Note':
                    return 'CN Not Found'
            else:
                    return 'DN Not Found'
               
            
    else:
        if row['match_score']==0.000000:
            return 'Invoice Not Found'
        elif row['match_score'] == 1.000000:
            if row['Net Amount']==0.000000:
                return 'Invoice Match'
            elif math.floor(row['amount_difference']) <= 3.000000:
                return 'Invoice Match'
            else:
                return 'Purchase Invoice Mismatch'
        else:
                return 'Invoice Mismatch'


# In[9]:


def add_remark_column(ledger_df1):
    # Define a function to determine the remark for each row
   

    ledger_df1['remarks'] = ledger_df1.apply(determine_remark, axis=1)

    # Return the updated DataFrame
    return ledger_df1


# In[10]:


def duplicates(ledger_df1):
    ledger_df1_duplicates=ledger_df1.duplicated(subset=['INV','Net Amount'],keep='first')
    ledger_df1_duplicates=ledger_df1[ledger_df1_duplicates]
    ledger_df1_duplicates['remarks']='Duplicate Invoice'
    ledger_df1 = ledger_df1.drop_duplicates(subset=['INV', 'Net Amount']).reset_index(drop=True)
    return ledger_df1,ledger_df1_duplicates


# In[11]:


def update_cumulative_net_amount(ledger_df1_mismatch, vendor_df1):
    # Group by 'INV' and sum the 'Net Amount' for each dataframe
    ledger_grouped = ledger_df1_mismatch.groupby('INV')['Net Amount'].sum().reset_index()
    vendor_grouped = vendor_df1.groupby('INV')['Net Amount'].sum().reset_index()

    # Merge the grouped DataFrames on 'INV'
    merged_df = pd.merge(ledger_grouped, vendor_grouped, on='INV', suffixes=('_ledger', '_vendor'))

    tolerance =50
    merged_df['Matched'] = np.isclose(merged_df['Net Amount_ledger'], merged_df['Net Amount_vendor'], atol=tolerance)
    merged_df['remarks'] = merged_df['Matched'].apply(lambda x: 'Invoice Match' if x else 'Invoice Mismatch')

    # Update the 'remarks' column in the original ledger_df1_mismatch DataFrame
    ledger_df1_mismatch = pd.merge(ledger_df1_mismatch, merged_df[['INV', 'remarks']], on='INV', how='left')
    ledger_df1_mismatch['remarks'] = ledger_df1_mismatch['remarks_y'].fillna(ledger_df1_mismatch['remarks_x'])
    ledger_df1_mismatch.drop(['remarks_x', 'remarks_y'], axis=1, inplace=True)
    
   
    ledger_df1_mismatch['INV'] = ledger_df1_mismatch['INV'].astype(str)


    return ledger_df1_mismatch


# In[12]:


def reconcile_ledgers_invoice(ledger_df,vendor_df):
    # Reconciliation logic here
    ledger_df1,vendor_df1,ledger_df2,vendor_df2=preprocess_invoice(ledger_df,vendor_df)
    ledger_df1,ledger_df1_duplicates=duplicates(ledger_df1)
    vendor_df1,vendor_df1_duplicates=duplicates(vendor_df1)
    ledger_df2,ledger_df2_duplicates=duplicates(ledger_df2)
    vendor_df2,vendor_df2_duplicates=duplicates(vendor_df2)  
    

    ledger_df1=reconcillation(ledger_df1,vendor_df1)

    ledger_df1=add_remark_column(ledger_df1)
    ledger_df1_mismatch=ledger_df1[(ledger_df1['remarks']=='Purchase Invoice Mismatch') | (ledger_df1['remarks']=='Invoice  Mismatch')]
    ledger_df1_matched= ledger_df1[~ledger_df1.index.isin(ledger_df1_mismatch.index)]
    
    vendor_df1=reconcillation(vendor_df2,ledger_df2)
    vendor_df1=add_remark_column(vendor_df1)
    vendor_df1_mismatch=vendor_df1[(vendor_df1['remarks']=='Purchase Invoice Mismatch') | (vendor_df1['remarks']=='Invoice  Mismatch')]
    vendor_df1_matched= vendor_df1[~vendor_df1.index.isin(vendor_df1_mismatch.index)]
    
    ledger_df1_mismatch = update_cumulative_net_amount(ledger_df1_mismatch, vendor_df1_mismatch)
    vendor_df1_mismatch = update_cumulative_net_amount(vendor_df1_mismatch, ledger_df1_mismatch)
    
    reconciled_ledger_invoice=pd.concat([ledger_df1_matched,ledger_df1_mismatch,ledger_df1_duplicates])

    #reconciled_vendor_invoice=pd.concat([vendor_df1_matched,vendor_df1_mismatch,vendor_df1_duplicates],ignore_index=True)
    
    return reconciled_ledger_invoice


# In[13]:


def reconcillation(ledger_df1,vendor_df1):
    all_entries = ledger_df1['INV'].tolist() + vendor_df1['INV'].tolist()

# Vectorize the tokens using TF-IDF
    vectorizer = TfidfVectorizer().fit(all_entries)
    ledger_vectors1 = vectorizer.transform(ledger_df1['INV'])
    vendor_vectors1 = vectorizer.transform(vendor_df1['INV'])
    matches1 = [
    find_best_match(
        ledger_vectors1[i],ledger_df1['Net Amount'][i],
        vendor_vectors1, 
        vendor_df1['INV'], 
        vendor_df1['Net Amount']) 
    for i in range(len(ledger_df1))]
    ledger_df1['best_match'] = [match[0] for match in matches1]
    ledger_df1['match_amount'] = [match[1] for match in matches1]
    ledger_df1['match_score'] = [match[2] for match in matches1]
    
    
    ledger_df1['amount_difference']=abs(ledger_df1['Net Amount']-ledger_df1['match_amount'])
    ledger_df1['match_score'] = ledger_df1['match_score'].round(6)

    return ledger_df1


# In[14]:


def reconcile_ledgers_payment(ledger_df,vendor_df):
    ledger_df1,vendor_df1,vendor_df2,ledger_df2=preprocess_payment(ledger_df,vendor_df)
    
    ledger_df1_updated=check_payment_match_date(ledger_df1,vendor_df1)
    vendor_df1_updated=check_payment_match_date(vendor_df1,ledger_df1)
    ledger_df2_updated=check_payment_match_date(ledger_df2,vendor_df2)
    vendor_df2_updated=check_payment_match_date(vendor_df2,ledger_df2)
    
    
    
    ledger_df1_mismatch=ledger_df1_updated[ledger_df1_updated['remarks']=='Payment Mismatch']
    vendor_df1_mismatch=vendor_df1_updated[vendor_df1_updated['remarks']=='Payment Mismatch']
    
    ledger_df1_matched=ledger_df1_updated[~ledger_df1_updated.index.isin(ledger_df1_mismatch.index)]
    vendor_df1_matched=vendor_df1_updated[~vendor_df1_updated.index.isin(vendor_df1_mismatch.index)]
    
    ledger_df2_mismatch=ledger_df2_updated[ledger_df2_updated['remarks']=='Payment Mismatch']
    vendor_df2_mismatch=vendor_df2_updated[vendor_df2_updated['remarks']=='Payment Mismatch']
    
    ledger_df2_matched=ledger_df2_updated[~ledger_df2_updated.index.isin(ledger_df2_mismatch.index)]
    vendor_df2_matched=vendor_df2_updated[~vendor_df2_updated.index.isin(vendor_df2_mismatch.index)]
    
    ledger_df1_mismatch=check_payment_match_month(ledger_df1_mismatch,vendor_df1_mismatch)
    vendor_df1_mismatch=check_payment_match_month(vendor_df1_mismatch,ledger_df1_mismatch)
    
    
    ledger_df2_mismatch=check_payment_match_month(ledger_df2_mismatch,vendor_df2_mismatch)
    vendor_df2_mismatch=check_payment_match_month(vendor_df2_mismatch,ledger_df2_mismatch)
    
    reconciled_ledger_payment=pd.concat([ledger_df1_matched,ledger_df1_mismatch,ledger_df2_matched,ledger_df2_mismatch])
    ##reconciled_vendor_payment=pd.concat([vendor_df1_matched,vendor_df1_mismatch,vendor_df2_matched,vendor_df2_mismatch])
    
    
    return reconciled_ledger_payment
    
def reconcile_ledgers_tds(ledger_df,vendor_df):
    ledger_df1,vendor_df1=preprocess_tds(ledger_df,vendor_df)
    
    ledger_df1_updated=check_tds_match_year(ledger_df1,vendor_df1)
    
    return ledger_df1_updated


# In[15]:


def reconcile_ledgers_note(ledger_df,vendor_df):
    ledger_df1,vendor_df1,ledger_df2,vendor_df2=preprocess_notes(ledger_df,vendor_df)
    ledger_df1,ledger_df1_duplicates=duplicates(ledger_df1)
    vendor_df1,vendor_df1_duplicates=duplicates(vendor_df1)
    ledger_df2,ledger_df2_duplicates=duplicates(ledger_df2)
    vendor_df2,vendor_df2_duplicates=duplicates(vendor_df2)  
    

    ledger_df1=reconcillation(ledger_df1,vendor_df1)

    ledger_df1=add_remark_column(ledger_df1)
    ledger_df1_mismatch=ledger_df1[(ledger_df1['remarks']=='CN/DN Mismatch') | (ledger_df1['remarks']=='CN Not Found') | (ledger_df1['remarks']=='DN Not Found')]
    ledger_df1_matched= ledger_df1[~ledger_df1.index.isin(ledger_df1_mismatch.index)]
    
    vendor_df1=reconcillation(vendor_df2,ledger_df2)
    vendor_df1=add_remark_column(vendor_df1)
    vendor_df1_mismatch=vendor_df1[(vendor_df1['remarks']=='CN/DN Mismatch') | (vendor_df1['remarks']=='CN Not Found') | (vendor_df1['remarks']=='DN Not Found')]
    vendor_df1_matched= vendor_df1[~vendor_df1.index.isin(vendor_df1_mismatch.index)]
    
    ledger_df1_mismatch = update_cumulative_net_amount(ledger_df1_mismatch, vendor_df1_mismatch)
    vendor_df1_mismatch = update_cumulative_net_amount(vendor_df1_mismatch, ledger_df1_mismatch)
    
    reconciled_ledger_note=pd.concat([ledger_df1_matched,ledger_df1_mismatch,ledger_df1_duplicates])

    #reconciled_vendor_note=pd.concat([vendor_df1_matched,vendor_df1_mismatch,vendor_df1_duplicates])
    
    return reconciled_ledger_note


# In[16]:


def reconcile_ledgers(ledger_df,vendor_df):
    included_types = ['Purchase', 'Sales', 'Payment', 'Receipt', 'Credit Note', 'Debit Note','TDS']
    filtered_ledger_df = ledger_df[ledger_df['TYPE'].isin(included_types)]
    filtered_vendor_df = vendor_df[vendor_df['TYPE'].isin(included_types)]
    
    remaining_ledger_df=ledger_df[~ledger_df['TYPE'].isin(included_types)]
    #remaining_vendor_df=vendor_df[~vendor_df['TYPE'].isin(included_types)]
    
    filtered_ledger_df_invoice=reconcile_ledgers_invoice(filtered_ledger_df,filtered_vendor_df)
    #filtered_vendor_df_invoice=reconcile_ledgers_invoice(filtered_vendor_df,filtered_ledger_df)
    
    
    filtered_ledger_df_payment=reconcile_ledgers_payment(filtered_ledger_df,filtered_vendor_df)
    #filtered_vendor_df_payment=reconcile_ledgers_payment(filtered_vendor_df,filtered_ledger_df)
    
    
    filtered_ledger_df_notes=reconcile_ledgers_note(filtered_ledger_df,filtered_vendor_df)
    #filtered_vendor_df_notes=reconcile_ledgers_note(filtered_vendor_df,filtered_ledger_df)
    
    filtered_ledger_df_tds=reconcile_ledgers_tds(filtered_ledger_df,filtered_vendor_df)
    
    reconciled_ledger=pd.concat([filtered_ledger_df_invoice,filtered_ledger_df_payment,filtered_ledger_df_notes,filtered_ledger_df_tds,remaining_ledger_df])
    #reconciled_vendor=pd.concat([filtered_vendor_df_invoice,filtered_vendor_df_payment,filtered_vendor_df_notes,remaining_vendor_df])
    
    return reconciled_ledger
    

