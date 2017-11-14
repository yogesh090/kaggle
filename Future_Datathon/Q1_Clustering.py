#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:45:26 2017

@author: yogi
"""

import numpy as np
import pandas as pd
import seaborn as sns

products = pd.read_csv('cproducts.csv')
payment = pd.read_csv('ctender.csv')

#get the feel of data
products.shape
products.columns
products.head()
products.info()
products.describe()

payment.shape
payment.columns
payment.head()
payment.info()
payment.describe()

sns.distplot(products['store_code'], bins=20, kde=False)
sns.distplot(products['product_code'], bins=20, kde=False)

print(products['product_code'].value_counts())
products['customerID'].nunique()
products['product_code'].nunique()
products['store_code'].nunique()
products['transaction_number_by_till'].nunique()

payment['tender_type'].nunique()
payment['transaction_number_by_till'].nunique()

#Missing Data
total = products.isnull().sum().sort_values(ascending=False)
percent = (products.isnull().sum()/products.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

total = payment.isnull().sum().sort_values(ascending=False)
percent = (payment.isnull().sum()/payment.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)