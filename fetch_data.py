#!/usr/bin/python
# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd

def test():
    """Get 3 years history for a specified stock.

    History with detailed information (candlestick chart data) then saved to csv format.

    # Arguments
        stock_index: stock index code.
        ktype: candlestick data type.
    """
    ts.set_token("9a593228e963680b16083698320390df36b0c974b65053d2ab145501")
    pro = ts.pro_api()
    # df = ts.pro_bar(pro_api=pro, ts_code='600519.SH')
    df = ts.pro_bar(pro_api=pro, ts_code='000002.SZ')

    # df = pro_api.daily(stock_index)
    # df = ts.pro_bar(stock_index, pro_api)

    df = df.sort_index(ascending=True)
    # df = df.reset_index(drop=True)
    df = df.reset_index()
    col_list = df.columns.tolist()
    col_list.remove('close')
    col_list.append('close')
    df = df[col_list]
    print('\nSaving DataFrame: \n', df.head(5))
    df.to_csv('{}-3-year.csv'.format('000002'), index=False)



def wash(df, target='close'):
    """Process the entered DataFrame object.
    
    The last column of the output DataFrame is our prediction target.
    
    # Arguments
        df: input Pandas DataFrame object.
    # Returns
        Postprocessed DataFrame object.
    """
    # df = df.reset_index(drop=True)
    df = df.reset_index()
    col_list = df.columns.tolist()
    col_list.remove(target)
    col_list.append(target)
    return df[col_list]

def get_3_years_history(stock_index, ktype='D'):
    """Get 3 years history for a specified stock.
    
    History with detailed information (candlestick chart data) then saved to csv format.
    
    # Arguments
        stock_index: stock index code.
        ktype: candlestick data type.
    """
    df = ts.get_hist_data(stock_index, ktype=ktype)
    df = wash(df)
    print('\nSaving DataFrame: \n', df.head(5))
    df.to_csv('{}-3-year.csv'.format(stock_index), index=False)

def get_all_history(stock_index, start, autype=None):
    """Get history for a specified stock during a specified period.
    
    Saved to csv format.
    
    # Arguments
        stock_index: stock index code.
        start: start date of the interested period.
        autype: rehabilitation type.
    """
    df = ts.get_h_data(stock_index, start=start, autype=autype)
    df = wash(df)
    print('\nSaving DataFrame: \n', df.head(5))
    df.to_csv('{}-from-{}.csv'.format(stock_index, start), index=False)
    
# get_all_history('000002', start='1995-01-01')
get_3_years_history('600019')


