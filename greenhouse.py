# Attempted simulation of greenhouse environment.
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import os
import datetime
# Set cwd
os.chdir(r'C:/Users/Tim/pythonscripts/Greenhouse')

class green_house():
    '''
    A class for creating data, and controlling a greenhouse to optimise for 
    gdd, production and minimise resource usage.
    '''       
    def __init__(self, crop='cucumber', fert=True, pest=True, area=200):
        self.crop = crop
        self.fert = fert
        self.pest = pest
        self.area = 200
        
    def set_up_temp_data(self, dataset='weather_data.csv'):
        '''
        Weather data obtained from:
        http://www.sciamachy-validation.org/climatology/daily_data/selection.cgi
        
        This function prepares daily temperature data by transforming it into
        
        '''
        # Import the weather dataset.
        self.dataset = dataset
        df = pd.read_csv(self.dataset)
        # Change datetime column to datetime format.
        df.YYYYMMDD = pd.to_datetime(df.YYYYMMDD, format='%Y%m%d', errors='ignore')
        print(df.columns.values) 
        # Change temperates from 0.1C to standard units
        df['   TX'] = df['   TX'] / 10
        df['   TN'] = df['   TN'] / 10
        # Plot temperature vs date.
        plt.plot(df.YYYYMMDD, df['   TX'])
        
        # Create time dataframe for greenhouse temperature.
        greenhouse_df = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T00:00:00.000Z',
                                                                 '2017-3-01T00:00:00.000Z',
                                                                 freq='H'),
                                                                 'temperature': 0})
        greenhouse_df_max_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T15:00:00.000Z',
                                                                 '2017-10-29T00:00:00.000Z',
                                                                 freq='D'),
                                               'temperature': df['   TX']})
        greenhouse_df_min_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T06:00:00.000Z',
                                                                 '2017-10-29T00:00:00.000Z',
                                                                 freq='D'),
                                               'temperature': df['   TN']})
        # Then we can join on date :)
        greenhouse = greenhouse_df.join(greenhouse_df_max_temp)
        
        return df, greenhouse_df, greenhouse_df_max_temp, greenhouse_df_min_temp
        
    def set_up_rainfall_data(self, dataset='weather_data.csv'):
        pass
    
    def growing_degree_days(self, daily_max_temp=20):
        if self.crop == 'cucumber':
            '''
            Formula and temperatures obtained from: 
            http://horttech.ashspublications.org/content/6/1/27.short
            
            gdd requirement for cucumbers from:
            http://nwhortsoc.com/wp-content/uploads/2016/01/Andrews-Croptime-1.pdf,
            adjusted to 900 for current formula with max temperature
            '''
            self.base_temp = 15.5
            self.max_temp = 32
            self.gdd_req = 482
            self.daily_max_temp = daily_max_temp
            self.gdd = 0
            
            if self.daily_max_temp >= self.max_temp:
                self.gdd += (self.max_temp - (self.daily_max_temp - self.max_temp)) - self.base_temp
                
            elif self.daily_max_temp <= self.base_temp:
                self.gdd += 0
                
            else:
                self.gdd += self.daily_max_temp - self.base_temp
    
    def water_storage_usage(self, daily_rainfall=10, tank_size=100):
        '''
        Calculates water level in the rainwater tank along with usage of water
        from irrigation of crops.
        
         - Daily rainfall in mm.
         - Tank size in m^3.
        '''
        pass

x = green_house()
df, greenhouse, green_house_max, green_house_min = x.set_up_temp_data()