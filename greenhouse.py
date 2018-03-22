# Attempted simulation of greenhouse environment.
# Import libraries
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# Import standard libraries.
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
        
    def set_up_data(self, dataset='weather_data.csv'):
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
        df = df.set_index('YYYYMMDD')
        # Plot temperature vs date.
#        plt.plot(df.YYYYMMDD, df['   TX'])
        
        # Create time dataframe for greenhouse temperature.
        greenhouse_df = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T00:00:00.000Z',
                                                                 '2017-3-01T00:00:00.000Z',
                                                                 freq='H'),
                                                                 'temperature': np.nan})
        greenhouse_df_max_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T15:00:00.000Z',
                                                                 '2017-10-29T00:00:00.000Z',
                                                                 freq='D'),
                                               'temperature': df['   TX']})
        greenhouse_df_min_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T06:00:00.000Z',
                                                                 '2017-10-29T00:00:00.000Z',
                                                                 freq='D'),
                                               'temperature': df['   TN']})
        # Then we can join on date.
        # Join temperature columns.
        greenhouse_df = greenhouse_df.merge(greenhouse_df_max_temp, left_on='datetimes', right_on='datetimes',
                                            how='left').drop('temperature_x', axis=1)
        greenhouse_df = greenhouse_df.merge(greenhouse_df_min_temp, left_on='datetimes', right_on='datetimes',
                                            how='left')
        # Fill the NaN values with values from minimum temps.
        greenhouse_df['temperature'] = greenhouse_df['temperature_y'].fillna(greenhouse_df['temperature'])
        # Interpolate with linear regression between vals.
        greenhouse_df['temperature'] = greenhouse_df['temperature_y'].interpolate(method='linear')
        # Fill the final NaN values.
        greenhouse_df['temperature'] = greenhouse_df['temperature'].fillna(7.25)
        # Drop the column we won't be using.
        greenhouse_df = greenhouse_df.drop('temperature_y', 1)
        # Make df index the date also.
        greenhouse_df = greenhouse_df.set_index('datetimes')
        # Create a column for sunshine
        # Join the sunshine column to temp df
        # Make timezones the same among df's
        df.index.tz = 'UTC'
        greenhouse_df = greenhouse_df.copy()
        greenhouse_df = greenhouse_df.merge(df, left_index=True,
                                            right_index=True, how='left')
        # Fill in NaN values with interpolate
        greenhouse_df = greenhouse_df.interpolate(method='linear')
        
        # Plot new temp values with time
        plt.plot(greenhouse_df.index, greenhouse_df.temperature)
        plt.show()
        # Plot first 5000 samples of temperature.
        plt.plot(greenhouse_df.index[:5000], greenhouse_df.temperature[:5000])
        plt.show()
        return df, greenhouse_df
    
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
    
    def water_storage_usage(self, daily_rainfall=10, tank_size=100, irrigation=True, irrigation_rate=2):
        '''
        Calculates water level in the rainwater tank along with usage of water
        from irrigation of crops.
        
         - Daily rainfall in mm.
         - Tank size in m^3.
        '''
        self.daily_rainfall = daily_rainfall
        self.tank_size = tank_size
        self.water_supply = 0
        self.water_input = self.daily_rainfall * self.area
        self.irrigation_rate = irrigation_rate
        
        # Also should use transpiration rate to calculate the water required.
        
        # For input to watertank via rainfall.
        if self.water_supply < (100 - self.water_input):
            self.water_supply += self.water_input
        
        # For outflow from water tank via irrigation of crops.
        if irrigation:
            self.water_supply -= self.irrigation_rate
    
    def greenhouse_temperature(self):
        '''
        Calculates the temperature in a greenhouse by hour based on daily temperature
        values.
        '''
        pass
    
    
    
    
    
    
    
x = green_house()
df, greenhouse  = x.set_up_data()