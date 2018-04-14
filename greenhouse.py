# Attempted simulation of greenhouse environment.
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# Plot style.
plt.style.use(['ggplot', 'dark_background'])
# Import standard libraries.
import os
import datetime
# Libraries for DeepQ.
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Set pandas options.
pd.options.mode.chained_assignment = None
# Set cwd
os.chdir(r'C:/Users/Tim/pythonscripts/Greenhouse')

class green_house():
    '''
    A class for creating data, and controlling a greenhouse to optimise for 
    gdd, production and minimise resource usage.
    '''       
    def __init__(self, crop='cucumber', fert=True, pest=True, area=200):
        
        # Type of crop.
        self.crop = crop
        # Use fertiliser.
        self.fert = fert
        # Use pesticide.
        self.pest = pest
        # Square area of the greenhouse.
        self.area = 200
        # Temperature outside the greenhouse
        self.outside_temp = 0
        # Current growing degree day count.
        self.gdd = 0
        # Max temperature of the day.
        self.daily_max_temp = 0
        # Current temperature inside the greenhouse.
        self.current_temperature = 0
        # Is the sun out?
        self.is_sunny = True
        # Temperature due to sun.
        self.sun_temperature = 0
        # Vent temperature effect.
        self.vent_temperature_effect = 0
        # Heating effect.
        self.heating_temperature = 0
        # Fogging temperature effect.
        self.fogging_temperature_effect = 0
        # Base NN value.
        self.nn_val = 1
        # Base NN val for vents.
        self.vent_nn_val = 1
        # Base NN val for heating.
        self.heating_nn_val = 1
        # Base NN val for screen,
        self.screen_nn_val = 1
        # Resource usage.
        self.resources_used = 0
        self.resources_used_per_harvest = []
        # Hours for harvest to occur.
        self.hours_for_harvest = 0
        self.harvest_times = []
        # gdd requirement for cucumbers
        if self.crop == 'cucumber':
            self.base_temp = 15.5
            self.max_temp = 28
            self.too_hot = 36
            # gdds turned into hours required. (Growing degree hours).
            self.gdd_req = 11568
        
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

        # Change time format.
        df = df.set_index('YYYYMMDD')
        # Plot temperature vs date.
#        plt.plot(df.YYYYMMDD, df['   TX'])
        
        # Create time dataframe for greenhouse temperature.
        greenhouse_df = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T00:00:00.000Z',
                                                                 '2016-3-01T00:00:00.000Z',
                                                                 freq='H'),
                                                                 'temperature': np.nan})
        greenhouse_df_max_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T15:00:00.000Z',
                                                                 '2016-03-02T00:00:00.000Z',
                                                                 freq='D'),
                                               'temperature': df['   TX']})
        greenhouse_df_min_temp = pd.DataFrame({'datetimes': pd.date_range('2010-3-01T06:00:00.000Z',
                                                                 '2016-03-02T00:00:00.000Z',
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
        
        # Remove spaces in cloud cover rows.
        greenhouse_df = greenhouse_df.replace(to_replace='     ', value=np.nan)
        # Turn cloud cover values to float for transformation.
        greenhouse_df['   NG'] = greenhouse_df['   NG'].astype(float)
        # Fill in NaN values with interpolate
        greenhouse_df = greenhouse_df.interpolate(method='linear')
        # Compute average temperature over last 5 hours.
        greenhouse_df['5_hour_temp_avg'] = pd.rolling_mean(greenhouse_df['temperature'], window=5, min_periods=1)
        # Add growing degree days column.
        greenhouse_df['growing_degree_days'] = 0
        # Add is sunny column. Only currently says 12 hours a day is sunny/non sunny.
        greenhouse_df['is_sunny'] = True
        for i, indx in enumerate(greenhouse_df.index):
            indx = pd.Timestamp(indx)
            hour = int(indx.hour)
            if hour > 6 and hour <= 18:
                greenhouse_df['is_sunny'][i] = True
            else:
                greenhouse_df['is_sunny'][i] = False
                
        # Plot new temp values with time
        plt.plot(greenhouse_df.index, greenhouse_df.temperature)
        plt.show()
        # Plot first 5000 samples of temperature.
        plt.plot(greenhouse_df.index[:5000], greenhouse_df.temperature[:5000])
        plt.show()
        # Plot rainfall.
        plt.plot(greenhouse_df.index, greenhouse_df['   RH'])
        plt.show()
        # Plot sunshine duration.
        plt.plot(greenhouse_df.index, greenhouse_df['   SQ'])
        plt.show()
        # Remove final NaN rows.
        self.greenhouse_df = greenhouse_df.dropna()
        
        return df, greenhouse_df
        
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
    
        return self.water_supply
    
    def soil_water_content(self):
        '''
        Calculates the soil water content in the greenhouse 
        '''
        self.wilting_capacity = 0
        self.soil_water = 0
        self.soil_saturation = 0
        return self.soil_water
    
    def greenhouse_control_evaluation(self):
        '''
        Controls opening and closing of vents, turning on of irrigation etc.
        '''
        self.vent_threshold = 0.5
        self.heating_threshold = 0.5
        self.screen_threshold = 0.5
        self.lighting_threshold = 0.5
        self.fogging_threshold = 0.5
        self.carbon_dioxide_threshold = 0.5
        self.irrigation_threshold = 0.5

        if self.vent_nn_val >= self.vent_threshold:            
            self.vents = True
        else:
            self.vents = False
        
        if self.heating_nn_val >= self.heating_threshold:
            self.heating = True
        else:
            self.heating = False
            
        if self.screen_nn_val >= self.screen_threshold:
            self.screens = True
        else:
            self.screens = False
            
        if self.nn_val >= self.lighting_threshold:
            self.lighting = True
        else:
            self.lighting = False
        
        if self.nn_val >= self.fogging_threshold:
            self.fogging = True
        else:
            self.fogging = False
            
        if self.nn_val >= self.irrigation_threshold:
            self.irigation = True
        else:
            self.irrigation = False
            
        if self.nn_val >= self.carbon_dioxide_threshold:
            self.carbon_dioxide_input = True
        else:
            self.carbon_dioxide_input = False
            
    def vent_effect(self):
        '''
        Effect of vents being opened or closed on the temperature within the
        greenhouse.
        '''
        # Reset vent change
        self.vent_temperature_effect = 0.0
        
        if self.vents:
            if self.current_temperature >= self.outside_temp and self.is_sunny:
                self.vent_temperature_effect -= 1.0
            elif self.current_temperature >= self.outside_temp and not self.is_sunny:
                self.vent_temperature_effect -= 3.0
        

        return self.vent_temperature_effect
                
    def heating_effect(self):
        '''
        Effect of heating being on or off on the temperature within the
        greenhouse.
        '''
        if self.heating:
            self.heating_temperature += 2.0
            self.resources_used += 1
        else:
            self.heating_temperature = 0.0
        
        return self.heating_temperature
                
    def screen_effect(self):
        '''
        Effect of the screens being up or down on the greenhouse.
        '''
        if self.screens:
            self.is_photosynthesizing = False
            self.insulation = 0.2 # 0.2 * heat loss from inside / heat gain from outside.
        else:
            self.insulation = 1.0

        return self.insulation
        
    def sun_effect(self):
        '''
        The effect of sunlight upon greenhouse temperature.
        '''
        if self.is_sunny:
            self.sun_temperature += 1
#        elif self.is_sunny == False and self.current_temperature <= (self.outside_temp + 4):
#            self.sun_temperature -= 4
        elif self.current_temperature > self.outside_temp + 4 and not self.is_sunny:
            self.sun_temperature -= 4
            # Sun temperature can't go below 0.
            if self.sun_temperature < 0:
                self.sun_temperature = 0
        
        return self.sun_temperature
        
    def lighting_effect(self):
        '''
        The effect that lighting has on plant growth in the greenhouse.
        '''
        if self.lighting:
            self.is_photosynthesizing = True

    def fogging_effect(self):
        '''
        The effect fogging has on plant growth.
        '''
        if self.fogging:
            self.resources_used += 1
            self.fogging_temperature_effect = -6
        else:
            self.fogging_temperature_effect = 0
            
        return self.fogging_temperature_effect
        
    def irrigation_effect(self):
        '''

        '''
        pass

    def carbon_input_effect(self):
        '''

        '''
        if self.carbon_dioxide_input:
            self.photosynthesis_rate *= 1.1
        
    def calculate_greenhouse_temperature(self, policy='basic'):
        '''
        Calculates the temperature in the greenhouse based on outside and controlled factors.
        
        Policies include: random, qlearning, markov and basic.
        '''
        self.greenhouse_df['greenhouse_temperature'] = 0
        
        for i, value in enumerate(self.greenhouse_df.index):
            # Completely random policy
            if policy == 'random':
                # Set the threshold value, this is with random control.
                self.nn_val = np.random.uniform(0, 1)
                self.screen_nn_val = np.random.uniform(0, 1)
                self.heating_nn_val = np.random.uniform(0, 1)
                self.vent_nn_val = np.random.uniform(0, 1)
            # Naive policy.
            elif policy == 'basic':
                if self.current_temperature > self.max_temp:
                    self.nn_val = 1
                    self.screen_nn_val = 1
                    self.vent_nn_val = 1
                    self.heating_nn_val = 0
                elif self.current_temperature <= self.base_temp:
                    self.nn_val = 0
                    self.screen_nn_val = 1
                    self.vent_nn_val = 0
                    self.heating_nn_val = 1
            
            self.greenhouse_control_evaluation()
            # Is it sunny this hour?
            self.is_sunny = self.greenhouse_df['is_sunny'][i]
    
            if i % 10000 == 0:
                print(self.vents, self.heating, self.screens, self.carbon_dioxide_input)
                print(self.heating_effect(), self.screen_effect(), self.vent_effect(), self.sun_effect())
                print(self.current_temperature, self.outside_temp)
            if i == 0:
                # Initial value for greenhouse temperature.
                self.greenhouse_df['greenhouse_temperature'][i] = self.greenhouse_df['5_hour_temp_avg'][i] + \
                self.fogging_effect() + self.vent_effect() + ((self.heating_effect() + self.sun_effect()) * self.screen_effect())
                
            else:
                # Any value for greenhouse temperature past the first row.
                self.greenhouse_df['greenhouse_temperature'][i] =((self.greenhouse_df['greenhouse_temperature'][i - 1] + \
                self.greenhouse_df['5_hour_temp_avg'][i]) / 2) + self.vent_effect() + \
                self.fogging_effect() + (self.heating_effect() + self.sun_effect()) * self.screen_effect()
            # Pass back the temperature to the functions.
            self.current_temperature = self.greenhouse_df['greenhouse_temperature'][i]
            # Pass back outside temperature.
            self.outside_temp = self.greenhouse_df['5_hour_temp_avg'][i]
        # Plot the greenhouse temperature.
        plt.plot(self.greenhouse_df['greenhouse_temperature'], alpha=0.2)
        plt.plot(self.greenhouse_df['5_hour_temp_avg'], alpha=1, color='red')
        plt.grid('off')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()
        # Plot the same over a shorter range to see daily variation.
        plt.plot(self.greenhouse_df['greenhouse_temperature'][:200], alpha=0.2)
        plt.plot(self.greenhouse_df['5_hour_temp_avg'][:200], alpha=1, color='red')
        plt.grid('off')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

        return self.greenhouse_df
        
    def calculate_gdd(self):
        '''
        Formula and temperatures obtained from: 
        http://horttech.ashspublications.org/content/6/1/27.short
            
        gdd requirement for cucumbers from:
        http://nwhortsoc.com/wp-content/uploads/2016/01/Andrews-Croptime-1.pdf,
        adjusted to 900 for current formula with max temperature
        '''   
        # Calculate hourly gdd counts.
        for i, temp in enumerate(self.greenhouse_df['greenhouse_temperature']):
            # For temperatures over the comfortable growing temperatures.
            if temp >= self.max_temp and temp < self.too_hot:
                # Still can grow because it isn't too hot.
                self.gdd += (self.max_temp - (temp - self.max_temp)) - self.base_temp
            # Can't grow as it is too hot.
            elif temp >= self.too_hot:
                self.gdd += 0
            
            # For temperatures less than comfortable growing conditions. 
            elif temp <= self.base_temp:
                self.gdd += 0
            
            # For comfortable growing temperatures
            else:
                self.gdd += temp - self.base_temp
            
            self.greenhouse_df['growing_degree_days'][i] = self.gdd

            if self.gdd >= self.gdd_req:
                self.gdd = 0
                self.harvest_times.append(self.hours_for_harvest)
                self.hours_for_harvest = 0
            else:
                self.hours_for_harvest += 1
                
        # Visualise the data.
        plt.plot(self.greenhouse_df['growing_degree_days'])
        plt.xlabel('Date')
        plt.ylabel('Growing degree hours')
        plt.title('Growing degree hours over time in greenhouse')
        plt.grid('off')
        axes = plt.gca()
        axes.set_ylim([0, 12000])
        plt.show()
        
        # How many resources were used per harvest
        print(self.resources_used_per_harvest, self.harvest_times)
        
        return self.greenhouse_df
        
    def calculate_greenhouse_production(self):
        pass

    def calculate_reward(self):
        '''
        Reward function is based on both resource use and time to harvest. Should
        probably be some points for a more stable temperature too. 0 is the perfect
        score albiet impossible, the closer to 0 the better.
        '''
        return -np.mean(self.harvest_times) - self.resources_used/len(self.harvest_times)

    def DeepQNetwork(self):
        '''
        A deep q algorithm for controlling greenhouse to optimise reward function.
        '''
        observation = 0 # Temperature from the greenhouse df.
        state = 0 # Temperature of the inside of the greenhouse.
        done = False
        
        # Build model
        model = Sequential()
        model.add(Dense(20, shape=(1,), init='uniform', activation='relu'))
        model.add(Dense(40, init='uniform', activation='relu'))
        model.add(Dense(40, init='uniform', activation='relu'))
        # Output is a probabality of making each action.
        model.add(Dense(4, init='uniform', activation='sigmoid'))
        model.compile(loss=self.loss_func, optimizer='adam', metrics=[self.loss_func])
        
    
    def simple_neural_net(self):
        '''
        A simple neural net to optimise the reward function.
        '''
        # Outside and inside temperature.
        n_inputs = 2
        n_hidden = 8
        # Fogging, screens, heating and vents.
        n_outputs = 4
        learning_rate = 0.01
        
        initializer = tf.contrib.layers.variance_scaling_initializer()
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        y = tf.placeholder(tf.float32, shape=[None, n_outputs])
        
        # Network creation.
        hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden, n_outputs)
        outputs = tf.nn.sigmoid(logits)
        p_on_or_off = tf.concat(axis=1, values=[outputs])
        
x = green_house()
df, greenhouse  = x.set_up_data()
print(greenhouse['5_hour_temp_avg'])
x.greenhouse_control_evaluation()
temps = x.calculate_greenhouse_temperature()
gdds = x.calculate_gdd()
reward = x.calculate_reward()
