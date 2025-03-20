
import pandas as pd
import numpy as np
import gym
from gym import spaces
import os
import pickle

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def normalize(raw_data):
    ST_min, ST_max = 18, 31
    IAT_min, IAT_max = 15, 35
    OAT_min, OAT_max = -20, 20
    OTW_min, OTW_max = -15, 10
    High_Pressure_min, High_Pressure_max = 5, 35
    Energy_min, Energy_max = 0, 18000


    raw_data['ST'] = (raw_data['ST'] - ST_min) / (ST_max - ST_min)
    raw_data['IAT'] = (raw_data['IAT'] - IAT_min) / (IAT_max - IAT_min)
    raw_data['OAT'] = (raw_data['OAT'] - OAT_min) / (OAT_max - OAT_min)
    raw_data['High_Pressure'] = (raw_data['High_Pressure'] - High_Pressure_min) / (High_Pressure_max - High_Pressure_min)
    raw_data['Delta_IAT_OAT'] = raw_data['IAT'] - raw_data['OAT']
    raw_data['Delta_ST_IAT'] = raw_data['ST'] - raw_data['IAT']
    raw_data['Delta_ST_OAT'] = raw_data['ST'] - raw_data['OAT']
    raw_data['OTW'] = (raw_data['OTW'] - OTW_min) / (OTW_max - OTW_min)
    raw_data['Energy'] = (raw_data['Energy'] - Energy_min) / (Energy_max - Energy_min)

    return raw_data


def inverse_normalization_IAT_and_ST(IAT, ST):
    IAT_min, IAT_max = 15, 35
    ST_min, ST_max = 18, 31
    iat = IAT * (IAT_max - IAT_min) + IAT_min
    st = ST * (ST_max - ST_min) + ST_min
    return iat, st

# Inheriting our environment directly from gym.Env class
class ODUEnvironment(gym.Env):
    def __init__(self, Dataset):

        # Read all datasets
        data_raw = Dataset
        dataset= normalize(data_raw.copy())
        self.RL_observation_data = dataset[['Day_sin', 'Day_cos', 'ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
        self.IAT_data = dataset[['ST', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','IAT']]
        self.energy_data = dataset[['ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
        self.OAT_data = dataset[['OAT']]


        # Create arrays that will hold data for the forecasting models
        # a) For IAT
        # self.IAT_data_array = self.IAT_data.values
        self.IAT_data_array = np.zeros((len(self.IAT_data),14))
        self.IAT_data_array [0] = self.IAT_data.iloc[0]
        self.IAT_data_array [1] = self.IAT_data.iloc[1]

        # b) for energy
        # self.energy_data_array = self.energy_data.values
        self.energy_data_array = np.zeros((len(self.energy_data),15))
        self.energy_data_array [0] = self.energy_data.iloc[0]
        self.energy_data_array [1] = self.energy_data.iloc[1]

        # Get bounds for all observations

        self.Day_sin_min = -1
        self.Day_sin_max = 1

        self.Day_cos_min = -1
        self.Day_cos_max = 1

        self.ST_min = 0
        self.ST_max = 1

        self.IAT_min = 0
        self.IAT_max = 1

        self.OAT_min = 0
        self.OAT_max = 1

        self.Delta_IAT_OAT_min = -1
        self.Delta_IAT_OAT_max = 1

        self.Delta_ST_IAT_min = -1
        self.Delta_ST_IAT_max = 1

        self.Delta_ST_OAT_min = -1
        self.Delta_ST_OAT_max = 1

        self.High_Pressure_min = 0
        self.High_Pressure_max = 1

        self.Op_Mode_min = 0
        self.Op_Mode_max = 1
        
        self.Year_sin_min = -1
        self.Year_sin_max = 1

        self.Year_cos_min = -1
        self.Year_cos_max = 1
        
        self.OTW_min = 0
        self.OTW_max = 1

        self.Energy_min = 0
        self.Energy_max = 1




        # Compose a Box observation/state space, i.e. continuous space that is bounded by the above values
        self.low_state = np.array([self.Day_sin_min, self.Day_cos_min, self.ST_min, self.IAT_min, self.OAT_min, self.Delta_IAT_OAT_min,
                                    self.Delta_ST_IAT_min, self.Delta_ST_OAT_min, self.High_Pressure_min, self.Op_Mode_min, self.Year_sin_min, self.Year_cos_min, self.OTW_min, self.Energy_min])
        
        self.high_state =np.array([self.Day_sin_max, self.Day_cos_max, self.ST_max, self.IAT_max, self.OAT_max, self.Delta_IAT_OAT_max,
                                    self.Delta_ST_IAT_max, self.Delta_ST_OAT_max, self.High_Pressure_max, self.Op_Mode_max, self.Year_sin_max, self.Year_cos_max, self.OTW_max, self.Energy_max])
        
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)



        # The action space is Discrete(TP, op mode)
        self.actions_represented = [(0.7, 0), (0.7, 1), (0.73333333, 1), (0.76666667, 1), (0.8, 1), (0.83333333, 1)]
        self.action_space = spaces.Discrete(6)

        # Load the forecasting models
        self.energy_model = pickle.load(open(os.path.join(os.getcwd(),'Energy.pkl'), 'rb'))
        self.IAT_model = pickle.load(open(os.path.join(os.getcwd(),'IAT.pkl'), 'rb'))

        self.length = len(self.RL_observation_data)-1  # Calculating the number of rows in the data (starting with the row at 0th position)

        self.start = 2 # values at index less that 2 are used for the prediction models initialization

        # Create variables for later USE
        self.initial_record = self.RL_observation_data.iloc[self.start]
        self.initial_state = np.float32(self.initial_record)
        self.next_record = []
        self.next_state = []
        self.reward_list = []



    def reset(self):

        self.start = 2
        self.initial_record = self.RL_observation_data.iloc[self.start]
        self.initial_state = np.float32(self.initial_record)

        return self.initial_state



    def step(self, action):
        action = action

        High_Pressure = self.actions_represented[action][0]
        Off_mode = 0
        On_mode = 1

        if (self.actions_represented[action][1]) == 0:
             # if the Op mode is 0
            Off_mode = 1 # for op_mode_0
            On_mode = 0 # for op_mode_1

        # If it's the first step
        if len(self.next_state) == 0:
            self.next_record = self.initial_record
            self.next_state = self.initial_state

        
        # step 1: IAT forecast based on forecasted OAT and action (HP, Op_Mode)
        IAT_forecast_array = np.zeros((3,14))
        IAT_forecast_array[0] = self.IAT_data_array[self.start - 2]
        IAT_forecast_array[1] = self.IAT_data_array[self.start - 1]

        

        # adding current observation
        IAT_forecast_array[2,0] = self.next_state[2] # ST
        IAT_forecast_array[2,1] = self.next_state[4] # OAT
        IAT_forecast_array[2,2] = self.next_state[5] # Delta_IAT_OAT
        IAT_forecast_array[2,3] = self.next_state[6] # Delta_ST_IAT
        IAT_forecast_array[2,4] = self.next_state[7] # Delta_ST_OAT
        IAT_forecast_array[2,5] = High_Pressure # using the HP chosen by the agent
        IAT_forecast_array[2,6] = self.next_state[0] # Day_sin
        IAT_forecast_array[2,7] = self.next_state[1] # Day_cos
        # using the op mode chosen by the agent
        IAT_forecast_array[2,8] = Off_mode # for op_mode_0
        IAT_forecast_array[2,9] = On_mode # for op_mode_1
        IAT_forecast_array[2,10] = self.next_state[10] # Year_sin
        IAT_forecast_array[2,11] = self.next_state[11] #Year_cos
        IAT_forecast_array[2,12] = self.next_state[12] # OTW
        IAT_forecast_array[2,13] = self.next_state[3] # IAT

        # step 2:
        # arranging inputs for forecasting
        IAT_forecast_array_reframed = series_to_supervised(IAT_forecast_array, 2, 1)
        IAT_forecast_array_reframed.drop(['var3(t)', 'var4(t)', 'var14(t)'], axis =1, inplace=True)
        IAT_forecast_array_reframed = IAT_forecast_array_reframed.values
        IAT_model_inputs = IAT_forecast_array_reframed
        forecasted_IAT = self.IAT_model.predict(IAT_model_inputs)[0]
        
        # SUbstituting IAT values with the forecasted value
        IAT_forecast_array[2,13] = forecasted_IAT
        # recalculating the differences because of forecasted IAT
        IAT_forecast_array[2,2] = IAT_forecast_array[2,13] - IAT_forecast_array[2,1] # Delta_IAT_OAT
        IAT_forecast_array[2,3] = IAT_forecast_array[2,0] - IAT_forecast_array[2,13] # Delta_ST_IAT
        # Appending to the array that holds IAT observations
        self.IAT_data_array[self.start] = IAT_forecast_array[2]


        # step 3: Energy forecasting
        # Use ST, OAT, IAT observed/forecasted values
        energy_forecast_array = np.zeros((3,15))
        energy_forecast_array[0] = self.energy_data_array[self.start - 2]
        energy_forecast_array[1] = self.energy_data_array[self.start - 1]
        # adding current observation
        energy_forecast_array[2,0] = self.next_state[2] # ST
        energy_forecast_array[2,1] = forecasted_IAT # using the forecasted IAT
        energy_forecast_array[2,2] = self.next_state[4] # OAT
        energy_forecast_array[2,3] = self.next_state[5] # Delta_IAT_OAT
        energy_forecast_array[2,4] = self.next_state[6] # Delta_ST_IAT
        energy_forecast_array[2,5] = self.next_state[7] # Delta_ST_OAT
        energy_forecast_array[2,6] = High_Pressure # using the HP chosen by the agent
        energy_forecast_array[2,7] = self.next_state[0]
        energy_forecast_array[2,8] = self.next_state[1]
        energy_forecast_array[2,9] = Off_mode # for op_mode_0
        energy_forecast_array[2,10] = On_mode # for op_mode_1
        energy_forecast_array[2,11] = self.next_state[10]
        energy_forecast_array[2,12] = self.next_state[11]
        energy_forecast_array[2,13] = self.next_state[12]
        energy_forecast_array[2,14] = self.next_state[13]
        
        # the energy value will be updated after forecasting
        # recalculating the differences because of ST and forecasted OAT, forecasted IAT
        energy_forecast_array[2,3] = energy_forecast_array[2,1] - energy_forecast_array[2,2] # Delta_IAT_OAT
        energy_forecast_array[2,4] = energy_forecast_array[2,0] - energy_forecast_array[2,1] # Delta_ST_IAT
        # energy_forecast_array[2,6] = energy_forecast_array[2,0] - energy_forecast_array[2,3] # Delta_ST_OAT
        
        # arranging inputs for forecasting
        energy_forecast_array_reframed = series_to_supervised(energy_forecast_array, 2, 1)
        energy_forecast_array_reframed = energy_forecast_array_reframed.values
        energy_model_inputs = energy_forecast_array_reframed[:, :-1]
        forecasted_energy = self.energy_model.predict(energy_model_inputs)[0]

        energy_forecast_array[2,14] = forecasted_energy
        self.energy_data_array[self.start] = energy_forecast_array[2]

        Set_temp = self.next_state[2]
        iat, st = inverse_normalization_IAT_and_ST(forecasted_IAT, Set_temp)

        if(st - iat) > 2:
            reward = -50 - 2*forecasted_energy

        elif 1.5 < (st - iat) <=2 :
            reward = -30 - 3*forecasted_energy

        elif 1 < (st - iat) <= 1.5 :
            reward = -10 - 4*forecasted_energy

        elif 0.5 < (st - iat) <= 1 :
            reward = -5 - 5*forecasted_energy

        elif 0 < (st - iat) <= 0.5 :
            reward = 0 - 5*forecasted_energy

        elif -0.5 < (st - iat) <= 0 :
            reward = 10 + 5/forecasted_energy

        elif -1 < (st - iat) <= -0.5 :
            reward = -30 - 15*forecasted_energy

        else:
            reward = -50 - 20*forecasted_energy


        # total reward
        reward = reward

        # Next state
        self.start +=1
        self.next_record = self.RL_observation_data.iloc[self.start]
        self.next_state = np.float32(self.next_record)
        # Adding the forecasted values to the next record, and calculating the differences

        self.next_state[3] = forecasted_IAT
        self.next_state[5] = self.next_state[3] - self.next_state[4] # Delta_IAT_OAT
        self.next_state[6] = self.next_state[2]- self.next_state[3] # Delta_ST_IAT
        self.next_state[7] = self.next_state[2] - self.next_state[4] # Delta_ST_OAT
        self.next_state[8] = High_Pressure # TP
        self.next_state[9] = On_mode # Op_mode
        self.next_state[13] = forecasted_energy

        if self.start == self.length:
            done = True
        else:
            done = False

        info = {}
        self.reward_list.append(reward)
        return self.next_state, reward, done, info
