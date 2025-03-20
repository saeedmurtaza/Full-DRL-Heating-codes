import gym   # python library interface helps in creation of the RL and provide a standard API to communicate between learning algorithms and environment
from stable_baselines3 import DQN  # to support reinforcement learning algorithms by offering an interface to train, evaluate agents and do hyperparameters tuning
from ODU1_env import ODUEnvironment  # To initilaize our environment

import pandas as pd  # to load data and analyse, manipulate and visualize the DataFrame
import numpy as np   # to deal with arrays and numerical values
import os  # to provide functions such as getting current directory
import pickle  # to load the trained models  

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

# The normalizing the inputs to transform features to be on a similar scale of 0 to 1 range

def normalize(raw_data):
    ST_min, ST_max = 18, 31
    IAT_min, IAT_max = 15, 35
    OAT_min, OAT_max = -20, 20
    OTW_min, OTW_max = -15, 10
    High_Pressure_min, High_Pressure_max = 5, 35
    Energy_min, Energy_max = 0, 15000


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

# Denormalizing for recovering the original value
def inverse_normalize(data):
    ST_min, ST_max = 18, 31
    IAT_min, IAT_max = 15, 35
    OAT_min, OAT_max = -20, 20
    OTW_min, OTW_max = -15, 10
    High_Pressure_min, High_Pressure_max = 5, 35
    Energy_min, Energy_max = 0, 15000

    data['ST'] =  data['ST'] * (ST_max - ST_min) + ST_min
    data['IAT'] = data['IAT'] * (IAT_max - IAT_min) + IAT_min
    data['OAT'] = data['OAT'] * (OAT_max - OAT_min) + OAT_min
    data['OTW'] = data['OTW'] * (OTW_max - OTW_min) + OTW_min
    data['High_Pressure'] = round(data['High_Pressure'] * (High_Pressure_max - High_Pressure_min) + High_Pressure_min, 0)
    data['Energy'] = data['Energy'] * (Energy_max - Energy_min) + Energy_min
    data['Delta_IAT_OAT'] = data['IAT'] - data['OAT']
    data['Delta_ST_IAT'] = data['ST'] - data['IAT']
    data['Delta_ST_OAT'] = data['ST'] - data['OAT']

    return data

# Make sure dataset is in the same location with this file, otherwise change below dataset directory 
Dataset = pd.read_csv("ALL_data_30min_train.csv")

# Read all data
data_raw = pd.read_csv("ALL_data_30min_test.csv")
all_data= normalize(data_raw.copy())
all_data['timestamp'] = pd.to_datetime(all_data.timestamp, dayfirst=False)
all_data['Date'] = all_data['timestamp'].dt.date
dates = all_data.Date.unique()
dates = dates.tolist()

# Make environment
env = ODUEnvironment(Dataset=Dataset)

# Load the trained agent
checkpoint_to_load = 'trained_DQN_agent'
model = DQN.load(checkpoint_to_load)

# Reset the environment and replace the random record from the dataset with the first entry values
state = env.reset()

# Load the forecasting models (energy and IAT)
energy_model = pickle.load(open(os.path.join(os.getcwd(),'Energy.pkl'), 'rb'))
IAT_model = pickle.load(open(os.path.join(os.getcwd(),'IAT.pkl'), 'rb'))

# Create empty list for later use
test_results = []  # to hold the RL simulation results
time_of_day = []   # to hold the all timestamp


for date in dates:
    sample = all_data[all_data['Date'] == date]
    all_dates = sample[['timestamp']]


# 
    RL_observation_data = sample[['Day_sin', 'Day_cos', 'ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
    IAT_data = sample[['ST', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','IAT']]
    energy_data = sample[['ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
    OAT_data = sample[['OAT']]


    # Create arrays that will hold data for the forecasting models
    # a) For IAT
    IAT_data_array = np.zeros((len(IAT_data),14))
    IAT_data_array [0] = IAT_data.iloc[0]
    IAT_data_array [1] = IAT_data.iloc[1]

    # b) for energy
    energy_data_array = np.zeros((len(energy_data),15))
    energy_data_array [0] = energy_data.iloc[0]
    energy_data_array [1] = energy_data.iloc[1]

    # Define a set of action 
    actions_represented = [(0.7, 0), (0.7, 1), (0.73333333, 1), (0.76666667, 1), (0.8, 1), (0.83333333, 1)]  # Normalized target pressure values


    length = len(RL_observation_data)-1
    start = 2
    if RL_observation_data.shape[0] <= 2:    # condition to check the initial record
        continue
    # Create variables for later USE
    initial_record = RL_observation_data.iloc[start]
    initial_state = np.float32(initial_record)
    next_record = []   # to hold next record of the environment after an agent takes an action
    next_state = []    # to hold the forecasted states

    while start < length:
        time_of_day.append(all_dates.iloc[start].values[0])

        if len(next_state) == 0:
            next_record = initial_record
            next_state = initial_state
    # to choose new action on the next state
        action, _states = model.predict(next_state)

        High_Pressure = actions_represented[action][0]
       

        Off_mode = 0
        On_mode = 1

        if (actions_represented[action][1]) == 0:
             # if the Op mode is 0
            Off_mode = 1 # for op_mode_0
            On_mode = 0 # for op_mode_1

        # Step 1: IAT forecast based on ofrecasted OAT and action (LP, Op_mode)
        IAT_forecast_array = np.zeros((3,14))
        IAT_forecast_array[0] = IAT_data_array[start - 2]
        IAT_forecast_array[1] = IAT_data_array[start - 1]

        # Adding current observation
        IAT_forecast_array[2,0] = next_state[2] # ST
        IAT_forecast_array[2,1] = next_state[4] # OAT
        IAT_forecast_array[2,2] = next_state[5] # Delta_IAT_OAT
        IAT_forecast_array[2,3] = next_state[6] # Delta_ST_IAT
        IAT_forecast_array[2,4] = next_state[7] # Delta_ST_OAT
        IAT_forecast_array[2,5] = High_Pressure # using the TP chosen by the agent
        IAT_forecast_array[2,6] = next_state[0] # Day_sin
        IAT_forecast_array[2,7] = next_state[1] # Day_cos
        # Using the op mode chosen by the agent
        IAT_forecast_array[2,8] = Off_mode # for op_mode_0
        IAT_forecast_array[2,9] = On_mode # for op_mode_1
        IAT_forecast_array[2,10] = next_state[10]
        IAT_forecast_array[2,11] = next_state[11]
        IAT_forecast_array[2,12] = next_state[12]
        IAT_forecast_array[2,13] = next_state[3] # IAT
        
        # Arranging inputs for forecasting
        IAT_forecast_array_reframed = series_to_supervised(IAT_forecast_array, 2, 1)
        IAT_forecast_array_reframed.drop(['var3(t)', 'var4(t)', 'var14(t)'], axis =1, inplace=True)
        IAT_forecast_array_reframed = IAT_forecast_array_reframed.values
        IAT_model_inputs = IAT_forecast_array_reframed
        forecasted_IAT = IAT_model.predict(IAT_model_inputs)[0]

        # Substituting IAT values with the forecasted value
        IAT_forecast_array[2,13] = forecasted_IAT
        # Recalculating the differences because of forecasted IAT
        IAT_forecast_array[2,2] = IAT_forecast_array[2,13] - IAT_forecast_array[2,1] # Delta_IAT_OAT
        IAT_forecast_array[2,3] = IAT_forecast_array[2,0] - IAT_forecast_array[2,13] # Delta_ST_IAT
        # Appending to the array that holds IAT observations
        IAT_data_array[start] = IAT_forecast_array[2]

        # Step 2: Energy forecasting
        # Use ST, OAT, IAT observed/forecasted values
        energy_forecast_array = np.zeros((3,15))
        energy_forecast_array[0] = energy_data_array[start - 2]
        energy_forecast_array[1] = energy_data_array[start - 1]
        
        # adding current observation
        energy_forecast_array[2,0] = next_state[2] # ST
        energy_forecast_array[2,1] = forecasted_IAT # using the forecasted IAT
        energy_forecast_array[2,2] = next_state[4] # OAT
        energy_forecast_array[2,3] = next_state[5] # Delta_IAT_OAT
        energy_forecast_array[2,4] = next_state[6] # Delta_ST_IAT
        energy_forecast_array[2,5] = next_state[7] # Delta_ST_OAT
        energy_forecast_array[2,6] = High_Pressure # using the TP chosen by the agent
        energy_forecast_array[2,7] = next_state[0]
        energy_forecast_array[2,8] = next_state[1]
        energy_forecast_array[2,9] = Off_mode # for op_mode_0
        energy_forecast_array[2,10] = On_mode # for op_mode_1
        energy_forecast_array[2,11] = next_state[10] # Year_sin
        energy_forecast_array[2,12] = next_state[11] # Year_cos
        energy_forecast_array[2,13] = next_state[12] # OTW
        energy_forecast_array[2,14] = next_state[13] # Energy

        # the energy value will be updated after forecasting
        # recalculating the differences because of ST and forecasted OAT, forecasted IAT
        energy_forecast_array[2,3] = energy_forecast_array[2,1] - energy_forecast_array[2,2] # Delta_IAT_OAT
        energy_forecast_array[2,4] = energy_forecast_array[2,0] - energy_forecast_array[2,1] # Delta_ST_IAT
        
        
        # arranging inputs for forecasting
        energy_forecast_array_reframed = series_to_supervised(energy_forecast_array, 2, 1)
        energy_forecast_array_reframed = energy_forecast_array_reframed.values
        energy_model_inputs = energy_forecast_array_reframed[:, :-1]
        forecasted_energy = energy_model.predict(energy_model_inputs)[0]
        energy_forecast_array[2,14] = forecasted_energy

        energy_data_array[start] = energy_forecast_array[2]
        test_results.append(energy_forecast_array[2])

        # Next state
        start +=1
        next_record = RL_observation_data.iloc[start]
        next_state = np.float32(next_record)
        
        # Adding the forecasted values to the next record, and calculating the temperature differences
        next_state[3] = forecasted_IAT
        next_state[5] = next_state[3] - next_state[4] # Delta_IAT_OAT
        next_state[6] = next_state[2]- next_state[3] # Delta_ST_IAT
        next_state[7] = next_state[2] - next_state[4] # Delta_ST_OAT
        next_state[8] = High_Pressure # TP
        next_state[9] = On_mode # Op_mode
        next_state[13] = forecasted_energy # Energy


test_results_df = pd.DataFrame(test_results, columns=["ST","IAT","OAT","Delta_IAT_OAT","Delta_ST_IAT","Delta_ST_OAT","High_Pressure","Day_sin","Day_cos","Op_mode_0.0","Op_mode_1.0","Year_sin","Year_cos","OTW","Energy"])
time_of_day_df = pd.DataFrame(time_of_day, columns=['timestamp'])
data_to_save_normalized = pd.concat([time_of_day_df, test_results_df], axis=1)

data_to_save = inverse_normalize(data_to_save_normalized.copy())
data_to_save.to_csv("RL_results.csv", index=False)


