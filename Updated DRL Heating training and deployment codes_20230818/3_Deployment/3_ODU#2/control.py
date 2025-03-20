from stable_baselines3 import DQN
from ODU2_env import ODUEnvironment

import pandas as pd
import numpy as np
import os
import pickle
from utils import *
from datetime import datetime, timedelta
pd.options.mode.chained_assignment = None


# Connect to the Db
connection = sqlite3.connect('C:/Users/HUBEL/Desktop/Backup_Patrick/01. Research projects/02_Samsung_NED/05_2023_heating/Heating DRL/3_Deployment/2_ODU#1/NED_SQLite.db')


Dataset = get_odu2_data()

# Read all data
data_raw = Dataset.copy()
all_data= normalize(data_raw.copy())
all_data['timestamp'] = pd.to_datetime(all_data.timestamp)
all_data['Date'] = all_data['timestamp'].dt.date
dates = all_data.Date.unique()
dates = dates.tolist()



# Initialize environment
env = ODUEnvironment(Dataset=Dataset)


# Load the trained agent
checkpoint_to_load = 'C:/Users/HUBEL/Desktop/Backup_Patrick/01. Research projects/02_Samsung_NED/05_2023_heating/Heating DRL/3_Deployment/3_ODU#2/trained_DQN_agent'
model = DQN.load(checkpoint_to_load)

# Reset the environment and replace the random record from the dataset
# with the first entry instead
state = env.reset()

# Load the forecasting models
# Load the forecasting models
energy_model = pickle.load(open(os.path.join("C:/Users/HUBEL/Desktop/Backup_Patrick/01. Research projects/02_Samsung_NED/05_2023_heating/Heating DRL/3_Deployment/3_ODU#2/",'Energy.pkl'), 'rb'))
IAT_model = pickle.load(open(os.path.join("C:/Users/HUBEL/Desktop/Backup_Patrick/01. Research projects/02_Samsung_NED/05_2023_heating/Heating DRL/3_Deployment/3_ODU#2/",'IAT.pkl'), 'rb'))


test_results = []
time_of_day = []



RL_observation_data = all_data[['Day_sin', 'Day_cos', 'ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
IAT_data = all_data[['ST', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','IAT']]
energy_data = all_data[['ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy']]
OAT_data = all_data[['OAT']]


# Create arrays that will hold data for the forecasting models
# a) For IAT
IAT_data_array = np.zeros((7,14))
IAT_data_array [0] = IAT_data.iloc[0]
IAT_data_array [1] = IAT_data.iloc[1]

# b) for energy
energy_data_array = np.zeros((7,15))
energy_data_array [0] = energy_data.iloc[0]
energy_data_array [1] = energy_data.iloc[1]

actions_represented = [(0.7, 0), (0.7, 1), (0.73333333, 1), (0.76666667, 1), (0.8, 1), (0.83333333, 1)]


length = 6
start = 2

initial_record = RL_observation_data.iloc[start]
initial_state = np.float32(initial_record)
next_record = []
next_state = []

time = datetime.now() + timedelta(minutes=30)

while start < length:

    if len(next_state) == 0:
        next_record = initial_record
        next_state = initial_state

    action, _states = model.predict(next_state)

    High_Pressure = actions_represented[action][0]

    outdoor_temp, outdoor_rh = get_outdoor_data()
    otw = compute_WBT(outdoor_temp, outdoor_rh)
    time_of_day.append(time)

    Off_mode = 0
    On_mode = 1

    if (actions_represented[action][1]) == 0:
            # if the Op mode is 0
        Off_mode = 1 # for op_mode_0
        On_mode = 0 # for op_mode_1

    # step 1: IAT forecast based on ofrecasted OAT and action (LP, Op_mode)
    IAT_forecast_array = np.zeros((3,14))
    IAT_forecast_array[0] = IAT_data_array[start - 2]
    IAT_forecast_array[1] = IAT_data_array[start - 1]

    # adding current observation
    IAT_forecast_array[2,0] = next_state[2] # ST
    IAT_forecast_array[2,1] = outdoor_temp # OAT
    IAT_forecast_array[2,2] = next_state[5] # Delta_IAT_OAT
    IAT_forecast_array[2,3] = next_state[6] # Delta_ST_IAT
    IAT_forecast_array[2,4] = next_state[7] # Delta_ST_OAT
    IAT_forecast_array[2,5] = High_Pressure # using the TP chosen by the agent
    IAT_forecast_array[2,6] = next_state[0] # day_sine
    IAT_forecast_array[2,7] = next_state[1] # day_cosine
    # using the op mode chosen by the agent
    IAT_forecast_array[2,8] = Off_mode # for op_mode_0
    IAT_forecast_array[2,9] = On_mode # for op_mode_1
    IAT_forecast_array[2,10] = next_state[10]  # YEar sine
    IAT_forecast_array[2,11] = next_state[11] # Year cosine
    IAT_forecast_array[2,12] = otw # OTW
    IAT_forecast_array[2,13] = next_state[3] # IAT

    # recalculating the differences because of ST and OAT
    IAT_forecast_array[2, 2] = IAT_forecast_array[2, 13] - IAT_forecast_array[2, 1]  # Delta_IAT_OAT
    IAT_forecast_array[2, 3] = IAT_forecast_array[2, 0] - IAT_forecast_array[2, 13]  # Delta_ST_IAT
    IAT_forecast_array[2, 4] = IAT_forecast_array[2, 0] - IAT_forecast_array[2, 1]  # Delta_ST_OAT

    # arranging inputs for forecasting
    IAT_forecast_array_reframed = series_to_supervised(IAT_forecast_array, 2, 1)
    IAT_forecast_array_reframed.drop(['var3(t)', 'var4(t)', 'var14(t)'], axis =1, inplace=True)
    IAT_forecast_array_reframed = IAT_forecast_array_reframed.values
    IAT_model_inputs = IAT_forecast_array_reframed
    forecasted_IAT = IAT_model.predict(IAT_model_inputs)[0]
    # SUbstituting IAT values with the forecasted value
    IAT_forecast_array[2,13] = forecasted_IAT
    # recalculating the differences because of forecasted IAT
    IAT_forecast_array[2,2] = IAT_forecast_array[2,13] - IAT_forecast_array[2,1] # Delta_IAT_OAT
    IAT_forecast_array[2,3] = IAT_forecast_array[2,0] - IAT_forecast_array[2,13] # Delta_ST_IAT
    # Appending to the array that holds IAT observations
    IAT_data_array[start] = IAT_forecast_array[2]

    # step 2: Energy forecasting
    # Use ST, OAT, IAT observed/forecasted values
    energy_forecast_array = np.zeros((3,15))
    energy_forecast_array[0] = energy_data_array[start - 2]
    energy_forecast_array[1] = energy_data_array[start - 1]
    # adding current observation
    energy_forecast_array[2,0] = next_state[2] # ST
    energy_forecast_array[2,1] = forecasted_IAT # using the forecasted IAT
    energy_forecast_array[2,2] = outdoor_temp  # OAT
    energy_forecast_array[2,3] = next_state[5] # Delta_IAT_OAT
    energy_forecast_array[2,4] = next_state[6] # Delta_ST_IAT
    energy_forecast_array[2,5] = next_state[7] # Delta_ST_OAT
    energy_forecast_array[2,6] = High_Pressure # using the TP chosen by the agent
    energy_forecast_array[2,7] = next_state[0]
    energy_forecast_array[2,8] = next_state[1]
    energy_forecast_array[2,9] = Off_mode # for op_mode_0
    energy_forecast_array[2,10] = On_mode # for op_mode_1
    energy_forecast_array[2,11] = next_state[10]
    energy_forecast_array[2,12] = next_state[11]
    energy_forecast_array[2,13] = otw # OTW
    energy_forecast_array[2,14] = next_state[13]
    # the energy value will be updated after forecasting
    # recalculating the differences because of ST and forecasted OAT, forecasted IAT
    energy_forecast_array[2,3] = energy_forecast_array[2,1] - energy_forecast_array[2,2] # Delta_IAT_OAT
    energy_forecast_array[2,4] = energy_forecast_array[2,0] - energy_forecast_array[2,1] # Delta_ST_IAT
    energy_forecast_array[2,5] = energy_forecast_array[2,0] - energy_forecast_array[2,2] # Delta_ST_OAT
    # arranging inputs for forecasting
    energy_forecast_array_reframed = series_to_supervised(energy_forecast_array, 2, 1)
    energy_forecast_array_reframed = energy_forecast_array_reframed.values
    energy_model_inputs = energy_forecast_array_reframed[:, :-1]
    forecasted_energy = energy_model.predict(energy_model_inputs)[0]
    energy_forecast_array[2,14] = forecasted_energy
    energy_data_array[start] = energy_forecast_array[2]
    test_results.append(energy_forecast_array[2])

    # Next state
    start += 1
    next_state = np.zeros((1,14), dtype="float32")
    next_state = next_state.ravel()
    # Adding the forecasted values to the next record, and calculating the differences
    time = time + timedelta(minutes=30)
    next_state[0] = get_sine_time(time)
    next_state[1] = get_cosine_time(time)
    next_state[3] = forecasted_IAT
    next_state[4] = outdoor_temp
    next_state[5] = next_state[3] - next_state[4] # Delta_IAT_OAT
    next_state[6] = next_state[2]- next_state[3] # Delta_ST_IAT
    next_state[7] = next_state[2] - next_state[4] # Delta_ST_OAT
    next_state[8] = High_Pressure # TP
    next_state[9] = On_mode # Op_mode
    next_state[10] = get_sin_year(time)
    next_state[11] = get_cos_year(time)
    next_state[12] = otw # OTW
    next_state[13] = forecasted_energy


test_results_df = pd.DataFrame(test_results, columns=['ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy'])

time_of_day_df = pd.DataFrame(time_of_day, columns=['timestamp'])
data_to_save_normalized = pd.concat([time_of_day_df, test_results_df], axis=1)

data_to_save = inverse_normalize(data_to_save_normalized.copy())

drl_policy = data_to_save[['timestamp']]
drl_policy['timestamp'] = drl_policy['timestamp'].dt.round('1s')
drl_policy['timestamp'] = drl_policy['timestamp'].map(lambda x : x.replace(second=0))
drl_policy["ODU_targetpress"] = data_to_save[ "High_Pressure"].astype(int)
drl_policy['ODU_onoff'] = data_to_save['Op_Mode_1'].astype(int)


# Updating the policy table
# Step 1. Saving the new policy to a temporary table
drl_policy.to_sql('oud2_policy_temp', connection, if_exists='replace', index=False)

# Step 2: Read the data from the main table
sql = "SELECT * FROM odu2_drl_policy;"
db_data = pd.read_sql(sql, connection)


# Step 3: Read the policy saved in the temporary table
sql = "SELECT * FROM oud2_policy_temp;"
read_tmp_data = pd.read_sql(sql, connection)

# Step 3:Comparing data
data_to_update = read_tmp_data[~(read_tmp_data['timestamp'].isin(db_data['timestamp']))].reset_index(drop=True)

# Updating and saving the data
data_to_update = data_to_update.set_index('timestamp')
db_data = db_data.set_index('timestamp')
read_tmp_data = read_tmp_data.set_index('timestamp')
db_data.update(read_tmp_data)
db_data_to_save = pd.concat([db_data,data_to_update])

db_data_to_save.reset_index(inplace=True)

db_data_to_save.to_sql('odu2_drl_policy', connection, if_exists='replace', index=False)