from stable_baselines3 import DQN # to support reinforcement learning algorithms by offering an interface to train, evaluate agents and do hyperparameters tuning
from ODU1_env import ODUEnvironment # To initilaize our environment
import pandas as pd # to load data into DataFrame
import numpy as np  # to deal with arrays and numerical values
import os   #  to provide functions such as getting current directory
import pickle  # to load the trained models
from utils import *  # to use the contained functions and variables
from datetime import datetime, timedelta # to get date and time # to represent the difference between two dates or times
pd.options.mode.chained_assignment = None # to ignore the pandas warnings


# Connect to the Db
connection = sqlite3.connect(r"C:\Users\user\Desktop\NED_SQLite.db")


Dataset = get_odu1_data()

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
checkpoint_to_load = r"C:\Users\user\Desktop\Synology_Friday_backup\Updated DRL Heating training and deployment codes_20230818\3_Deployment\2_ODU#1\trained_DQN_agent.zip"
model = DQN.load(checkpoint_to_load)

# Reset the environment and replace the random record from the dataset
# with the first entry instead
state = env.reset()

# Load the forecasting models
energy_model = pickle.load(open(os.path.join(r"C:\Users\user\Desktop\Synology_Friday_backup\Updated DRL Heating training and deployment codes_20230818\3_Deployment\2_ODU#1\Energy.pkl"), 'rb'))
IAT_model = pickle.load(open(os.path.join(r"C:\Users\user\Desktop\Synology_Friday_backup\Updated DRL Heating training and deployment codes_20230818\3_Deployment\2_ODU#1\IAT.pkl"), 'rb'))

# Create empty list for later use
test_results = [] # to hold the RL simulation results
time_of_day = []  # to hold the all timestamp


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
# Define a set of action 
actions_represented = [(0.7, 0), (0.7, 1), (0.73333333, 1), (0.76666667, 1), (0.8, 1), (0.83333333, 1)]  # Normalized target pressure values

# length of possible actions finite set 
length = 6
start = 2

initial_record = RL_observation_data.iloc[start]
initial_state = np.float32(initial_record)
next_record = []  # to hold the next DRL state
next_state = []   # to hold the forecasted states (np array)

time = datetime.now() + timedelta(minutes=30)

# iterate in discrete spaces
while start < length:

    if len(next_state) == 0:
        next_record = initial_record
        next_state = initial_state
# to choose an action for the next state
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

    # step 1: IAT forecast based on forecasted OAT and action (LP, Op_mode)
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

    # to check the forecasted IAT
    print (f"forecasted_IAT: {forecasted_IAT:.2f}")
    # Substituting IAT values with the forecasted value
    IAT_forecast_array[2,13] = forecasted_IAT
    # recalculating the differences because of forecasted IAT
    IAT_forecast_array[2,2] = IAT_forecast_array[2,13] - IAT_forecast_array[2,1] # Delta_IAT_OAT
    IAT_forecast_array[2,3] = IAT_forecast_array[2,0] - IAT_forecast_array[2,13] # Delta_ST_IAT
    # Appending to the array that holds IAT observations

    IAT_data_array[start] = IAT_forecast_array[2]

    # step 2: Energy forecasting
    # Use OAT, IAT observed/forecasted values
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

    # to check the forecasted energy
    print (f"forecasted_energy: {forecasted_energy:.2f}")

    energy_forecast_array[2,14] = forecasted_energy
    energy_data_array[start] = energy_forecast_array[2]
    test_results.append(energy_forecast_array[2])

    # Next state
    start += 1
    next_state = np.zeros((1,14), dtype="float32")
    next_state = next_state.ravel()
    # Adding the forecasted values to the next record, and calculating the temperature differences
    time = time + timedelta(minutes=30)
    next_state[0] = get_sine_time(time)
    next_state[1] = get_cosine_time(time)
    next_state[3] = forecasted_IAT # forecasted value
    next_state[4] = outdoor_temp # forecasted value
    next_state[5] = next_state[3] - next_state[4] # Delta_IAT_OAT
    next_state[6] = next_state[2]- next_state[3] # Delta_ST_IAT
    next_state[7] = next_state[2] - next_state[4] # Delta_ST_OAT
    next_state[8] = High_Pressure # TP
    next_state[9] = On_mode # Op_mode
    next_state[10] = get_sin_year(time)
    next_state[11] = get_cos_year(time)
    next_state[12] = otw # OTW
    next_state[13] = forecasted_energy

# save the results into DataFrame
test_results_df = pd.DataFrame(test_results, columns=['ST', 'IAT', 'OAT', 'Delta_IAT_OAT', 'Delta_ST_IAT', 'Delta_ST_OAT', 'High_Pressure', 'Day_sin', 'Day_cos', 'Op_Mode_0', 'Op_Mode_1', 'Year_sin','Year_cos','OTW','Energy'])
time_of_day_df = pd.DataFrame(time_of_day, columns=['timestamp'])
data_to_save_normalized = pd.concat([time_of_day_df, test_results_df], axis=1)
data_to_save = inverse_normalize(data_to_save_normalized.copy())

# creating drl policy 
drl_policy = data_to_save[['timestamp']]
drl_policy['timestamp'] = drl_policy['timestamp'].dt.round('1s')
drl_policy['timestamp'] = drl_policy['timestamp'].map(lambda x : x.replace(second=0))
drl_policy["ODU_targetpress"] = data_to_save[ "High_Pressure"].astype(int)
drl_policy['ODU_onoff'] = data_to_save['Op_Mode_1'].astype(int)
drl_policy['ODU_onoff'].replace({1: "일반 운전", 0: "정지"}, inplace=True)


# Updating the policy table
# Step 1. Saving the new policy to a temporary table
drl_policy.to_sql('oud1_policy_temp', connection, if_exists='replace', index=False)

# Step 2: Read the data from the main table
sql = "SELECT * FROM odu1_drl_policy;"
db_data = pd.read_sql(sql, connection)


# Step 3: Read the policy saved in the temporary table
sql = "SELECT * FROM oud1_policy_temp;"
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

db_data_to_save.to_sql('odu1_drl_policy', connection, if_exists='replace', index=False)