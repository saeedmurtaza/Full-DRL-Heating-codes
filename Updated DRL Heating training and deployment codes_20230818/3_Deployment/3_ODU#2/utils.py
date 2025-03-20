import pandas as pd
import numpy as np
import sqlite3

# Connect to the Db
connection = sqlite3.connect('C:/Users/HUBEL/Desktop/Backup_Patrick/01. Research projects/02_Samsung_NED/05_2023_heating/Heating DRL/3_Deployment/2_ODU#1/NED_SQLite.db')


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

#=================================================================================
day = 60 * 60 * 24
def get_sine_time(time):
    seconds = int(round(time.timestamp()))
    time_sine = np.sin(seconds * (2 * np.pi / day))

    return time_sine


def get_cosine_time(time):
    seconds = int(round(time.timestamp()))
    time_cosine = np.cos(seconds * (2 * np.pi / day))

    return time_cosine


year = (365.2425)*day
def get_sin_year(Timestamp):
    seconds_y = int(round(Timestamp.timestamp()))
    year_sine = np.sin(seconds_y * (2 * np.pi / year))

    return year_sine

def get_cos_year(Timestamp):
    seconds_y = int(round(Timestamp.timestamp()))
    year_cosine = np.cos(seconds_y * (2 * np.pi / year))

    return year_cosine

#===================================================================================
def get_odu2_data():

    sql = "SELECT * FROM odu2_data WHERE ODU_onoff='일반 운전' or ODU_onoff='정지' ORDER BY timestamp DESC LIMIT 100 "

    odu2_data = pd.read_sql_query(sql, connection)

    odu2_data = odu2_data.sort_values(by='timestamp')
    odu2_data['timestamp'] = pd.to_datetime(odu2_data['timestamp'])

    # operation mode encoding
    odu2_data['IDU0_onoff'].replace({"운전 On": 1, "운전 Off": 0}, inplace=True)
    odu2_data['IDU1_onoff'].replace({"운전 On": 1, "운전 Off": 0}, inplace=True)
    odu2_data['IDU2_onoff'].replace({"운전 On": 1, "운전 Off": 0}, inplace=True)
    odu2_data['IDU3_onoff'].replace({"운전 On": 1, "운전 Off": 0}, inplace=True)
    odu2_data['IDU4_onoff'].replace({"운전 On": 1, "운전 Off": 0}, inplace=True)

    odu2_data['ODU_onoff'].replace({"일반 운전": 1, "정지": 0}, inplace=True)


    # Capacity weighted calaculation

    odu2_data.loc[(odu2_data["IDU0_onoff"] == 0) & (odu2_data["IDU1_onoff"] == 0) &
            (odu2_data["IDU2_onoff"] == 0) & (odu2_data["IDU3_onoff"] == 0) & (odu2_data["IDU4_onoff"] == 0), ["IDU0_onoff",
            "IDU1_onoff", "IDU2_onoff", "IDU3_onoff", "IDU4_onoff"]] = 1

    odu2_data["ST"] = ((odu2_data['IDU0_onoff'] * odu2_data['IDU0_settemp'] * odu2_data['IDU0_capa'])+
                (odu2_data['IDU1_onoff'] * odu2_data['IDU1_settemp'] * odu2_data['IDU1_capa']) +
                (odu2_data['IDU2_onoff'] * odu2_data['IDU2_settemp'] * odu2_data['IDU2_capa']) +
                (odu2_data['IDU3_onoff'] * odu2_data['IDU3_settemp'] * odu2_data['IDU3_capa']) +
                (odu2_data['IDU4_onoff'] * odu2_data['IDU4_settemp'] * odu2_data['IDU4_capa'])) / \
                ((odu2_data['IDU0_onoff'] * odu2_data['IDU0_capa']) +
                (odu2_data['IDU1_onoff'] * odu2_data['IDU1_capa']) +
                (odu2_data['IDU2_onoff'] * odu2_data['IDU2_capa']) +
                (odu2_data['IDU3_onoff'] * odu2_data['IDU3_capa']) +
                (odu2_data['IDU4_onoff'] * odu2_data['IDU4_capa']))


    odu2_data["IAT"] = ((odu2_data['IDU0_onoff'] * odu2_data['IDU0_intemp'] * odu2_data['IDU0_capa'])+
                (odu2_data['IDU1_onoff'] * odu2_data['IDU1_intemp'] * odu2_data['IDU1_capa']) +
                (odu2_data['IDU2_onoff'] * odu2_data['IDU2_intemp'] * odu2_data['IDU2_capa']) +
                (odu2_data['IDU3_onoff'] * odu2_data['IDU3_intemp'] * odu2_data['IDU3_capa']) +
                (odu2_data['IDU4_onoff'] * odu2_data['IDU4_intemp'] * odu2_data['IDU4_capa'])) / \
                ((odu2_data['IDU0_onoff'] * odu2_data['IDU0_capa']) +
                (odu2_data['IDU1_onoff'] * odu2_data['IDU1_capa']) +
                (odu2_data['IDU2_onoff'] * odu2_data['IDU2_capa']) +
                (odu2_data['IDU3_onoff'] * odu2_data['IDU3_capa']) +
                (odu2_data['IDU4_onoff'] * odu2_data['IDU4_capa']))

    odu2_data = odu2_data[['timestamp', 'ODU_outtemp', 'ODU_targetpress','ODU_highpress', 'ODU_power', 'ODU_onoff', 'ST', 'IAT']]

    odu2_data = odu2_data[~((odu2_data['ODU_power'] > 100) & (odu2_data['ODU_onoff'] == 0))]

    odu2_data = odu2_data.set_index('timestamp')
    odu2_data_avg = odu2_data.groupby([pd.Grouper(freq = '30min')]).mean().dropna()

    odu2_data_avg['ODU_onoff'] = odu2_data_avg['ODU_onoff'].apply(lambda x: 1 if x >= 0.5 else 0)
    odu2_data_avg.reset_index(inplace=True)


    time_sine = []
    time_cosine = []
    year_sine = []
    year_cosine = []

    for index, row in odu2_data_avg.iterrows():
        time_sine.append(get_sine_time(row["timestamp"]))
        time_cosine.append(get_cosine_time(row["timestamp"]))
        year_sine.append(get_sin_year(row["timestamp"]))
        year_cosine.append(get_cos_year(row["timestamp"]))

    odu2_data_avg['Day_sin'] = pd.DataFrame(time_sine)
    odu2_data_avg['Day_cos'] = pd.DataFrame(time_cosine)
    odu2_data_avg['Year_sin'] = pd.DataFrame(year_sine)
    odu2_data_avg['Year_cos'] = pd.DataFrame(year_cosine)

    odu2_data_avg.rename(columns={"ODU_onoff": "Op_Mode_1", "ODU_power": "Energy", "ODU_highpress":"High_Pressure", "ODU_outtemp": "OAT"}, inplace=True)

    Op_mode_0 = []
    for index, row in odu2_data_avg.iterrows():
        Op_mode_0.append(int(not (row["Op_Mode_1"])))
    odu2_data_avg['Op_Mode_0'] = pd.DataFrame(Op_mode_0)

    return odu2_data_avg.tail(3)



#========================================================================

def get_outdoor_data():
    sql = "SELECT * FROM outdoor_data"
    outdoor_data = pd.read_sql_query(sql, connection)
    outdoor_temp = outdoor_data.iloc[1][1]
    outdoor_rh = outdoor_data.iloc[1][2]
    return outdoor_temp, outdoor_rh

def get_outdoor_data_current():
    sql = "SELECT * FROM outdoor_data"
    outdoor_data = pd.read_sql_query(sql, connection)
    outdoor_temp = outdoor_data.iloc[0][1]
    outdoor_rh = outdoor_data.iloc[0][2]
    return outdoor_temp, outdoor_rh
#==========================================================================
def normalize(raw_data):
    ST_min, ST_max = 18, 31
    IAT_min, IAT_max = 15, 35
    OAT_min, OAT_max = -20, 20
    OTW_min, OTW_max = -15, 10
    High_Pressure_min, High_Pressure_max = 5, 35
    Energy_min, Energy_max = 0, 18000

    outdoor_temp, outdoor_rh = get_outdoor_data_current()
    otw = compute_WBT(outdoor_temp, outdoor_rh)
    raw_data['OTW'] = otw
    raw_data['ST'] = (raw_data['ST'] - ST_min) / (ST_max - ST_min)
    raw_data['IAT'] = (raw_data['IAT'] - IAT_min) / (IAT_max - IAT_min)
    raw_data['OAT'] = (raw_data['OAT'] - OAT_min) / (OAT_max - OAT_min)
    raw_data['OTW'] = (raw_data['OTW'] - OTW_min) / (OTW_max - OTW_min)

    raw_data['High_Pressure'] = (raw_data['High_Pressure'] - High_Pressure_min) / (High_Pressure_max - High_Pressure_min)
    raw_data['Delta_IAT_OAT'] = raw_data['IAT'] - raw_data['OAT']
    raw_data['Delta_ST_IAT'] = raw_data['ST'] - raw_data['IAT']
    raw_data['Delta_ST_OAT'] = raw_data['ST'] - raw_data['OAT']
    raw_data['Energy'] = (raw_data['Energy'] - Energy_min) / (Energy_max - Energy_min)

    return raw_data


def inverse_normalize(data):
    ST_min, ST_max = 18, 31
    IAT_min, IAT_max = 15, 35
    OAT_min, OAT_max = -20, 20
    OTW_min, OTW_max = -15, 10
    High_Pressure_min, High_Pressure_max = 5, 35
    Energy_min, Energy_max = 0, 18000

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

#=============================================================================================

def inverse_normalization_IAT_and_ST(IAT, ST):
    IAT_min, IAT_max = 15, 35
    ST_min, ST_max = 18, 31
    iat = IAT * (IAT_max - IAT_min) + IAT_min
    st = ST * (ST_max - ST_min) + ST_min
    return iat, st

def inverse_normalization_IAT(IAT):
    IAT_min, IAT_max = 15, 35
    iat = IAT * (IAT_max - IAT_min) + IAT_min
    return iat

def compute_WBT(temp, rh):
    otw = temp * np.arctan(0.151977 * (rh + 8.313659) ** 0.5) + np.arctan(temp + rh) - np.arctan(rh - 1.676331) + 0.00391838 * rh ** (1.5) * np.arctan(0.023101 * rh) -4.686035
    return otw








