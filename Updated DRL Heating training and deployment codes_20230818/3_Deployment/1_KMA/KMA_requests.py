import requests    # to make HTTP request
from bs4 import BeautifulSoup # to pull data out of HTML and XML files
from datetime import datetime # to get date and time  
import pandas as pd # to load data into DataFrame
import numpy as np # # to deal with arrays and numerical values
import sqlite3  # to read, query and execute database statements
import time    # to deal with time for a particular day

# Connect to the Db
connection = sqlite3.connect(r"C:\Users\user\Desktop\NED_SQLite.db")

# Getting the current time
now = datetime.now()
year = str(now.year)
month = str(now.month) if len(str(now.month))==2 else '0'+str(now.month)
day = str(now.day) if len(str(now.day))==2 else '0'+str(now.day)
hour = str(now.hour) if len(str(now.hour))==2 else '0'+str(now.hour)
minute = '00'

base_date = year + month + day
base_time = hour + minute

# Making requests
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'
params ={'serviceKey' : 'VIK8O3wBYmiq4uQXwxCBy6FqX8DLD8/lafo2yq4MNmwObXVhdGkiiqftAMVpFsjMZ+9u8jy5eV+QIYyu64kj7g==',
     'pageNo' : '1', 'numOfRows' : '1000', 'dataType' : 'XML',
     'base_date' : base_date, 'base_time' : base_time,
     'nx' : '61', 'ny' : '125' }

# Handling the response
valid = False
count_error = 0
count_trial = 0
while not valid:
    try:
        r = requests.get(url, params=params)
        soup = BeautifulSoup(r.content, 'xml')

        # Check response
        temp_array = soup.find_all(lambda tag: tag.name == "item" and "T1H" in tag.text)
        rh_array = soup.find_all(lambda tag: tag.name == "item" and "REH" in tag.text)

        # Dictionary to hold filtered response
        results_dict = {
            'timestamp': [],
            'OAT': [],
            'ORH': []
        }
        if len(temp_array) != 0 and len(rh_array) != 0:
            for item in temp_array:
                time_str = item.fcstTime.text
                time_min_sec = [time_str[i:i + 2] for i in range(0, len(time_str), 2)]
                timestamp = year + '-' + month + '-' + day + ' ' + time_min_sec[0] + ':' + time_min_sec[1] + ':00'

                results_dict['timestamp'].append(timestamp)
                results_dict['OAT'].append(int(item.fcstValue.text))
            for item in rh_array:
                results_dict['ORH'].append(int(item.fcstValue.text))

            results_df = pd.DataFrame.from_dict(results_dict)
            
        # store the DataFrame into SQLite DB
            results_df.to_sql(name='outdoor_data', con=connection, if_exists='replace', index=False)
            connection.close()
        # error checking 
            valid = True
        else:
            print("Requesting again")
            count_trial += 1
            if count_trial <= 40:
                time.sleep(30)
                print('requesting again')
                continue
            else:
                break
    except:

        time.sleep(60)
        count_error += 1

        if count_error <= 20:
            print('trying again')
            continue
        else:
            break
