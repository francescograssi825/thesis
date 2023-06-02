import json
from os import path
import requests
from requests.auth import HTTPBasicAuth


class WattTimeController:
    def __init__(self):
        self.username = 'roberto.vergallo'
        self.password = 'H0tUqytKdZ_J;M5'
        self.email = 'alessio.errico1@studenti.unisalento.com'
        self.token = self.log_to_watt_time()

    def watt_time_registration(self):
        register_url = 'https://api2.watttime.org/v2/register'
        params = {'username': self.username,
                  'password': self.password,
                  'email': self.email,
                  'org': 'launcher energy conumption'}
        rsp = requests.post(register_url, json=params)
        print(rsp.text)

    def log_to_watt_time(self):
        login_url = 'https://api2.watttime.org/v2/login'
        rsp = requests.get(login_url, auth=HTTPBasicAuth(self.username, self.password))
        return rsp.json()['token']

    def get_all_area(self):
        list_url = 'https://api2.watttime.org/v2/ba-access'
        headers = {'Authorization': 'Bearer {}'.format(self.token)}
        params = {'all': 'true'}
        rsp = requests.get(list_url, headers=headers, params=params)
        return json.loads(rsp.text)

    def get_historical_grid_emissions(self, grid_area_code):
        historical_url = 'https://api2.watttime.org/v2/historical'
        headers = {'Authorization': 'Bearer {}'.format(self.token)}
        ba = grid_area_code
        params = {'ba': ba}
        rsp = requests.get(historical_url, headers=headers, params=params)
        cur_dir = path.dirname(path.realpath('__file__'))
        file_path = path.join(cur_dir, '{}_historical.zip'.format(ba))
        with open(file_path, 'wb') as fp:
            fp.write(rsp.content)

        print('Wrote historical data for {} to {}'.format(ba, file_path))
