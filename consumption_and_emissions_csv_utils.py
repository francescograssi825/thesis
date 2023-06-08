import csv
import os
import random
from datetime import datetime, timedelta

import numpy as np

import watt_time_controller


class ConsumptionAndEmissionsCsvUtils:
    def __init__(self, region_zone, reference_year="2021", reference_month="01"):
        self.watt_time = watt_time_controller.WattTimeController()
        self.region_zone = region_zone
        self.libs_to_grams = 453.59
        self.mega_to_kilowatt_hours = 1000
        self.reference_years = reference_year
        self.reference_month = reference_month
        self.wattTime_all_dir_paths = "csv_dir/region_emissions"
        self.ElectricityMaps_csv = "csv_dir/italy_csv"
        self.date_format_str = '%Y-%m-%dT%H:%M:%S%z'
        self.reference_region = {

            "other_regions":
                {
                    "reference_name": "IT_2021-01_MOER.csv",
                    "reference_path": "csv_dir/region_emissions/IT_historical/IT_2021-01_MOER.csv",
                    "dir_path": "csv_dir/region_emissions/IT_historical",
                    "dir_historical": "csv_dir/region_emissions/IT_historical/IT",
                    "historical_dirs_path": "csv_dir/region_emissions",
                    "region_key": "IT"

                },

            "italy":
                {
                    "reference_name": "IT-SO.csv",
                    "reference_path": "csv_dir/italy_csv/IT-SO.csv",
                    "dir_path": "csv_dir/italy_csv"
                }
        }

    @staticmethod
    def get_workload_energy_consumption_from_csv(workload: str):
        with open(f"csv_dir/{workload}_consumption.csv", 'r') as csv_consumption:
            dict_reader = csv.DictReader(csv_consumption)
            return [{'timestamp': i['timestamp'],
                     'cpu_energy_consumed': float(i['cpu_energy_consumed']),
                     'gpu_energy_consumed': float(i['gpu_energy_consumed']),
                     'ram_energy_consumed': float(i['ram_energy_consumed']),
                     'total_consumption': float(i['total_consumption'])} for i in list(dict_reader)]

    def get_emissions_from_csv(self, file_path='', mode='list_of_dict'):
        file_path = file_path if file_path else self.reference_region[self.region_zone]["reference_path"]

        if self.region_zone == "other_regions":

            with open(file_path, 'r') as csv_emission:
                dict_reader = csv.DictReader(csv_emission)
                if mode == 'list_of_dict':
                    return list(dict_reader)
                if mode == 'dict':
                    return {row['timestamp']: float(row['MOER']) * (self.libs_to_grams / self.mega_to_kilowatt_hours)
                            for row in dict_reader}

        elif self.region_zone == "italy":
            return self.from_electricity_map_to_wattTime(file_path)

    def from_electricity_map_to_wattTime(self, file_path):
        end_datetime = "2021-01-31T23:00:00+00:00"
        n_intervals = int(60 / 5) + 1

        with open(file_path, 'r') as csv_emission:
            dict_reader = csv.DictReader(csv_emission)
            date_dict = {}
            list_row = [row for row in dict_reader]
            for row in list_row:
                lower_b = row
                upper_b = list_row[list_row.index(row) + 1]
                interpolated_list = np.linspace(float(lower_b['carbon_intensity_avg']),
                                                float(upper_b['carbon_intensity_avg']), n_intervals)
                range_date_list = [self.get_date_for_intervals(lower_b['datetime'], 5 * i) for i in range(n_intervals)]
                date_dict.update({range_date_list[i]: interpolated_list[i] for i in range(n_intervals)})
                if row['datetime'] == end_datetime:
                    return date_dict

    def get_date_for_intervals(self, string_date, delta):

        string_to_date = datetime.strptime(string_date, self.date_format_str) + timedelta(minutes=delta)
        return datetime.strftime(string_to_date, self.date_format_str).replace('+0000', '+00:00')

    def generate_mock_emissions(self, region_number):
        emissions = self.get_emissions_from_csv()
        for i in range(0, region_number):
            simulated_emissions = emissions.copy()
            for sim_em in simulated_emissions:
                sim_em['MOER'] = float(sim_em['MOER']) + random.randint(-20, 20)
            csv_name = f"csv_dir/region_emissions/region_{i}.csv"
            with open(csv_name, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, simulated_emissions[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(simulated_emissions)

    def list_all_area_wattTime(self, reference_years='2021', reference_month='01'):

        all_regions = os.listdir(self.wattTime_all_dir_paths)
        all_regions.remove('.DS_Store')

        return [{"name": f"{i.split('_')[0]}_{reference_years}-{reference_month}_MOER.csv",
                 "path": f"{self.wattTime_all_dir_paths}/{i}/{i.split('_')[0]}_{reference_years}-{reference_month}_MOER.csv"}
                for i in all_regions]

    def list_all_area_ElectricityMaps(self, reference_years='2021', reference_month='01'):

        all_regions = os.listdir(self.ElectricityMaps_csv)

        return [{"name": i,
                 "path": f"{self.ElectricityMaps_csv}/{i}"}
                for i in all_regions]

    def all_region_emission_for_a_region_zone(self, reference_year='2021', reference_month='01'):
        all_region_list = self.list_all_area_wattTime(reference_years=reference_year,
                                                      reference_month=reference_month) if self.region_zone == 'other_regions' else self.list_all_area_ElectricityMaps(
            reference_years=reference_year, reference_month=reference_month)
        region_dict = {}
        for region in all_region_list:
            region_dict.update({region['name']: self.get_emissions_from_csv(
                file_path=region['path'],
                mode='dict')})
        return region_dict


def areas_code_to_csv(self):
    all_areas = self.watt_time.get_all_area()
    keys = all_areas[0].keys()
    with open('csv_dir/areas_code.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_areas)
