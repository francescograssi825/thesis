import csv
import os
import random
from datetime import datetime, timedelta

import watt_time_controller

watt_time = watt_time_controller.WattTimeController()
libs_to_grams = 453.59
mega_to_kilowatt_hours = 1000


def get_workload_energy_consumption_from_csv(workload: str):
    with open(f"csv_dir/{workload}_consumption.csv", 'r') as csv_consumption:
        dict_reader = csv.DictReader(csv_consumption)
        return [{'timestamp': i['timestamp'],
                 'cpu_energy_consumed': float(i['cpu_energy_consumed']),
                 'gpu_energy_consumed': float(i['gpu_energy_consumed']),
                 'ram_energy_consumed': float(i['ram_energy_consumed']),
                 'total_consumption': float(i['total_consumption'])} for i in list(dict_reader)]


def get_emissions_from_csv(file_path='', mode='list_of_dict', regions="other_regions"):
    if regions == "other_regions":
        file_path = file_path if file_path else 'csv_dir/region_emissions/CAISO_NORTH_2018-01_MOER.csv'
        with open(file_path, 'r') as csv_emission:
            dict_reader = csv.DictReader(csv_emission)
            if mode == 'list_of_dict':
                return list(dict_reader)
            if mode == 'dict':
                return {row['timestamp']: float(row['MOER']) * (libs_to_grams / mega_to_kilowatt_hours) for row in dict_reader}
    elif regions == "italy":
        file_path = file_path if file_path else 'csv_dir/italy_csv/IT-SO.csv'
        return from_electricity_map_to_wattTime(file_path)


def from_electricity_map_to_wattTime(file_path):
    end_datetime = "2021-01-31T23:00:00+00:00"

    with open(file_path, 'r') as csv_emission:
        dict_reader = csv.DictReader(csv_emission)
        date_dict = {}
        list_row = [row for row in dict_reader]
        for row in list_row:
            date_dict.update({row['datetime']: float(row['carbon_intensity_avg'])})
            last_datetime = row['datetime']
            for minutes in range(12-1): # 60 min divided by 5 min = 12
                current_val = float(list_row[list_row.index(row)]['carbon_intensity_avg'])
                next_val = float(list_row[list_row.index(row)+1]['carbon_intensity_avg'])
                emission_val = (current_val + next_val)/2
                last_datetime = get_date_for_intervals(last_datetime, 5)
                date_dict.update({last_datetime: emission_val})

            if row['datetime'] == end_datetime:
                return date_dict


def get_date_for_intervals(string_date, delta):
    date_format_str = '%Y-%m-%dT%H:%M:%S%z'
    string_to_date = datetime.strptime(string_date, date_format_str) + timedelta(
        minutes=delta)
    return datetime.strftime(string_to_date, date_format_str).replace('+0000', '+00:00')


def generate_mock_emissions(region_number):
    emissions = get_emissions_from_csv()
    for i in range(0, region_number):
        simulated_emissions = emissions.copy()
        for sim_em in simulated_emissions:
            sim_em['MOER'] = float(sim_em['MOER']) + random.randint(-20, 20)
        csv_name = f"csv_dir/region_emissions/region_{i}.csv"
        with open(csv_name, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, simulated_emissions[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(simulated_emissions)


def list_all_area(regions):

    all_regions = os.listdir("csv_dir/region_emissions") if regions == "other_regions" else os.listdir("csv_dir/italy_csv")
    if regions == "other_regions":
        all_regions.remove('.DS_Store')
    return all_regions


def read_all_regions_marginal_emissions(regions):
    all_csv = list_all_area(regions=regions)
    dict_all_csv = {}
    for csv_file in all_csv:
        file_path = f"csv_dir/region_emissions/{csv_file}" if regions == "other_regions" else f"csv_dir/italy_csv/{csv_file}"
        region_dict = get_emissions_from_csv(file_path=file_path, mode="dict")
        dict_all_csv.update({csv_file: region_dict})
    return dict_all_csv


def areas_code_to_csv():
    all_areas = watt_time.get_all_area()
    keys = all_areas[0].keys()
    with open('csv_dir/areas_code.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_areas)
