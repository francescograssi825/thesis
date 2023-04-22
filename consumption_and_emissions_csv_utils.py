import csv
import os
import random

from codecarbon import EmissionsTracker
import watt_time_controller

watt_time = watt_time_controller.WattTimeController()

tracker = EmissionsTracker(measure_power_secs=5 * 60, tracking_mode='process')


def write_to_csv_workload_consumption():
    if os.path.exists('csv_dir/Custom_Consumption.csv'):
        os.remove('csv_dir/Custom_Consumption.csv')


    print('-Start tracking energy consumption-')
    tracker.start()
    os.system("/Users/alessioerrico/Desktop/Tesi/Workload/main.py")
    tracker.stop()


def get_workload_energy_consumption_from_csv():
    with open("csv_dir/Custom_Consumption.csv", 'r') as csv_consumption:
        dict_reader = csv.DictReader(csv_consumption)
        return [{'timestamp': i['timestamp'],
                 'cpu_energy_consumed': float(i['cpu_energy_consumed']),
                 'gpu_energy_consumed': float(i['gpu_energy_consumed']),
                 'ram_energy_consumed': float(i['ram_energy_consumed']),
                 'total_consumption': float(i['total_consumption'])} for i in list(dict_reader)]


def get_emissions_from_csv(file_path='csv_dir/region_emissions/CAISO_NORTH_2018-01_MOER.csv', mode='list_of_dict'):
    with open(file_path, 'r') as csv_emission:
        dict_reader = csv.DictReader(csv_emission)
        if mode == 'list_of_dict':
            return list(dict_reader)
        if mode == 'dict':
            dictionary = {}
            for row in dict_reader:
                dictionary.update({row['timestamp']: float(row['MOER'])})
            return dictionary


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


def list_all_area():
    regions = os.listdir("csv_dir/region_emissions")
    regions.remove('.DS_Store')
    return regions


def read_all_regions_marginal_emissions():
    all_csv = list_all_area()
    dict_all_csv = {}
    for csv_file in all_csv:
        region_dict = get_emissions_from_csv(file_path=f"csv_dir/region_emissions/{csv_file}", mode="dict")
        dict_all_csv.update({csv_file: region_dict})
    return dict_all_csv


def areas_code_to_csv():
    all_areas = watt_time.get_all_area()
    keys = all_areas[0].keys()
    with open('csv_dir/areas_code.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_areas)
