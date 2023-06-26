import math
import time
from datetime import datetime
from operator import itemgetter
from statistics import mean

from matplotlib import pyplot as plt
import numpy as np
from consumption_and_emissions_csv_utils import ConsumptionAndEmissionsCsvUtils


class TrainOptimization:
    def __init__(self, start_time, workload, region_zone="other_regions"):
        self.csv_utils = ConsumptionAndEmissionsCsvUtils(region_zone=region_zone)
        self.workload_energy = self.csv_utils.get_workload_energy_consumption_from_csv(workload)
        self.workload = workload
        self.csv_region_zone = region_zone
        self.start_time = start_time
        self.minimum_workload_len = 5 * len(self.workload_energy)
        self.consumption_values = [i['total_consumption'] for i in self.workload_energy]
        self.kwH_per_GB = 0.02875  # Kwh/Gb
        self.dataset_dim = 0.3198  # Gb
        self.kwH_for_dataset = self.kwH_per_GB * self.dataset_dim
        self.reference_region = self.csv_utils.reference_region[region_zone]["reference_name"]

    def flexible_start(self, end_time, start_time='', print_result=False, emissions_path=''):
        start_t = start_time if start_time else self.start_time
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emissions_path)
        start_index = self.get_index_by_key(emissions, start_t)
        end_window = self.get_index_by_key(emissions, end_time) + 1

        last_total_emissions = 0
        optimal_start_time = ''
        for start_window_index in range(start_index + 1, end_window):
            ranged_emissions = list(emissions.values())[
                               start_window_index: start_window_index + len(self.workload_energy)]
            total_emissions = np.asarray(self.consumption_values).dot(np.asarray(ranged_emissions))
            if last_total_emissions == 0 or total_emissions < last_total_emissions:
                last_total_emissions = total_emissions
                optimal_start_time = self.get_key_by_index(emissions, start_window_index)

        optimal_emissions = last_total_emissions
        optimal_end_time = self.csv_utils.get_date_for_intervals(optimal_start_time, self.minimum_workload_len)
        strategy_duration = datetime.strptime(optimal_end_time, self.csv_utils.date_format_str) - datetime.strptime(
            optimal_start_time, self.csv_utils.date_format_str)
        if print_result:
            print("\n -FLEXIBLE START-")
            print(f"OPTIMAL STARTING TIME: {optimal_start_time}")
            print(f"EMISSIONS\t ---> \t{optimal_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{strategy_duration}")

        return optimal_start_time, optimal_emissions

    def no_echo_mode(self, start_time='', print_result=False, emission_path=''):
        start_t = start_time if start_time else self.start_time
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emission_path)
        start_index = self.get_index_by_key(emissions, start_t)
        end_time_index = start_index + len(self.workload_energy)
        total_emission = sum(i for i in np.multiply(self.consumption_values,
                                                    list(emissions.values())[start_index + 1:end_time_index + 1]))
        strategy_duration = datetime.strptime(
            self.csv_utils.get_date_for_intervals(start_t, self.minimum_workload_len),
            self.csv_utils.date_format_str) - datetime.strptime(start_t,
                                                                self.csv_utils.date_format_str)

        if print_result:
            print("\n-NO ECO MODE-")
            print(f"STARTING TIME: {start_t}")
            print(f"EMISSIONS\t ---> \t{total_emission} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{strategy_duration}")

        return total_emission, strategy_duration.total_seconds() / 60

    def pause_and_resume(self, end_time, start_time='', emissions_path='', print_result=False):
        start_t = start_time if start_time else self.start_time
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emissions_path)
        if self.csv_utils.get_date_for_intervals(start_t, self.minimum_workload_len) > end_time:
            raise Exception(
                f"PAUSE AND RESUME EXCEPTION: Minimum ending time required is {self.csv_utils.get_date_for_intervals(start_t, self.minimum_workload_len)} , but {end_time} is given")

        start_index = list(emissions).index(start_t)
        end_index = list(emissions).index(end_time)

        interval_marginal_emissions = [{
            'region': self.csv_utils.reference_region[self.csv_region_zone]['reference_name'],
            'start_time': list(emissions.keys())[em_index],
            'end_time': list(emissions.keys())[em_index + 1],
            'marginal_emission': list(emissions.values())[em_index + 1]
        } for em_index in range(start_index, end_index)]

        sorted_interval_by_emissions = sorted(interval_marginal_emissions, key=itemgetter('marginal_emission'))[
                                       :len(self.workload_energy)]
        sorted_interval_by_start = sorted(sorted_interval_by_emissions, key=itemgetter('start_time'))

        emissions_list = [
            sorted_interval_by_start[emission_index]['marginal_emission'] * self.workload_energy[emission_index][
                'total_consumption'] for emission_index in range(len(sorted_interval_by_start))]
        carbon_emission = sum(i for i in emissions_list)

        if print_result:
            print("\n -PAUSE AND RESUME-")
            print(f"INTERVALS : {sorted_interval_by_start}")
            print(f"EMISSIONS\t ---> \t{carbon_emission} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(sorted_interval_by_start)}")

        return sorted_interval_by_start, carbon_emission

    @staticmethod
    def get_index_by_key(dictionary, key):
        return list(dictionary).index(key)

    def strategy_duration(self, best_intervals):
        strategy_duration = datetime.strptime(best_intervals[-1]['end_time'],
                                              self.csv_utils.date_format_str) - datetime.strptime(
            best_intervals[0]['start_time'], self.csv_utils.date_format_str)
        return strategy_duration.total_seconds() / 60

    @staticmethod
    def get_key_by_index(dictionary, index):
        return list(dictionary.keys())[index]

    @staticmethod
    def get_value_by_index(dictionary, index):
        return list(dictionary.values())[index]

    def get_data_transfer_emission_during_run(self, year, month, best_intervals: list, region_emissions: dict):
        taken_region = [self.reference_region]
        reg = f'{self.csv_utils.reference_region["other_regions"]["region_key"]}_{year}-{month}_MOER.csv' if self.csv_region_zone == "other_regions" else self.reference_region
        marginal_emissions = [region_emissions[reg][best_intervals[0]['start_time']]]
        for interval in best_intervals:
            if interval['region'] not in taken_region:
                marginal_emissions.append(region_emissions[interval['region']][interval['end_time']])
                taken_region.append(interval['region'])

        return sum([(marginal_emissions[i] + marginal_emissions[i + 1]) / 2 * self.kwH_for_dataset for i in
                    range(len(marginal_emissions) - 1)]), len(taken_region)-2

    def get_upstream_data_transfer_emissions(self, year, month, best_intervals: list, region_emissions: dict,
                                             start_time, emissions_path=''):
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emissions_path)
        start_time_index = self.get_index_by_key(emissions, start_time)
        start_run_index = self.get_index_by_key(emissions, best_intervals[0]['start_time'])
        region_set = set(i['region'] for i in best_intervals)
        reg = f'{self.csv_utils.reference_region["other_regions"]["region_key"]}_{year}-{month}_MOER.csv' if self.csv_region_zone == "other_regions" else self.reference_region
        region_set.add(reg)
        best_emission = 0
        start_time_transfer = ''
        for i in range(start_time_index, start_run_index + 1):

            emission = self.kwH_for_dataset * mean(
                [self.get_value_by_index(region_emissions[j], i) for j in region_set])
            if best_emission == 0 or emission < best_emission:
                start_time_transfer = self.get_key_by_index(emissions, i)
                best_emission = emission
        return best_emission, start_time_transfer

    def follow_the_sun_optimized(self, end_time, start_time='', emissions_path='', print_result=False, run_duration=5,
                                 year='2021', month='01'):
        """
        Compute emissions choosing the 5 min intervals withe the lowest marginal emissions between different regions.
        The time to transfer the computation must be considered
        """
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emissions_path)
        start_t = start_time if start_time else self.start_time
        regions_emissions = self.csv_utils.all_region_emission_for_a_region_zone(reference_month=month,
                                                                                 reference_year=year)
        best_intervals = []
        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len)
                         for i in range(math.ceil(len(self.workload_energy) / run_len))]

        start_window_index = self.get_index_by_key(self.get_value_by_index(regions_emissions, 0), start_t)
        end_window_index = self.get_index_by_key(emissions, end_time)

        for start_window_index in range(start_window_index, end_window_index):
            """ Loop to try every starting time """
            interval_end_index = start_window_index
            intervals = []
            consumption_start_index = 0
            for interval_len in intervals_len:
                """ Loop to compose the train by the intervals picked from the regions with lowest marginal emissions """
                last_interval = {}
                consumption_end_index = consumption_start_index + interval_len
                interval_start_index = interval_end_index
                interval_end_index = interval_start_index + interval_len
                for region_name, region_list in regions_emissions.items():
                    """ Loop to get the best interval with size specified in interval_len """
                    em_list = np.asarray(
                        list(regions_emissions[region_name].values())[interval_start_index + 1:interval_end_index + 1])
                    interval = {
                        'start_time': self.get_key_by_index(regions_emissions[region_name], interval_start_index),
                        'end_time': self.get_key_by_index(regions_emissions[region_name], interval_end_index),
                        'emission': np.asarray(em_list).dot(
                            np.asarray(self.consumption_values[consumption_start_index:consumption_end_index])),
                        'region': region_name
                    }

                    if last_interval == {} or interval['emission'] < last_interval['emission']:
                        last_interval = interval

                intervals.append(last_interval)
                consumption_start_index = consumption_end_index
            if best_intervals == [] or sum(item['emission'] for item in intervals) < sum(
                    item['emission'] for item in best_intervals):
                best_intervals = intervals.copy()

        total_emissions = sum(i['emission'] for i in best_intervals)
        transfer_on_run_emission, n_transfer = self.get_data_transfer_emission_during_run(year, month, best_intervals,
                                                                              regions_emissions)
        upstream_data_transfer_emission, upstream_data_transfer_start_time = self.get_upstream_data_transfer_emissions(
            year, month,
            best_intervals, regions_emissions, emissions_path=emissions_path, start_time=start_t)

        if print_result:
            print("\n-FOLLOW THE SUN OPTIMIZED-")
            print(f"INTERVALS : {best_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(best_intervals)}")

        return best_intervals, total_emissions + transfer_on_run_emission, total_emissions + upstream_data_transfer_emission, n_transfer

    def static_start_follow_the_sun(self, print_result=False, emissions_path='', start_time='', run_duration=5,
                                    month='01', year='2021'):
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict', file_path=emissions_path)
        start_t = start_time if start_time else self.start_time
        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len)
                         for i in range(math.ceil(len(self.workload_energy) / run_len))]

        start_time_index = self.get_index_by_key(emissions, start_t)
        selected_intervals = []

        regions_emissions = self.csv_utils.all_region_emission_for_a_region_zone(reference_year=year,
                                                                                 reference_month=month)
        emission_start_index = start_time_index
        consumption_start_index = 0
        for interval_len in intervals_len:
            last_interval = {}
            for region_name, region_list in regions_emissions.items():
                """ Loop to get the best interval with size specified in interval_len """
                interval_emission = np.asarray(list(regions_emissions[region_name].values())[
                                               emission_start_index + 1: emission_start_index + 1 + interval_len])
                interval_consumption = np.asarray(
                    self.consumption_values[consumption_start_index:consumption_start_index + interval_len])
                interval = {
                    'start_time': self.get_key_by_index(regions_emissions[region_name], emission_start_index),
                    'end_time': self.get_key_by_index(regions_emissions[region_name],
                                                      emission_start_index + interval_len),
                    'emission': np.dot(interval_emission, interval_consumption),
                    'region': region_name
                }

                if last_interval == {} or interval['emission'] < last_interval['emission']:
                    last_interval = interval

            selected_intervals.append(last_interval)
            emission_start_index = emission_start_index + interval_len
            consumption_start_index = consumption_start_index + interval_len

        total_emissions = sum(i['emission'] for i in selected_intervals)
        transfer_on_run_emission, n_transfer = self.get_data_transfer_emission_during_run(year, month, selected_intervals,
                                                                              regions_emissions)
        upstream_data_transfer_emission, upstream_data_transfer_start_time = self.get_upstream_data_transfer_emissions(
            year, month,
            selected_intervals, regions_emissions, emissions_path=emissions_path, start_time=start_t)

        if print_result:
            print("\n-STATIC START FTS-")
            print(f"INTERVALS : {selected_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(selected_intervals)}")

        return selected_intervals, total_emissions + transfer_on_run_emission, total_emissions + upstream_data_transfer_emission, n_transfer

    def launcher(self, end_time, fts_run_duration, print_result=True):
        try:

            no_echo = self.no_echo_mode()
            f_start = self.flexible_start(end_time=end_time, print_result=print_result)
            p_and_r = self.pause_and_resume(end_time=end_time, print_result=print_result)
            # f_the_sun = self.follow_the_sun(end_time=end_time, print_result=print_result,
            # run_duration=fts_run_duration)
            f_the_sun_optimized = self.follow_the_sun_optimized(end_time=end_time, print_result=print_result,
                                                                run_duration=fts_run_duration)
            return no_echo, f_start, p_and_r, f_the_sun_optimized
        except Exception as ex:
            raise ex

    @staticmethod
    def compute_percentage_decrease(initial_val, final_val):
        return ((initial_val - final_val) / initial_val) * 100

    def bar_plot(self, no_echo_mode,
                 flexible_fts_emissions_on_run,
                 flexible_fts_emissions_upstream_transfer,
                 static_start_fts_emissions_on_run,
                 static_start_fts_emissions_upstream_transfer,
                 flex_start_emission,
                 p_r_emissions,
                 title_param=''):

        fig, ax = plt.subplots()
        fx_fts_up = self.compute_percentage_decrease(no_echo_mode, flexible_fts_emissions_upstream_transfer)
        fx_fts_on_run = self.compute_percentage_decrease(no_echo_mode, flexible_fts_emissions_on_run)
        s_fts_emissions_on_run = self.compute_percentage_decrease(no_echo_mode, static_start_fts_emissions_on_run)
        s_fts_emissions_upstream = self.compute_percentage_decrease(no_echo_mode,
                                                                    static_start_fts_emissions_upstream_transfer)
        flx_start = self.compute_percentage_decrease(no_echo_mode, flex_start_emission)
        p_and_r = self.compute_percentage_decrease(no_echo_mode, p_r_emissions)

        fruits = ['flexible_fts_upstream',
                  'flexible_fts_on_run',
                  's_fts_upstream',
                  's_fts_on_run',
                  'flexible_start',
                  'pause&resume']
        counts = [fx_fts_up, fx_fts_on_run, s_fts_emissions_on_run, s_fts_emissions_upstream, flx_start, p_and_r]
        bar_labels = ['red', 'blue', 'yellow', 'orange', 'green', 'pink']
        bar_colors = ['red', 'blue', 'yellow', 'orange', 'green', 'pink']

        ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

        ax.set_ylabel('percentage reduction')
        ax.set_title(f'Emission percentage reduction for {title_param}', fontsize=7)
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()

        plt.show()

    def years_avarage_reduction_bar_plot(self, run_duration, years='2021'):
        print("Graph computing")
        ref_days = ["01T00:00:00+00:00", "05T00:00:00+00:00", "10T00:00:00+00:00", "15T00:00:00+00:00",
                    "20T00:00:00+00:00", "25T00:00:00+00:00"]
        end_time_val = [6, 12, 18, 24]
        month_list = [1, 2, 3, 4, 6, 8, 10, 11]
        for t in end_time_val:
            fts_reduction_upstream = []
            fts_reduction_on_run = []
            static_fts_reduction_upstream = []
            static_fts_reduction_on_run = []
            p_r_reduction = []
            fs_reduction = []
            n_transf_s_fts = []
            n_transf_f_fts = []

            for d in ref_days:

                flexible_fts_emissions_upstream_transfer = []
                flexible_fts_emissions_on_run = []
                static_fts_emissions_upstream = []
                static_fts_emissions_on_run = []

                p_r_emissions = []
                fs_emissions = []
                no_echo_mode = []

                for i in month_list:
                    start_t = f"{years}-{str(i).zfill(2)}-{d}"
                    finish_hour = self.csv_utils.get_date_for_intervals(start_t, (t * 60) + self.minimum_workload_len)

                    fts_set_res = self.follow_the_sun_optimized(year=years, month=str(i).zfill(2), end_time=finish_hour,
                                                                emissions_path=f"{self.csv_utils.reference_region['other_regions']['dir_historical']}_{years}-{str(i).zfill(2)}_MOER.csv",
                                                                start_time=start_t, run_duration=run_duration)
                    s_fts_res = self.static_start_follow_the_sun(year=years, month=str(i).zfill(2),
                                                                 emissions_path=f"{self.csv_utils.reference_region['other_regions']['dir_historical']}_{years}-{str(i).zfill(2)}_MOER.csv",
                                                                 start_time=start_t, run_duration=run_duration)

                    flexible_fts_emissions_upstream_transfer.append(fts_set_res[2])
                    n_transf_f_fts.append(fts_set_res[3])
                    flexible_fts_emissions_on_run.append(fts_set_res[1])
                    static_fts_emissions_upstream.append(s_fts_res[2])
                    n_transf_s_fts.append(s_fts_res[3])
                    static_fts_emissions_on_run.append(s_fts_res[1])
                    p_r_emissions.append(self.pause_and_resume(end_time=finish_hour, start_time=start_t,
                                                               emissions_path=f"{self.csv_utils.reference_region['other_regions']['dir_historical']}_{years}-{str(i).zfill(2)}_MOER.csv")[
                                             1])
                    fs_emissions.append(self.flexible_start(end_time=finish_hour, start_time=start_t,
                                                            emissions_path=f"{self.csv_utils.reference_region['other_regions']['dir_historical']}_{years}-{str(i).zfill(2)}_MOER.csv")[
                                            1])

                    no_echo_mode.append(self.no_echo_mode(start_time=start_t,
                                                          emission_path=f"{self.csv_utils.reference_region['other_regions']['dir_historical']}_{years}-{str(i).zfill(2)}_MOER.csv")[
                                            0])
                    fts_reduction_upstream.append(self.compute_percentage_decrease(no_echo_mode[-1],
                                                                                   flexible_fts_emissions_upstream_transfer[
                                                                                       -1]))
                    fts_reduction_on_run.append(
                        self.compute_percentage_decrease(no_echo_mode[-1], flexible_fts_emissions_on_run[-1]))
                    static_fts_reduction_upstream.append(
                        self.compute_percentage_decrease(no_echo_mode[-1], static_fts_emissions_upstream[-1]))
                    static_fts_reduction_on_run.append(
                        self.compute_percentage_decrease(no_echo_mode[-1], static_fts_emissions_on_run[-1]))
                    p_r_reduction.append(self.compute_percentage_decrease(no_echo_mode[-1], p_r_emissions[-1]))
                    fs_reduction.append(self.compute_percentage_decrease(no_echo_mode[-1], fs_emissions[-1]))

            fig, ax = plt.subplots()

            avg_f_fts_up = np.average(np.asarray(fts_reduction_upstream))
            avg_f_fts_on_run = np.average(np.asarray(fts_reduction_on_run))
            avg_s_fts_up = np.average(np.asarray(static_fts_reduction_upstream))
            avg_s_fts_on_run = np.average(np.asarray(static_fts_reduction_on_run))
            avg_p_r = np.average(np.asarray(p_r_reduction))
            avg_fs = np.average(np.asarray(fs_reduction))
            fruits = ['f-fts upstream',
                      'f-fts in-training',
                      's-fts upstream',
                      's-fts in-training',
                      'p&r',
                      'fs']
            counts = [avg_f_fts_up,
                      avg_f_fts_on_run,
                      avg_s_fts_up,
                      avg_s_fts_on_run,
                      avg_p_r,
                      avg_fs]
            bar_labels = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']
            bar_colors = ['red', 'blue', 'green', 'yellow', 'violet', 'orange']

            ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

            ax.set_ylabel('Percentage reduction')
            ax.set_title(f"Hour parameter for time-window:{t} Checking-time:{run_duration}", fontsize=7)
            plt.xticks(fontsize=9, rotation=45)
            plt.tight_layout()
            self.bar_plot_n_transfer(s_fts_number_fts=n_transf_s_fts, f_fts_number_fts=n_transf_f_fts, run_duration=run_duration)

            plt.show()
            print("end graph")





    def bar_plot_n_transfer(self, s_fts_number_fts, f_fts_number_fts, run_duration):
        fig, ax = plt.subplots()

        avg_f_fts_number = np.average(np.asarray(f_fts_number_fts))
        avg_s_fts_number = np.average(np.asarray(s_fts_number_fts))

        fruits = ['flexible fts', 'static fts']
        counts = [avg_f_fts_number, avg_s_fts_number]
        bar_labels = ['red', 'blue']
        bar_colors = ['red', 'blue']

        ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

        ax.set_ylabel('Percentage reduction')
        ax.set_title(f"Checking-time {run_duration}", fontsize=7)
        plt.xticks(fontsize=7)
        plt.tight_layout()

        plt.show()
        print("end graph")



    def compute_graphics(self, run_duration):
        end_windows_set = [6, 12, 18, 24]
        ending_time_list = []
        for t in end_windows_set:
            ending_time_list.append(
                self.csv_utils.get_date_for_intervals(self.start_time, (t * 60) + self.minimum_workload_len))
        strategy_consumption = {
            'flexible_start':
                {
                    'end_time': [],
                    'emission': [],

                },
            'flexible_follow_the_sun_data_on_run':
                {
                    'end_time': [],
                    'emission': [],

                },
            'flexible_follow_the_sun_upstream_data_transfer':
                {
                    'end_time': [],
                    'emission': [],

                },
            'static_start_follow_the_sun_data_on_run':
                {
                    'end_time': [],
                    'emission': [],

                },
            'static_start_follow_the_sun_upstream_data_transfer':
                {
                    'end_time': [],
                    'emission': [],

                },
            'pause_and_resume':
                {
                    'end_time': [],
                    'emission': [],

                },
            'no_echo_mode':
                {
                    'end_time': [],
                    'emission': [],

                }
        }

        print('GRAPHIC COMPUTING')

        for end_t in ending_time_list:
            flexible_fts_intervals, flexible_fts_emissions_on_run, flexible_fts_emissions_upstream_transfer, _ = self.follow_the_sun_optimized(end_time=end_t, run_duration=run_duration)
            strategy_consumption['flexible_follow_the_sun_data_on_run']['emission'].append(
                flexible_fts_emissions_on_run)
            strategy_consumption['flexible_follow_the_sun_data_on_run']['end_time'].append(end_t.split("+")[0])

            strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['emission'].append(
                flexible_fts_emissions_upstream_transfer)
            strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['end_time'].append(
                end_t.split("+")[0])

            flex_start_time, flex_start_emission = self.flexible_start(end_time=end_t)
            strategy_consumption['flexible_start']['emission'].append(flex_start_emission)
            strategy_consumption['flexible_start']['end_time'].append(end_t.split("+")[0])

            static_start_fts_intervals, static_start_fts_emissions_on_run, static_start_fts_emissions_upstream_transfer, _ = self.static_start_follow_the_sun(
                run_duration=run_duration)
            strategy_consumption['static_start_follow_the_sun_data_on_run']['emission'].append(
                static_start_fts_emissions_on_run)
            strategy_consumption['static_start_follow_the_sun_data_on_run']['end_time'].append(end_t.split("+")[0])

            strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['emission'].append(
                static_start_fts_emissions_upstream_transfer)
            strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['end_time'].append(
                end_t.split("+")[0])

            flex_start_time, flex_start_emission = self.flexible_start(end_time=end_t)
            strategy_consumption['flexible_start']['emission'].append(flex_start_emission)
            strategy_consumption['flexible_start']['end_time'].append(end_t.split("+")[0])

            p_r_intervals, p_r_emissions = self.pause_and_resume(end_time=end_t)
            strategy_consumption['pause_and_resume']['emission'].append(p_r_emissions)
            strategy_consumption['pause_and_resume']['end_time'].append(end_t.split("+")[0])

            no_echo_emission = self.no_echo_mode()[0]
            strategy_consumption['no_echo_mode']['emission'].append(no_echo_emission)
            strategy_consumption['no_echo_mode']['end_time'].append(end_t.split("+")[0])

            """ BAR PLOT REDUCTION """
            # self.bar_plot(no_echo_mode=no_echo_emission,
            #               flexible_fts_emissions_on_run=flexible_fts_emissions_on_run,
            #               flexible_fts_emissions_upstream_transfer=flexible_fts_emissions_upstream_transfer,
            #               static_start_fts_emissions_on_run=static_start_fts_emissions_on_run,
            #               static_start_fts_emissions_upstream_transfer=static_start_fts_emissions_upstream_transfer,
            #               flex_start_emission=flex_start_emission,
            #               p_r_emissions=p_r_emissions,
            #               title_param=f"{self.workload} for window end time {end_t}"
            #               )
            """ TIMELINE  PLOT """
            # self.timeline_graph(flexible_fts_intervals=flexible_fts_intervals,
            #                     static_start_fts_intervals=static_start_fts_intervals,
            #                     flexible_start_start=flex_start_time,
            #                     pause_resume_intervals=p_r_intervals, run_len=run_duration, ending_time=end_t)

            """ REGION PLOT """
            self.region_graph(fts_method=flexible_fts_intervals, run_len=run_duration, ending_time=end_t,
                              fts_version="flexible_fts")
            """ REGION PLOT """
            # self.region_graph(fts_method=static_start_fts_intervals, run_len=run_duration, ending_time=end_t,
            #                   fts_version="static_fts")

        line_w = 2.6

        """ LINE PLOT EMISSIONS """
        plt.plot(strategy_consumption['flexible_follow_the_sun_data_on_run']['end_time'],
                 strategy_consumption['flexible_follow_the_sun_data_on_run']['emission'],
                 label='f-fts-on-run', linewidth=line_w, linestyle='dashdot')
        plt.plot(strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['end_time'],
                 strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['emission'],
                 label='f-fts-upstream', linewidth=line_w, linestyle='dotted')

        plt.plot(strategy_consumption['static_start_follow_the_sun_data_on_run']['end_time'],
                 strategy_consumption['static_start_follow_the_sun_data_on_run']['emission'],
                 label='s-fts-on-run', linewidth=line_w, linestyle='dotted')
        plt.plot(strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['end_time'],
                 strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['emission'],
                 label='s-fts-upstream', linewidth=line_w, linestyle='dotted')

        plt.plot(strategy_consumption['flexible_start']['end_time'], strategy_consumption['flexible_start']['emission'],
                 label='fs', linewidth=line_w, linestyle='dotted')
        plt.plot(strategy_consumption['pause_and_resume']['end_time'],
                 strategy_consumption['pause_and_resume']['emission'], label='plt&r', linewidth=line_w,
                 linestyle='dashed')
        plt.plot(strategy_consumption['no_echo_mode']['end_time'], strategy_consumption['no_echo_mode']['emission'],
                 label='no_echo', linewidth=line_w, linestyle='dashdot')

        plt.xlabel('Window end time', fontsize=7)
        plt.ylabel('gCO2eq', fontsize=7)

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", prop={'size': 6.5})
        plt.title(f"{self.workload} emissions-FtS run duration {run_duration} min ", fontsize=9)
        plt.xticks(rotation=45, fontsize=7)
        plt.yticks(fontsize=7)
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.yticks(np.arange(ymin, ymax, 100))

        plt.tight_layout()

        plt.show()

    def to_timestamp(self, date: str):
        return int(datetime.timestamp(datetime.strptime(date, self.csv_utils.date_format_str)))

    def to_datetime(self, date: str):
        return datetime.strptime(date, self.csv_utils.date_format_str)

    def timeline_graph(self, flexible_fts_intervals: list, static_start_fts_intervals: list,
                       pause_resume_intervals: list, flexible_start_start: str, run_len,
                       ending_time):
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict')
        flexible_fts_start_timestamp = self.to_timestamp(flexible_fts_intervals[0]['start_time'])
        flexible_fts_end_timestamp = self.to_timestamp(flexible_fts_intervals[-1]['end_time'])
        flexible_fts_end_datetime = self.to_datetime(flexible_fts_intervals[-1]['end_time'])

        static_start_fts_start_timestamp = self.to_timestamp(static_start_fts_intervals[0]['start_time'])
        static_start_fts_end_timestamp = self.to_timestamp(static_start_fts_intervals[-1]['end_time'])
        static_start_fts_end_datetime = self.to_datetime(static_start_fts_intervals[-1]['end_time'])

        pause_resume_end_datetime = self.to_datetime(pause_resume_intervals[-1]['end_time'])

        fs_start_timestamp = self.to_timestamp(flexible_start_start)
        fs_end_timestamp = self.to_timestamp(
            self.csv_utils.get_date_for_intervals(flexible_start_start, self.minimum_workload_len))
        fs_end_datetime = self.to_datetime(
            self.csv_utils.get_date_for_intervals(flexible_start_start, self.minimum_workload_len))

        end_datetime = max(flexible_fts_end_datetime, static_start_fts_end_datetime, pause_resume_end_datetime,
                           fs_end_datetime)

        start_date_index = self.get_index_by_key(emissions, self.start_time)
        end_date_index = self.get_index_by_key(emissions,
                                               datetime.strftime(end_datetime, self.csv_utils.date_format_str).replace(
                                                   '+0000',
                                                   '+00:00'))
        date_list = [self.get_key_by_index(emissions, i) for i in range(start_date_index, end_date_index + 1, 12)]
        timestamp_list = [self.to_timestamp(i) for i in date_list]

        flexible_fts_list = [(flexible_fts_start_timestamp, flexible_fts_end_timestamp - flexible_fts_start_timestamp)]
        static_start_fts_list = [
            (static_start_fts_start_timestamp, static_start_fts_end_timestamp - static_start_fts_start_timestamp)]
        fs_list = [(fs_start_timestamp, fs_end_timestamp - fs_start_timestamp)]
        p_r_list = [(self.to_timestamp(i['start_time']), 300) for i in
                    pause_resume_intervals]  # 300 because is 5 min in sec

        fig, ax = plt.subplots()
        ax.broken_barh(flexible_fts_list, (10, 9), facecolors='tab:blue')
        ax.broken_barh(static_start_fts_list, (20, 9), facecolors='tab:green')
        ax.broken_barh(p_r_list, (30, 9), facecolors='tab:orange')
        ax.broken_barh(fs_list, (40, 9), facecolors='tab:red')
        ax.set_ylim(5, 50)

        plt.xticks(ticks=timestamp_list, labels=date_list, fontsize=7, rotation=90)
        ax.set_xlabel('time')
        ax.set_yticks([15, 25, 35, 45],
                      labels=['flexible_fts', 'static_start_fts', 'plt&r', 'fs'])  # Modify y-axis tick labels
        ax.grid(True)  # Make grid lines visible
        plt.title(
            f"{self.workload}, fts-run-duration {run_len} min, window-ending time {ending_time}",
            fontsize=7)
        plt.tight_layout()
        plt.show()

    def region_graph(self, fts_method, run_len, ending_time, fts_version):
        emissions = self.csv_utils.get_emissions_from_csv(mode='dict')
        fts_method_end_datetime = self.to_datetime(fts_method[-1]['end_time'])

        end_datetime = fts_method_end_datetime

        start_date_index = self.get_index_by_key(emissions, self.start_time)
        end_date_index = self.get_index_by_key(emissions,
                                               datetime.strftime(end_datetime, self.csv_utils.date_format_str).replace(
                                                   '+0000',
                                                   '+00:00'))
        date_list = [self.get_key_by_index(emissions, i) for i in range(start_date_index, end_date_index + 1, 12)]
        timestamp_list = [self.to_timestamp(i) for i in date_list]

        fts_method_list_set = self.get_grouped_list_of_set_by_region(fts_method, run_len)

        fig, ax = plt.subplots()
        count = 1

        for key, value in fts_method_list_set.items():
            ax.broken_barh(value, (10 * count, 10), facecolors='tab:green')
            count = count + 1

        ax.set_ylim(5, (len(fts_method_list_set) * 10) + 15)

        plt.xticks(ticks=timestamp_list, labels=date_list, fontsize=7, rotation=90)
        ax.set_xlabel('time')

        ax.set_yticks([15 + (i * 10) for i in range(0, len(fts_method_list_set))],
                      labels=[i.replace('.csv', '') for i in fts_method_list_set])  # Modify y-axis tick labels
        ax.grid(True)  # Make grid lines visible
        plt.title(
            f"{self.workload}, {fts_version}-run-duration {run_len} min, window-ending time {ending_time}",
            fontsize=7)
        plt.tight_layout()
        plt.show()

    def get_grouped_list_of_set_by_region(self, list_of_dict: list, run_len):
        result_dict = {}
        run_len_sec = run_len * 60
        for el in list_of_dict:
            region = el['region']
            interval_list = result_dict.get(region, [])
            interval_list.append((self.to_timestamp(el['start_time']), run_len_sec))
            result_dict.update({region: interval_list})
        return result_dict
