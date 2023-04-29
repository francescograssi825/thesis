import math
import time
from datetime import datetime, timedelta
from operator import itemgetter
import matplotlib.pyplot as p
import numpy as np

import consumption_and_emissions_csv_utils


class TrainOptimization:
    def __init__(self, start_time):
        self.workload_energy = consumption_and_emissions_csv_utils.get_workload_energy_consumption_from_csv()
        self.emissions = consumption_and_emissions_csv_utils.get_emissions_from_csv(mode='dict')
        self.start_time = start_time
        self.minimum_workload_len = 5 * len(self.workload_energy)
        self.date_format_str = '%Y-%m-%dT%H:%M:%S%z'
        self.consumption_values = [i['total_consumption'] for i in self.workload_energy]

    def flexible_start(self, end_time, print_result=False):
        start_exc = time.time()
        start_index = self.get_index_by_key(self.emissions, self.start_time)
        end_window = self.get_index_by_key(self.emissions, end_time) + 1

        last_total_emissions = 0
        optimal_start_time = ''
        for start_window_index in range(start_index + 1, end_window):
            ranged_emissions = list(self.emissions.values())[
                               start_window_index: start_window_index + len(self.workload_energy)]
            total_emissions = np.asarray(self.consumption_values).dot(np.asarray(ranged_emissions))
            if last_total_emissions == 0 or total_emissions < last_total_emissions:
                last_total_emissions = total_emissions
                optimal_start_time = self.get_key_by_index(self.emissions, start_window_index)

        optimal_emissions = last_total_emissions
        optimal_end_time = self.get_date_for_intervals(optimal_start_time, self.minimum_workload_len)
        strategy_duration = datetime.strptime(optimal_end_time, self.date_format_str) - datetime.strptime(
            optimal_start_time, self.date_format_str)
        if print_result:
            print("\n -FLEXIBLE START-")
            print(f"OPTIMAL STARTING TIME: {optimal_start_time}")
            print(f"EMISSIONS\t ---> \t{optimal_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{strategy_duration}")
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)
        return optimal_start_time, optimal_emissions, strategy_duration.total_seconds() / 60

    def no_echo_mode(self, print_result=False):
        start_exc = time.time()
        start_index = self.get_index_by_key(self.emissions, self.start_time)
        end_time_index = start_index + len(self.workload_energy)
        total_emission = sum(i for i in np.multiply(self.consumption_values,
                                                    list(self.emissions.values())[start_index + 1:end_time_index + 1]))
        strategy_duration = datetime.strptime(self.get_date_for_intervals(self.start_time, self.minimum_workload_len),
                                              self.date_format_str) - datetime.strptime(self.start_time,
                                                                                        self.date_format_str)

        if print_result:
            print("\n-NO ECO MODE-")
            print(f"STARTING TIME: {self.start_time}")
            print(f"EMISSIONS\t ---> \t{total_emission} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{strategy_duration}")
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)

        return total_emission, strategy_duration.total_seconds() / 60

    def pause_and_resume(self, end_time, region='csv_dir/region_emissions/CAISO_NORTH_2018-01_MOER.csv',
                         print_result=False):
        start_exc = time.time()
        if self.get_date_for_intervals(self.start_time, self.minimum_workload_len) > end_time:
            raise Exception(
                f"PAUSE AND RESUME EXCEPTION: Minimum ending time required is {self.get_date_for_intervals(self.start_time, self.minimum_workload_len)} , but {end_time} is given")

        start_index = list(self.emissions).index(self.start_time)
        end_index = list(self.emissions).index(end_time)

        interval_marginal_emissions = [{
            'region': region.split("/")[2],
            'start_time': list(self.emissions.keys())[em_index],
            'end_time': list(self.emissions.keys())[em_index + 1],
            'marginal_emission': list(self.emissions.values())[em_index + 1]
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
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)

        return sorted_interval_by_start, carbon_emission, self.strategy_duration(sorted_interval_by_start)

    @staticmethod
    def get_index_by_key(dictionary, key):
        return list(dictionary).index(key)

    def strategy_duration(self, best_intervals):
        strategy_duration = datetime.strptime(best_intervals[-1]['end_time'], self.date_format_str) - datetime.strptime(
            best_intervals[0]['start_time'], self.date_format_str)
        return strategy_duration.total_seconds() / 60

    def get_date_for_intervals(self, string_date, delta):
        string_to_date = datetime.strptime(string_date, self.date_format_str) + timedelta(
            minutes=delta)
        return datetime.strftime(string_to_date, self.date_format_str).replace('+0000', '+00:00')

    @staticmethod
    def get_key_by_index(dictionary, index):
        return list(dictionary.keys())[index]

    @staticmethod
    def get_value_by_index(dictionary, index):
        return list(dictionary.values())[index]

    def group_intervals(self, start_index, end_index, region_emissions, region):
        em_list = (list(region_emissions.values())[start_index + 1:end_index + 1])

        return {
            'start_time': self.get_key_by_index(region_emissions, start_index),
            'end_time': self.get_key_by_index(region_emissions, end_index),
            'marginal_emission': sum(em_list),
            'region': region
        }

    # TODO PARAMETERS : DATASET SIZE, INTERVAL SIZE
    def follow_the_sun(self, end_time, print_result=False, run_duration=5):
        """
        Compute emissions choosing the 5 min intervals withe the lowest marginal emissions between different regions.
        The time to transfer the computation must be considered
        """
        start_exc = time.time()
        regions = consumption_and_emissions_csv_utils.list_all_area()
        regions_emissions = {}

        best_intervals = []

        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len)
                         for i in range(math.ceil(len(self.workload_energy) / run_len))]

        for region in regions:
            regions_emissions.update({region: consumption_and_emissions_csv_utils.get_emissions_from_csv(
                file_path=f"csv_dir/region_emissions/{region}", mode='dict')})

        start_window_index = self.get_index_by_key(self.get_value_by_index(regions_emissions, 0), self.start_time)
        end_window_index = self.get_index_by_key(self.get_value_by_index(regions_emissions, 0), end_time)
        last_start_window_index = end_window_index - len(self.workload_energy)

        for start_window_index in range(start_window_index, last_start_window_index + 1):
            interval_end_index = start_window_index
            intervals = []

            for interval_len in intervals_len:
                last_interval = {}
                interval_start_index = interval_end_index
                interval_end_index = interval_start_index + interval_len

                for region_name, region_list in regions_emissions.items():
                    interval = self.group_intervals(interval_start_index, interval_end_index, region_list, region_name)

                    if last_interval == {} or interval['marginal_emission'] < last_interval['marginal_emission']:
                        last_interval = interval

                intervals.append(last_interval)
            if best_intervals == [] or sum(item['marginal_emission'] for item in intervals) < sum(
                    item['marginal_emission'] for item in best_intervals):
                best_intervals = intervals.copy()

        total_emissions = 0
        energy_index = 0
        for big_interval in best_intervals:
            current_region = big_interval['region']
            s_index = self.get_index_by_key(regions_emissions[current_region], big_interval['start_time'])
            e_index = self.get_index_by_key(regions_emissions[current_region], big_interval['end_time'])

            for int_index in range(s_index + 1, e_index + 1):
                total_emissions = total_emissions + (
                        self.get_value_by_index(regions_emissions[current_region], int_index) *
                        self.workload_energy[energy_index]['total_consumption'])
                energy_index += 1

        if print_result:
            print("\n-FOLLOW THE SUN-")
            print(f"INTERVALS : {best_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)
        return best_intervals, total_emissions

    def compute_id_interval(self, s_index, e_index):
        start_t = self.get_key_by_index(self.emissions, s_index)
        end_t = self.get_key_by_index(self.emissions, e_index)
        return start_t + end_t

        # TODO PARAMETERS : DATASET SIZE, INTERVAL SIZE

    def follow_the_sun_optimized(self, end_time, print_result=False, run_duration=5):
        """
        Compute emissions choosing the 5 min intervals withe the lowest marginal emissions between different regions.
        The time to transfer the computation must be considered
        """

        start_exc = time.time()
        regions = consumption_and_emissions_csv_utils.list_all_area()
        regions_emissions = {}
        best_intervals = []
        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len) for i in range(math.ceil(len(self.workload_energy) / run_len))]

        for region in regions:
            regions_emissions.update({region: consumption_and_emissions_csv_utils.get_emissions_from_csv(file_path=f"csv_dir/region_emissions/{region}", mode='dict')})

        start_window_index = self.get_index_by_key(self.get_value_by_index(regions_emissions, 0), self.start_time)
        end_window_index = self.get_index_by_key(self.emissions, end_time)

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
                        'emission': np.asarray(em_list).dot(np.asarray(self.consumption_values[consumption_start_index:consumption_end_index])),
                        'region': region_name
                    }

                    if last_interval == {} or interval['emission'] < last_interval['emission']:
                        last_interval = interval

                intervals.append(last_interval)
                consumption_start_index = consumption_end_index
            if best_intervals == [] or sum(item['emission'] for item in intervals) < sum(item['emission'] for item in best_intervals):
                best_intervals = intervals.copy()

        total_emissions = sum(i['emission'] for i in best_intervals)

        if print_result:
            print("\n-FOLLOW THE SUN OPTIMIZED-")
            print(f"INTERVALS : {best_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(best_intervals)}")
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)
        return best_intervals, total_emissions, self.strategy_duration(best_intervals)

    def launcher(self, end_time, fts_run_duration, print_result=True, mode='no_eco'):
        try:
            if mode == 'no_eco':
                return self.no_echo_mode()
            elif mode == 'pause_and_resume':
                return self.pause_and_resume(end_time=end_time, print_result=print_result)
            elif mode == 'flexibleStart':
                return self.flexible_start(end_time=end_time, print_result=print_result)
            elif mode == 'follow_the_sun':
                return self.follow_the_sun(end_time=end_time, print_result=print_result, run_duration=fts_run_duration)
            elif mode == 'compare':
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

    def compute_graphics(self, run_duration):
        end_windows_set = [6, 12, 18, 24]
        ending_time_list = []
        for t in end_windows_set:
            ending_time_list.append(self.get_date_for_intervals(self.start_time, (t * 60) + self.minimum_workload_len))
        strategy_consumption = {
            'flexible_start':
                {
                    'end_time': [],
                    'emission': [],
                    'duration': []
                },
            'follow_the_sun':
                {
                    'end_time': [],
                    'emission': [],
                    'duration': []

                },
            'pause_and_resume':
                {
                    'end_time': [],
                    'emission': [],
                    'duration': []
                },
            'no_echo_mode':
                {
                    'end_time': [],
                    'emission': [],
                    'duration': []
                }
        }

        print('GRAPHIC COMPUTING')

        for end_t in ending_time_list:
            strategy_consumption['follow_the_sun']['emission'].append(
                self.follow_the_sun_optimized(end_time=end_t, run_duration=run_duration)[1])
            strategy_consumption['follow_the_sun']['end_time'].append(end_t.split("+")[0])
            strategy_consumption['follow_the_sun']['duration'].append(
                self.follow_the_sun_optimized(end_time=end_t, run_duration=run_duration)[2])

            strategy_consumption['flexible_start']['emission'].append(self.flexible_start(end_time=end_t)[1])
            strategy_consumption['flexible_start']['end_time'].append(end_t.split("+")[0])
            strategy_consumption['flexible_start']['duration'].append(self.flexible_start(end_time=end_t)[2])

            strategy_consumption['pause_and_resume']['emission'].append(self.pause_and_resume(end_time=end_t)[1])
            strategy_consumption['pause_and_resume']['end_time'].append(end_t.split("+")[0])
            strategy_consumption['pause_and_resume']['duration'].append(self.pause_and_resume(end_time=end_t)[2])

            strategy_consumption['no_echo_mode']['emission'].append(self.no_echo_mode()[0])
            strategy_consumption['no_echo_mode']['end_time'].append(end_t.split("+")[0])
            strategy_consumption['no_echo_mode']['duration'].append(self.no_echo_mode()[1])

        """ LINE PLOT EMISSIONS """
        p.plot(strategy_consumption['follow_the_sun']['end_time'], strategy_consumption['follow_the_sun']['emission'],
               label='follow the sun', linewidth=3, linestyle='solid')
        p.plot(strategy_consumption['flexible_start']['end_time'], strategy_consumption['flexible_start']['emission'],
               label='flexible start', linewidth=3, linestyle='dotted')
        p.plot(strategy_consumption['pause_and_resume']['end_time'],
               strategy_consumption['pause_and_resume']['emission'], label='pause_and_resume', linewidth=3,
               linestyle='dashed')
        p.plot(strategy_consumption['no_echo_mode']['end_time'], strategy_consumption['no_echo_mode']['emission'],
               label='no_echo', linewidth=3, linestyle='dashdot')

        p.xlabel('Window end time', fontsize=9)
        p.ylabel('Emissions', fontsize=9)
        p.legend()
        p.title(f"Emissions for a time window with FtS run duration {run_duration} min ")
        p.xticks(rotation=45, fontsize=9)
        p.tight_layout()
        p.show()

        """ LINE  PLOT  DURATION """
        p.plot(strategy_consumption['follow_the_sun']['end_time'], strategy_consumption['follow_the_sun']['duration'],
               label='follow the sun', linewidth=3, linestyle='solid')
        p.plot(strategy_consumption['flexible_start']['end_time'], strategy_consumption['flexible_start']['duration'],
               label='flexible start', linewidth=3, linestyle='dotted')
        p.plot(strategy_consumption['pause_and_resume']['end_time'],
               strategy_consumption['pause_and_resume']['duration'], label='pause_and_resume', linewidth=2,
               linestyle='dashed')
        p.plot(strategy_consumption['no_echo_mode']['end_time'], strategy_consumption['no_echo_mode']['duration'],
               label='no_echo', linewidth=1, linestyle='dashdot')

        p.xlabel('Window end time', fontsize=9)
        p.ylabel('Duration in minutes', fontsize=9)
        p.legend()
        p.title("Train duration for different time-window ")
        p.xticks(rotation=45, fontsize=9)
        p.tight_layout()
        p.show()
