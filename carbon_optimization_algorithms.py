import math
import time
from datetime import datetime, timedelta
from operator import itemgetter
from statistics import mean


from matplotlib import pyplot as plt
import numpy as np
import consumption_and_emissions_csv_utils


class TrainOptimization:
    def __init__(self, start_time, workload, regions="other_regions"):
        self.workload_energy = consumption_and_emissions_csv_utils.get_workload_energy_consumption_from_csv(workload)
        self.workload = workload
        self.csv_regions = regions
        self.emissions = consumption_and_emissions_csv_utils.get_emissions_from_csv(mode='dict', regions= self.csv_regions)
        self.start_time = start_time
        self.minimum_workload_len = 5 * len(self.workload_energy)
        self.date_format_str = '%Y-%m-%dT%H:%M:%S%z'
        self.consumption_values = [i['total_consumption'] for i in self.workload_energy]
        self.kwH_per_GB = 0.028  # Kwh/Gb
        self.dataset_dim = 0.1  # Gb
        self.kwH_for_dataset = self.kwH_per_GB * self.dataset_dim
        self.reference_region = 'CAISO_NORTH_2018-01_MOER.csv' if self.csv_regions == "other_regions" else "IT-SO.csv"

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
        return optimal_start_time, optimal_emissions

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

        return sorted_interval_by_start, carbon_emission

    @staticmethod
    def get_index_by_key(dictionary, key):
        return list(dictionary).index(key)

    def strategy_duration(self, best_intervals):
        strategy_duration = datetime.strptime(best_intervals[-1]['end_time'], self.date_format_str) - datetime.strptime(
            best_intervals[0]['start_time'], self.date_format_str)
        return strategy_duration.total_seconds() / 60

    def get_time_delta_from_dates(self, start_date, end_date):
        return (datetime.strptime(end_date, self.date_format_str) - datetime.strptime(start_date,
                                                                                      self.date_format_str)).total_seconds() // 5

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

    def compute_id_interval(self, s_index, e_index):
        start_t = self.get_key_by_index(self.emissions, s_index)
        end_t = self.get_key_by_index(self.emissions, e_index)
        return start_t + end_t

    def get_data_transfer_emission_during_run(self, best_intervals: list, region_emissions: dict):
        taken_region = [self.reference_region]
        marginal_emissions = [region_emissions[self.reference_region][best_intervals[0]['start_time']]]
        for interval in best_intervals:
            if interval['region'] not in taken_region:
                marginal_emissions.append(region_emissions[interval['region']][interval['end_time']])
                taken_region.append(interval['region'])

        return sum([(marginal_emissions[i] + marginal_emissions[i + 1]) / 2 * self.kwH_for_dataset for i in
                    range(len(marginal_emissions) - 1)])

    def get_upstream_data_transfer_emissions(self, best_intervals: list, region_emissions: dict):
        start_time_index = self.get_index_by_key(self.emissions, self.start_time)
        start_run_index = self.get_index_by_key(self.emissions, best_intervals[0]['start_time'])
        region_set = set(i['region'] for i in best_intervals)
        region_set.add(self.reference_region)
        best_emission = 0
        start_time_transfer = ''
        for i in range(start_time_index, start_run_index + 1):

            emission = self.kwH_for_dataset * mean(
                [self.get_value_by_index(region_emissions[j], i) for j in region_set])
            if best_emission == 0 or emission < best_emission:
                start_time_transfer = self.get_key_by_index(self.emissions, i)
                best_emission = emission
        return best_emission, start_time_transfer

    def follow_the_sun_optimized(self, end_time, print_result=False, run_duration=5):
        """
        Compute emissions choosing the 5 min intervals withe the lowest marginal emissions between different regions.
        The time to transfer the computation must be considered
        """

        start_exc = time.time()
        regions = consumption_and_emissions_csv_utils.list_all_area(regions=self.csv_regions)
        regions_emissions = {}
        best_intervals = []
        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len)
                         for i in range(math.ceil(len(self.workload_energy) / run_len))]
        file_path = "csv_dir/region_emissions" if self.csv_regions == "other_regions" else "csv_dir/italy_csv"
        for region in regions:
            regions_emissions.update({region: consumption_and_emissions_csv_utils.get_emissions_from_csv(
                file_path=f"{file_path}/{region}", mode='dict', regions=self.csv_regions)})

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
        transfer_on_run_emission = self.get_data_transfer_emission_during_run(best_intervals, regions_emissions)
        upstream_data_transfer_emission, upstream_data_transfer_start_time = self.get_upstream_data_transfer_emissions(
            best_intervals, regions_emissions)

        if print_result:
            print("\n-FOLLOW THE SUN OPTIMIZED-")
            print(f"INTERVALS : {best_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(best_intervals)}")
            end_exc = time.time()
            print("Execution time :", end_exc - start_exc)
        return best_intervals, total_emissions + transfer_on_run_emission, total_emissions + upstream_data_transfer_emission

    def static_start_follow_the_sun(self, print_result=False, run_duration=5):

        regions = consumption_and_emissions_csv_utils.list_all_area(regions=self.csv_regions)

        run_len = run_duration // 5
        intervals_len = [run_len if i < len(self.workload_energy) // run_len else (len(self.workload_energy) % run_len)
                         for i in range(math.ceil(len(self.workload_energy) / run_len))]

        start_time_index = self.get_index_by_key(self.emissions, self.start_time)
        selected_intervals = []

        file_path = "csv_dir/region_emissions" if self.csv_regions == "other_regions" else "csv_dir/italy_csv"
        regions_emissions = {region: consumption_and_emissions_csv_utils.get_emissions_from_csv(file_path=f"{file_path}/{region}", mode='dict', regions=self.csv_regions) for region in regions}

        emission_start_index = start_time_index
        consumption_start_index = 0
        for interval_len in intervals_len:
            last_interval = {}
            for region_name, region_list in regions_emissions.items():
                """ Loop to get the best interval with size specified in interval_len """
                interval_emission = np.asarray(list(regions_emissions[region_name].values())[emission_start_index + 1: emission_start_index + 1 + interval_len])
                interval_consumption = np.asarray(self.consumption_values[consumption_start_index:consumption_start_index + interval_len])
                interval = {
                    'start_time': self.get_key_by_index(regions_emissions[region_name], emission_start_index),
                    'end_time': self.get_key_by_index(regions_emissions[region_name], emission_start_index + interval_len),
                    'emission': np.dot(interval_emission, interval_consumption),
                    'region': region_name
                }

                if last_interval == {} or interval['emission'] < last_interval['emission']:
                    last_interval = interval

            selected_intervals.append(last_interval)
            emission_start_index = emission_start_index + interval_len
            consumption_start_index = consumption_start_index + interval_len

        total_emissions = sum(i['emission'] for i in selected_intervals)
        transfer_on_run_emission = self.get_data_transfer_emission_during_run(selected_intervals, regions_emissions)
        upstream_data_transfer_emission, upstream_data_transfer_start_time = self.get_upstream_data_transfer_emissions(
            selected_intervals, regions_emissions)

        if print_result:
            print("\n-STATIC START FTS-")
            print(f"INTERVALS : {selected_intervals}")
            print(f"EMISSIONS\t ---> \t{total_emissions} C02eq would been emitted")
            print(f"STRATEGY DURATION\t ---> \t{self.strategy_duration(selected_intervals)}")

        return selected_intervals, total_emissions + transfer_on_run_emission, total_emissions + upstream_data_transfer_emission

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

    def bar_plot(self, no_echo_mode,
                 flexible_fts_emissions_on_run,
                 flexible_fts_emissions_upstream_transfer,
                 static_start_fts_emissions_on_run,
                 static_start_fts_emissions_upstream_transfer,
                 flex_start_emission,
                 p_r_emissions,
                 title_param=''):

        fig, ax = plt.subplots()
        fx_fts_up = (no_echo_mode-flexible_fts_emissions_upstream_transfer)*(100/flexible_fts_emissions_upstream_transfer)
        fx_fts_on_run = (no_echo_mode-flexible_fts_emissions_on_run)*(100/flexible_fts_emissions_on_run)
        s_fts_emissions_on_run = (no_echo_mode-static_start_fts_emissions_on_run)*(100/static_start_fts_emissions_on_run)
        s_fts_emissions_upstream = (no_echo_mode-static_start_fts_emissions_upstream_transfer)*(100/static_start_fts_emissions_upstream_transfer)
        flx_start = (no_echo_mode-flex_start_emission)*(100/flex_start_emission)
        p_and_r = (no_echo_mode-p_r_emissions)*(100/p_r_emissions)

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


    def compute_graphics(self, run_duration):
        end_windows_set = [6, 12,18, 24]
        ending_time_list = []
        for t in end_windows_set:
            ending_time_list.append(self.get_date_for_intervals(self.start_time, (t * 60) + self.minimum_workload_len))
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
            flexible_fts_intervals, flexible_fts_emissions_on_run, flexible_fts_emissions_upstream_transfer = self.follow_the_sun_optimized(
                end_time=end_t, run_duration=run_duration)
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

            static_start_fts_intervals, static_start_fts_emissions_on_run, static_start_fts_emissions_upstream_transfer = self.static_start_follow_the_sun(
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
            self.bar_plot(no_echo_mode=no_echo_emission,
                          flexible_fts_emissions_on_run=flexible_fts_emissions_on_run,
                          flexible_fts_emissions_upstream_transfer=flexible_fts_emissions_upstream_transfer,
                          static_start_fts_emissions_on_run=static_start_fts_emissions_on_run,
                          static_start_fts_emissions_upstream_transfer=static_start_fts_emissions_upstream_transfer,
                          flex_start_emission=flex_start_emission,
                          p_r_emissions=p_r_emissions,
                          title_param=f"{self.workload} for window end time {end_t}"
                          )
            """ TIMELINE  PLOT """
            self.timeline_graph(flexible_fts_intervals=flexible_fts_intervals, static_start_fts_intervals= static_start_fts_intervals, flexible_start_start=flex_start_time,
                                pause_resume_intervals=p_r_intervals, run_len=run_duration, ending_time=end_t)

        """ LINE PLOT EMISSIONS """
        plt.plot(strategy_consumption['flexible_follow_the_sun_data_on_run']['end_time'],
               strategy_consumption['flexible_follow_the_sun_data_on_run']['emission'],
               label='f-fts-on-run', linewidth=1.5, linestyle='solid')
        plt.plot(strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['end_time'],
               strategy_consumption['flexible_follow_the_sun_upstream_data_transfer']['emission'],
               label='f-fts-upstream', linewidth=1.5, linestyle='solid')

        plt.plot(strategy_consumption['static_start_follow_the_sun_data_on_run']['end_time'],
               strategy_consumption['static_start_follow_the_sun_data_on_run']['emission'],
               label='s-fts-on-run', linewidth=1.5, linestyle='solid')
        plt.plot(strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['end_time'],
               strategy_consumption['static_start_follow_the_sun_upstream_data_transfer']['emission'],
               label='s-fts-upstream', linewidth=1.5, linestyle='solid')

        plt.plot(strategy_consumption['flexible_start']['end_time'], strategy_consumption['flexible_start']['emission'],
               label='fs', linewidth=1.5, linestyle='dotted')
        plt.plot(strategy_consumption['pause_and_resume']['end_time'],
               strategy_consumption['pause_and_resume']['emission'], label='plt&r', linewidth=1.5,
               linestyle='dashed')
        plt.plot(strategy_consumption['no_echo_mode']['end_time'], strategy_consumption['no_echo_mode']['emission'],
               label='no_echo', linewidth=1.5, linestyle='dashdot')

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
        return int(datetime.timestamp(datetime.strptime(date, self.date_format_str)))

    def to_datetime(self, date: str):
        return datetime.strptime(date, self.date_format_str)

    def timeline_graph(self, flexible_fts_intervals: list, static_start_fts_intervals: list,
                       pause_resume_intervals: list, flexible_start_start: str, run_len,
                       ending_time):

        flexible_fts_start_timestamp = self.to_timestamp(flexible_fts_intervals[0]['start_time'])
        flexible_fts_end_timestamp = self.to_timestamp(flexible_fts_intervals[-1]['end_time'])
        flexible_fts_end_datetime = self.to_datetime(flexible_fts_intervals[-1]['end_time'])

        static_start_fts_start_timestamp = self.to_timestamp(static_start_fts_intervals[0]['start_time'])
        static_start_fts_end_timestamp = self.to_timestamp(static_start_fts_intervals[-1]['end_time'])
        static_start_fts_end_datetime = self.to_datetime(static_start_fts_intervals[-1]['end_time'])

        pause_resume_end_datetime = self.to_datetime(pause_resume_intervals[-1]['end_time'])

        fs_start_timestamp = self.to_timestamp(flexible_start_start)
        fs_end_timestamp = self.to_timestamp(
            self.get_date_for_intervals(flexible_start_start, self.minimum_workload_len))
        fs_end_datetime = self.to_datetime(self.get_date_for_intervals(flexible_start_start, self.minimum_workload_len))

        end_datetime = max(flexible_fts_end_datetime, static_start_fts_end_datetime, pause_resume_end_datetime,
                           fs_end_datetime)

        start_date_index = self.get_index_by_key(self.emissions, self.start_time)
        end_date_index = self.get_index_by_key(self.emissions,
                                               datetime.strftime(end_datetime, self.date_format_str).replace('+0000',
                                                                                                             '+00:00'))
        date_list = [self.get_key_by_index(self.emissions, i) for i in range(start_date_index, end_date_index + 1, 12)]
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
        plt.title(f"Timeline for {self.workload} with fts run length of {run_len} min and window-ending time {ending_time}", fontsize=8)
        plt.tight_layout()
        plt.show()
