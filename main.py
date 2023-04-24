import time

from carbon_optimization_algorithms import TrainOptimization
import consumption_and_emissions_csv_utils

from consumption_and_emissions_csv_utils import generate_mock_emissions


def main():
    test_op = TrainOptimization(start_time='2018-01-10T00:00:00+00:00')
    end_time = test_op.minimum_workload_len + (6*60)
    test_op.launcher(end_time=test_op.get_date_for_intervals('2018-01-10T00:00:00+00:00', end_time), mode='compare', fts_run_duration=120)
    #start = time.time()



    #intervals = test_op.get_best_intervals_for_region(end_time='2018-01-02T00:00:00+00:00', run_len=4*60)
    # end = time.time()
    #
    # print("time for old:", end-start)
    #start = time.time()
    #
    # new_intervals = test_op.get_best_intervals_for_region_new(end_time='2018-01-02T00:00:00+00:00', run_len=4*60)
    # end = time.time()
    #
    #start = time.time()

    #fts = test_op.follow_the_sun(end_time='2018-01-02T00:00:00+00:00', run_duration=2*60)
    #end = time.time()
    #print("time for fts:", end - start)

    #fts_naive = test_op.follow_the_sun_naive(end_time='2018-01-02T00:00:00+00:00')

    #print(intervals)
    #launcher('2018-01-01T01:20:00+00:00', '2018-01-01T03:20:00+00:00', 'compare')
    #consumption_and_emissions_csv_utils.write_to_csv_workload_consumption()
    #launcher(start_time='2018-01-01T00:20:00+00:00', end_time='2018-01-02T23:55:00+00:00', mode='compare', data_transfer_time=30, print_result = True)
    #consumption_and_emissions_csv_utils.get_workload_energy_consumption_from_csv()
    #launcher('2018-01-01T00:20:00+00:00', '2018-01-01T22:20:00+00:00', 'compare', window_workload=10, data_transfer_time=20)
    #generate_mock_emissions(5)
    #consumption_and_emissions_csv_utils.read_all_reagions_marginal_emissions()
    #carbon_optimization_algorithms.follow_the_sun('2018-01-01T00:20:00+00:00', window_workload=10, data_transfer_time=0)
    #workload_algorithms.isolation_forrest()

    run_len_set = [30, 60, 90, 120]
    for run_set in run_len_set:
        test_op.compute_graphics(run_set)


if __name__ == "__main__":
     main()
