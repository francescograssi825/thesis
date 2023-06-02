import watt_time_controller
from carbon_optimization_algorithms import TrainOptimization


def main():
    test_op = TrainOptimization(start_time='2021-01-20T00:00:00+00:00', workload='isolation_forrest', region_zone="other_regions")
    #test_op = TrainOptimization(start_time='2021-01-01T00:00:00+00:00', workload='isolation_forrest', region_zone="italy")

    run_len_set = [120]
    for run_set in run_len_set:
        test_op.compute_graphics(run_set)

    # wat=watt_time_controller.WattTimeController()
    # reg=['IT', 'UK', 'SE', 'FR', 'IE', 'ES', 'DE']
    # for r in reg:
    #     wat.get_historical_grid_emissions(grid_area_code=r)


if __name__ == "__main__":
    main()
