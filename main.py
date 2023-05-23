from carbon_optimization_algorithms import TrainOptimization


def main():
    test_op = TrainOptimization(start_time='2018-01-01T00:00:00+00:00', workload='isolation_forrest')

    run_len_set = [120]
    for run_set in run_len_set:
        test_op.compute_graphics(run_set)


if __name__ == "__main__":
    main()
