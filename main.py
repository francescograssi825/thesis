import numpy as np
from matplotlib import pyplot as plt

import watt_time_controller
from carbon_optimization_algorithms import TrainOptimization
from consumption_and_emissions_csv_utils import ConsumptionAndEmissionsCsvUtils


def puntual_comparison(workload):
    test_op = TrainOptimization(start_time='2021-01-15T00:00:00+00:00', workload=workload, region_zone="other_regions")
    run_len_set = [15, 30, 60, 120]
    for run_set in run_len_set:
        test_op.compute_graphics(run_set)


def avarage_reduction(workload):
    run_len = [15, 30, 60, 120]
    test_op = TrainOptimization(start_time='2021-01-15T00:00:00+00:00', workload=workload,
                                region_zone="other_regions")
    for r in run_len:
        test_op.years_avarage_reduction_bar_plot(run_duration=r, years="2021")


def avarage_emissions_no_echo():
    print("Graph computing")
    csv_obj = ConsumptionAndEmissionsCsvUtils("other_region")
    algo = ['isolation_forest', 'svm', 'hf_sca', 'autoencoder']
    alg_emissions = {i: {} for i in algo}
    areas = csv_obj.list_all_area_wattTime()
    month_list = ['01', '02', '03', '04', '06', '08', '10', '11']
    # month_list = ['01']
    year = '2021'
    ref_days = ["01T00:00:00+00:00", "05T00:00:00+00:00", "10T00:00:00+00:00", "15T00:00:00+00:00",
                "20T00:00:00+00:00", "25T00:00:00+00:00"]
    # ref_days = ["01T00:00:00+00:00"]
    reg_path = []
    for i in areas:
        path_array = []
        for month in month_list:
            file_day = f"{year}-{month}"
            file_name = f"{i['name'].split('_')[0]}_{file_day}_MOER.csv"
            file_path = '/'.join(i['path'].split("/")[:3])
            path_array.append({'name': f"{file_path}/{file_name}", 'day': file_day})
        reg_path.append({'region': i['name'].split('_')[0],
                         'path': path_array})

    for reg in reg_path:
        print("Region", reg)
        for p in reg['path']:
            for al in algo:
                al_array = []

                for day in ref_days:
                    star_time = f"{p['day']}-{day}"
                    emission = \
                    TrainOptimization(start_time=star_time, workload=al, region_zone="other_regions").no_echo_mode(
                        emission_path=p['name'])[0]
                    al_array.append(emission)
                alg_emissions[al][reg['region']] = al_array
    alg_avg_emissions = {}

    for key, value in alg_emissions.items():
        avg_per_years = []
        for reg_key, reg_val in value.items():
            avg_per_years.append(np.asarray(reg_val).mean())

        alg_avg_emissions.update({key: np.asarray(avg_per_years).mean()})

    height = [value for key, value in alg_avg_emissions.items()]

    # Choose the names of the bars
    alg_name = [key for key, value in alg_avg_emissions.items()]

    fig, ax = plt.subplots()
    ax.bar(alg_name, height)

    ax.set_ylabel('gCO2eq')

    plt.tight_layout()

    plt.show()

    # wat=watt_time_controller.WattTimeController()
    # reg=['IT', 'UK', 'SE', 'FR', 'IE', 'ES', 'DE']
    # for r in reg:
    #     wat.get_historical_grid_emissions(grid_area_code=r)


if __name__ == "__main__":
    avarage_emissions_no_echo()
    #puntual_comparison(workload='isolation_forest')
    #avarage_reduction(workload='isolation_forest')
