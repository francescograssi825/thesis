IMPORTANTE

-Ricorda di aggiungere le seguenti righe di codice nel file emission.tracker dalla riga 588 in poi :
## CUSTOM ADD BY ME IN ORDER TO GET ENERGY CONSUMPTION IN THE CSV EACH N SECONDS ##
        consumption_dict = {
            "timestamp": datetime.now(),
            "cpu_energy_consumed": self._total_cpu_energy.kWh,
            "gpu_energy_consumed": self._total_gpu_energy.kWh,
            "ram_energy_consumed": self._total_ram_energy.kWh,
            "total_consumption": self._total_energy.kWh
        }
        if os.path.exists('csv_dir/Custom_Consumption.csv'):
            with open('csv_dir/Custom_Consumption.csv', 'a') as f:
                w = csv.DictWriter(f, fieldnames=consumption_dict.keys())
                w.writerow(consumption_dict)
        else:
            with open('csv_dir/Custom_Consumption.csv', 'w') as f:
                w = csv.DictWriter(f, fieldnames=consumption_dict.keys())
                w.writeheader()
                w.writerow(consumption_dict)
Modificano la libreria in maniera tale da scrivere su csv ogni 5 min i consumi energetici