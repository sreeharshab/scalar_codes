from pipelines import gibbs_free_energy

if __name__=="__main__":
    temperature = 300
    potential = 0.5
    
    calc_root_dir = "/path/to/root"
    calc = gibbs_free_energy(calc_root_dir)
    G0 = calc.get_gibbs_free_energy(temperature, potential=potential)