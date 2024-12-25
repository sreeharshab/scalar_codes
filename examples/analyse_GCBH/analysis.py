from pipelines import analyse_GCBH

def energy_operation(e):
    return (e + 5.43*84 + 3.75*4)/4

if __name__=="__main__":
    analyse_GCBH(save_data=False, energy_operation=energy_operation, label="Aluminum Substitution Energy (eV/Al)")