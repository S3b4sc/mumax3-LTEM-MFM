import sys


def input_menu():
    message = '''
    ----------------------------------------------------------------------------------------
                    MENU – Research pipeline launcher (Mumax / LTEM / ML)
    ----------------------------------------------------------------------------------------

    1   Run and visualize a single Mumax demo simulation
        - Uses demo.mx3
        - For quick physical sanity checks

    2   Generate one LTEM image from an existing OVF file
        - Forward LTEM contrast simulation
        - Used for validation and examples

    3   Run one Mumax/MFM simulation and convert output
        - Executes demo.mx3 and converts OVF to PNG/NumPy
        - Mostly legacy / exploratory

    4   Generate OVF files for full parameter sweep
        - Core dataset generation step (MuMax runs)

    5   Generate image datasets from existing OVF runs
        - Converts OVF → mz / LTEM images for ML training

    6   Visualize a specific OVF run
        - Manual inspection of individual simulations

    7   Time-limited batch generation
        - Run dataset generation for a fixed duration (e.g. overnight)

    8   Rebuild training_index.csv (recovery tool)
        - Used if training metadata CSV is lost or corrupted

    9   Diagnostic histogram analysis of generated images
        - Pixel-value statistics and domain-wall inspection

    0   Exit

    ----------------------------------------------------------------------------------------
    NOTE:
    This menu reflects the historical evolution of the project.
    Not all options are required to reproduce the final results.
    Some options are legacy or for exploratory purposes.
    '''
    
    try:
        usrChoice = int(input(message))
        return usrChoice
    
    except ValueError:
        sys.exit('Exiting... The input was not an integer. Run and try again.')
     