import sys


def input_menu():
    message = '''
    ----------------------------------------------------------------------------------------
                            MENU: CHOOSE ONE OF THE FOLLOWING OPTIONS
    ----------------------------------------------------------------------------------------
    
    1   Run and visualize Mumax simulation from demo.mx3
    2   Run LTEM simulation from existing output demo.mx3
    3   Run MFM simulation from .mx3 
    4   
    5   
    6   
    7   
    8   
    9   
    10  

    ----------------------------------------------------------------------------------------
    '''
    
    try:
        usrChoice = int(input(message))
        return usrChoice
    
    except ValueError:
        sys.exit('Exiting... The input was not an integer. Run and try again.')
     