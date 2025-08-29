from python_scripts import read_plot, input_menu, generate_ltem_image
import subprocess



if __name__ == '__main__':
    usrChoice = input_menu()

    if usrChoice == 1:
        # Run Mumax simulation
        mumax_in_archive = './mumax_files/demo.mx3'
        mumax_out_dir = './mumax_files'
        subprocess.run(['mumax3', mumax_in_archive])

        # Convert mumax ovf output to npy file
        subprocess.run(['mumax3-convert', '-numpy','./mumax_files/demo.out/final.ovf'])

        # Read and Plot the simulation results
        read_plot(in_route='./mumax_files/demo.out/final.npy')
    
    if usrChoice == 2:
        generate_ltem_image()