from python_scripts import read_plot, input_menu, generate_ltem_image, start_ovfs_gen, gen_ltem_dataset
import subprocess



if __name__ == '__main__':
    usrChoice = input_menu()

    if usrChoice == 1:
        oommf_file = True

        if oommf_file:
            #oommf_in_archive = './oommf_files/two_layer_withD.mif'
            #subprocess.run(['oommf.tcl', 'boxsi', oommf_in_archive])
            read_plot(in_route='oommf_files/two_layer_withD-Oxs_TimeDriver-Magnetization-00-0005834.omf', out_route_html="./html_plots/oommf_zcolor.html",oommf_file=oommf_file, z_color=True)

        else: 

            # Run Mumax simulation
            mumax_in_archive = './mumax_files/PtCo.mx3'
            mumax_out_dir = './mumax_files'
            subprocess.run(['mumax3', mumax_in_archive])    

            # Read and Plot the simulation results
            read_plot(in_route='./mumax_files/demo.out/final.npy', z_color=False)
            #mumax in_route = './mumax_files/PtCo.out/PtCo'
            read_plot(in_route='./mumax_files/PtCo.out/PtCo', oommf_file=oommf_file)
    
    if usrChoice == 2:
        generate_ltem_image(ovf_path='oommf_files/two_layer_withD-Oxs_TimeDriver-Magnetization-00-0005834.omf')

    elif usrChoice == 3:
        # Run Mumax simulation
        mumax_in_archive = './mumax_files/demo.mx3'
        mumax_out_dir = './mumax_files'
        subprocess.run(['mumax3', mumax_in_archive])

        # Convert mumax ovf output to npy file
        subprocess.run(['mumax3-convert', '-png','./mumax_files/demo.out/final.ovf'])

    elif usrChoice == 4:
        start_ovfs_gen()

    elif usrChoice == 5:
        gen_ltem_dataset()
    
    elif usrChoice == 6:
        # Convert mumax ovf output to npy file
        subprocess.run(['mumax3-convert', '-numpy','./mumax_files/logs/final_154.ovf'])
        # Read and Plot the simulation results
        read_plot(in_route='./mumax_files/logs/final_154.npy', out_route_html='./html_plots/plot_154.html', z_color=False)