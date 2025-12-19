from python_scripts import read_plot, input_menu, generate_ltem_image, start_ovfs_gen, gen_ltem_dataset, start_gen, run_mz_dataset   
import subprocess
import time


if __name__ == '__main__':
    usrChoice = input_menu()

    if usrChoice == 1:
        oommf_file = False

        if oommf_file:
            #oommf_in_archive = './oommf_files/two_layer_withD.mif'
            #subprocess.run(['oommf.tcl', 'boxsi', oommf_in_archive])
            read_plot(in_route='oommf_files/two_layer_withD-Oxs_TimeDriver-Magnetization-00-0005834.omf', out_route_html="./html_plots/oommf_zcolor.html",oommf_file=oommf_file, z_color=True)

        else: 

            # Run Mumax simulation
            mumax_in_archive = './mumax_files/demo.mx3'
            mumax_out_dir = './mumax_files'

            start = time.time()
            subprocess.run(['mumax3', mumax_in_archive])    
            end = time.time()
            print(f"Simulation time: {end - start} seconds")
            # Read and Plot the simulation results
            read_plot(in_route='./mumax_files/demo.out/final_run500', out_route_html="./html_plots/demo512_run1.html", binary_mz=False, z_color=True)
            #read_plot(in_route='./mumax_dataset_ku_by_block_disorder_phy_130ns/run_00022/final_22', out_route_html="./html_plots/demo73_run1.html", binary_mz=False, z_color=True)
            #mumax in_route = './mumax_files/PtCo.out/PtCo'
            #read_plot(in_route='./mumax_files/PtCo.out/PtCo', out_route_html="./html_plots/demo512.html")
    
    if usrChoice == 2:
        #generate_ltem_image(ovf_path='oommf_files/two_layer_withD-Oxs_TimeDriver-Magnetization-00-0005834.omf')
        generate_ltem_image(ovf_path='./mumax_files/demo.out/final_run500.ovf')
        #generate_ltem_image(ovf_path='./mumax_dataset_ku_by_block_disorder/run_00001/final_1.ovf', save_path='./images/LTEM_images_ku_by_blocks/')

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
        #gen_ltem_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder/", save_path = './images/LTEM_images_ku_by_blocks/')
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder/", save_path = './images/LTEM_images_ku_by_blocks_mz_binary/', binary=True)   
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy/", save_path = './images/images_ku_by_blocks_mz/', binary=False)  
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_100ns/", save_path = './images/images_ku_by_blocks_mz_100ns/', binary=False)  
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_130ns/", save_path = './images/images_ku_by_blocks_mz_130ns/', binary=False)  
        run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_corrected/", save_path = './images/corrected/')  
        
    elif usrChoice == 6:
        # Convert mumax ovf output to npy file
        #subprocess.run(['mumax3-convert', '-numpy','./mumax_dataset_ku_by_block_disorder/run_00001/final_1.ovf'])
        #subprocess.run(['mumax3-convert', '-png','./mumax_dataset_ku_by_block_disorder/run_00001/final_1.ovf'])
        # Read and Plot the simulation results
        read_plot(in_route='./mumax_dataset_ku_by_block_disorder/run_00600/final_600', out_route_html='./html_plots/plot_500number.html', z_color=True, binary_mz=False)
        #read_plot(in_route='./mumax_dataset_ku_by_block_disorder/run_00001/final_1', out_route_html='./html_plots/plot_512corrected.html', binary_mz=True)

    elif usrChoice == 7:
        duration = float(input("Enter the desired runtime in minutes (e.g., 120 for 2 hours): "))
        start_gen(max_runtime_minutes=duration)
        