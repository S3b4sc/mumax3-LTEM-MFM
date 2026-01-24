# Research pipeline launcher.
# This script evolved during the development of the thesis
# and contains both production and diagnostic utilities.


from python_scripts import read_plot, input_menu, generate_ltem_image, start_ovfs_gen, gen_ltem_dataset, start_gen, run_mz_dataset   
import subprocess
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ONe use only: rebuild training index csv file
import rebuild_training_index as rti


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
            #read_plot(in_route='./mumax_files/demo.out/paper_6', out_route_html="./html_plots/demo512_run1.html", binary_mz=False, z_color=True)
            #read_plot(in_route='./mumax_dataset_ku_by_block_disorder_phy_130ns/run_00022/final_22', out_route_html="./html_plots/demo73_run1.html", binary_mz=False, z_color=True)
            #mumax in_route = './mumax_files/PtCo.out/PtCo'
            #read_plot(in_route='./mumax_files/PtCo.out/PtCo', out_route_html="./html_plots/demo512.html")
    
    if usrChoice == 2:
        #generate_ltem_image(ovf_path='oommf_files/two_layer_withD-Oxs_TimeDriver-Magnetization-00-0005834.omf')
        #generate_ltem_image(ovf_path='./mumax_files/demo.out/final_run500.ovf')
        generate_ltem_image(ovf_path='./mumax_dataset_ku_by_block_disorder_phy_corrected_2/run_00001/final_1.ovf', save_path='./')

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
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_corrected_2/", save_path = './images/corrected_2/')  
        #run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_corrected_3/", save_path = './images/corrected_3/')  
        run_mz_dataset(ovf_path = "./mumax_dataset_ku_by_block_disorder_phy_outside_clamp/", save_path = './images/mumax_dataset_ku_by_block_disorder_phy_outside_clamp_both_above/')  
        
        
        
        
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
    
    elif usrChoice == 8:
        # FAllback for lsot raining csv file
        rti.main()
        print("Rebuilt training_index.csv file.") 
        
    elif usrChoice == 9:
        

        img = np.array(Image.open("images/corrected_3/mz_binary_14.png"))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Full histogram
        axes[0].hist(img.ravel(), bins=256, range=(0, 65535), log=True)
        axes[0].set_title("Full histogram (log scale)")
        axes[0].set_xlabel("Pixel value")

        # Zoom into middle range (domain walls)
        middle = img[(img > 5000) & (img < 60000)]
        axes[1].hist(middle.ravel(), bins=100)
        axes[1].set_title(f"Domain wall pixels only: {len(middle)} ({100*len(middle)/img.size:.1f}%)")

        # Show the image with enhanced contrast for walls
        axes[2].imshow(img, cmap='gray', vmin=20000, vmax=45000)
        axes[2].set_title("Enhanced contrast (domain walls visible)")

        plt.tight_layout()
        plt.savefig("diagnostic_histogram.png", dpi=150)
        plt.show()

        print(f"Pixels near 0 (<1000): {(img < 1000).sum()} ({100*(img < 1000).mean():.1f}%)")
        print(f"Pixels near max (>64000): {(img > 64000).sum()} ({100*(img > 64000).mean():.1f}%)")
        print(f"Domain wall pixels: {((img >= 1000) & (img <= 64000)).sum()} ({100*((img >= 1000) & (img <= 64000)).mean():.1f}%)")