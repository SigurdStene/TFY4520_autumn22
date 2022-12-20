#TODO: Find overlap between the two objects in a reasonable way.
#TODO: Test with smaller k-space. See if the object gets smaller, and if there potentially are any ways to fix this.

#TODO: Convolve diffraction data with a gaussian kernel to simulate experimental noise.
#TODO: Convolve diffraction data with aproprate kernel to simulate experimental conditions, e.g. small sample etc.


#TODO: When looking at the error_analysis data. The objects in the gifs aren't symmetric. Meaning that something weird must have happened. How is the difference from the real and reconstructed object exactly zero. Don't believe it.

#TODO: Change the deletion/renaming of initial objects either in reconstruction or in error analysis in stead of diffraction.


#TODO: Find out if the files should be stored as npz compressed with the np.savez_compressed function.
#TODO: All the printing shuold just be done if the verbose flag is set to True. If not, just look in the logger file.
#TODO: Figure out if the reconstructed objects usually have sizes divisible by 2. If not, then the grid size should be changed to be divisible by 2. This was at least the case for the objects generated initially (even though they could be cubes with side length 33).


import getopt
import sys
import argparse
import six 
import time

from CDI_create_objects import ObjectGeneration
from CDI_diffraction import DiffractionExperiment
from CDI_reconstruction import Reconstruction
from CDI_error_analysis import ErrorAnalysis
from CDI_dictionaries import grid_size, object_dict_list, error_source_dict_list, number_of_reconstructions
from CDI_logger import Logger
from CDI_description import Description
from CDI_plot_results import PlotResults

#TODO: Important.
"""
    Diffucult to measure how much intensity is reduced due to the lack of knowledge of the total intensity due to the beam stop.
    Can try to look at proper diffraction patterns, but don't think it will help.
    Could rather try to simulate a real object, and then see how much intensity is lost due to the beam stop, by just removing a part of a 2D diffractogram corresponding to the size of the beamstop.
    This could give an approximation of the intensity lost due to the beam stop.
    Can then compare this with how much intensity is registered in the wide angle detector.
    And this should probably be integrated over a circle.

"""

# input_command = six.moves.input("Do you want to test this functionallity? (y/n)")





"""
def usage():
    return NotImplementedError


try:
    opts, args = getopt.getopt(sys.argv[1:], "tt", ["Test"])

    if len(opts) < 1:
        print("No or invalid options provided.")
        usage()
        sys.exit(1)
except getopt.GetoptError as err:
    print("GetoptError: {}".format(err))    
    usage()

for opt, arg in opts:
    if  opt == "-t":
        testing = 1
    elif opt == "-r":
        testing_men_med_r = 1
    else:
        print("Her har vi vel en case?")

def parsing():
    "Handles parsing of the command line arguments for this function."
    parser = argparse.ArgumentParser(description = "Command tool to input simulation parameters.")
    subparsers = parser.add_subparsers(dest = "cmd")

    #Define subparsers. Different functions that might be run. Could probably create this as different files
    # and have the structure input as different parsers.

    parser_df = subparsers.add_parser("diffraction",   help =  "Perform diffraction on given dataset.")
    parser_rc = subparsers.add_parser("reconstruction", help = "Reconstruct 3D object from diffraction data.")

    #Have to ensure that support update is defined. If not, define generic support from the diffraction data or something. Idk.
    parser_rc.add_argument("algorithm", type = str, default = "ER**200,(sup*HIO**20)**40", help = "Reconstruction algorithm.")

    #Optional parameters valid for the above subparsers.
    parser.add_argument("-T", "-testing", action = "test", help = "det testes")

    return parser



def reconstruction():
    return NotImplementedError


def select_cmd(cmd, parser):
    if cmd == "diffraction":
        return diffraction
    elif cmd == "reconstruction":
        return reconstruction

"""
def main():
    """
        TODO: Have the ability to do multiple processes and give in a range of possibilities.
    """
    # parser = parsing()
    # args = parser.parse_args() 
    # if args.verbose == True:
    #     print("\n" + 10*"*" + " Verbose output " + 10*"*")
    #     for key in sorted(args.__dict__):
    #         print("{}{}{}".format(key, " "*(17 - len(key)), args.__dict__[key]))
    # cmd_function = select_cmd(args.cmd, parser)
    
    #TODO: Fix this in a better way. This is just a quick fix.
    i = 0

    #Testing with smaller size to speed things up-
    size = grid_size*2
    group_filename_prefix = f"normal_wloopy_{size}"

    for ind_o, object_dict in enumerate(object_dict_list):
        for ind_e, error_source_dict in enumerate(error_source_dict_list):
            print(f"Object {ind_o} and error source combination {ind_e}")
            print(f"Combination {1 + ind_e + ind_o*len(error_source_dict_list)} of {len(object_dict_list)*len(error_source_dict_list)}")

            try:
                #TODO: Fix this
                #Now, the intensity reduction value comes first. Simpler. Should still be fixed properly.r
                filename_prefix = f"{group_filename_prefix}_{error_source_dict['index']}_{object_dict['index']}_{size}"
            except KeyError:
                filename_prefix = f"{group_filename_prefix}_unspecified_{i}_{size}"
                i += 1
             
            logger = Logger(filename_prefix)
            description = Description(filename_prefix)
            description.add_description("grid_size", grid_size)
            description.add_description("k_size", size)
            #Object generation
            try:
                print(f"\nGenerating object from object dictionary {filename_prefix}")
                logger.add_log(f"Generating object from object dictionary {filename_prefix}")
                logger.add_log(f"Object generation started at {time.asctime()}")
                
                ObjectGeneration(object_dict=object_dict, filename_prefix=filename_prefix, logger=logger, grid_size=grid_size, description = description, verbose = False)
                
                logger.add_log(f"Object generation ended at {time.asctime()}\n")
            
            except Exception as e:
                print(f"\nError in object generation: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in object generation: {e}")
                logger.add_log("Continuing with next object.\n")
                continue
            
            #Diffraction
            try:
                print(f"\nPerforming diffraction on objects given by {filename_prefix}")
                logger.add_log(f"Performing diffraction on objects given by {filename_prefix}")
                logger.add_log(f"Diffraction started at {time.asctime()}")
                
                DiffractionExperiment(filename_prefix=filename_prefix, nh = size, nk = size, nl = size, logger=logger, description=description, error_source_dict=error_source_dict)
                
                logger.add_log(f"Diffraction ended at {time.asctime()}\n")

            except Exception as e:
                print(f"\nError in diffraction experiment: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in diffraction experiment: {e}")
                logger.add_log("Continuing with next object.\n")
                logger.save_log()
                continue
            
            #Reconstruction
            try:
                print(f"\nReconstructing object from diffraction data given by {filename_prefix}")
                logger.add_log(f"Reconstructing object from diffraction data given by {filename_prefix}")
                logger.add_log(f"Reconstruction started at {time.asctime()}")
                
                Reconstruction(filename_prefix=filename_prefix, number_of_reconstructions=number_of_reconstructions, logger=logger, description=description, algorithm=None, mask=None, error_source_dict=error_source_dict, verbose=False)
                
                logger.add_log(f"Reconstruction ended at {time.asctime()}\n")
            
            except Exception as e:
                print(f"\nError in reconstruction: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in reconstruction: {e}")
                logger.add_log("Continuing with next object.\n")
                logger.save_log()
                continue
            
            #Error analysis. Done on all the reconstructions together.
            try:
                print(f"\nError analysis of reconstruction given by {filename_prefix}")
                logger.add_log(f"Error analysis of reconstruction given by {filename_prefix}")
                logger.add_log(f"Error analysis started at {time.asctime()}")

                ErrorAnalysis(group = group_filename_prefix, filename_prefix=filename_prefix, logger=logger, description=description, plot_gifs=False)

                logger.add_log(f"Error analysis ended at {time.asctime()}\n")

            except Exception as e:
                print(f"\nError in error analysis: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in error analysis: {e}")
                logger.add_log("Continuing with next object.\n")
                logger.save_log()
                continue

            
            logger.save_log()
            description.save_description()

    logger = Logger(group_filename_prefix)
    description = Description(group_filename_prefix)
    PlotResults(group = group_filename_prefix, logger=logger, description=description)
    logger.save_log()
    description.save_description()

def main_proper():
     #TODO: Fix this in a better way. This is just a quick fix.
    i = 0

    #Testing with smaller size to speed things up-
    size = grid_size*2
    group_filename_prefix = f"normal_stds_fixed_mean_{size}"

    for ind_o, object_dict in enumerate(object_dict_list):
        logger = Logger(group_filename_prefix)
        description = Description(group_filename_prefix)
        description.add_description("grid_size", grid_size)
        description.add_description("k_size", size)
        #Object generation
        try:
            print(f"\nGenerating object from object dictionary {filename_prefix}")
            logger.add_log(f"Generating object from object dictionary {filename_prefix}")
            logger.add_log(f"Object generation started at {time.asctime()}")
            
            ObjectGeneration(object_dict=object_dict, filename_prefix=filename_prefix, logger=logger, grid_size=grid_size, description = description, verbose = False)
            
            logger.add_log(f"Object generation ended at {time.asctime()}\n")
        
        except Exception as e:
            print(f"\nError in object generation: {e}")
            print("Continuing with next object.")
            logger.add_log(f"Error in object generation: {e}")
            logger.add_log("Continuing with next object.\n")
            continue
        
        #Diffraction: Should only be done one time per object. 
        #TODO: Make own class to apply error sources to the diffraction data.
        try:
            print(f"\nPerforming diffraction on objects given by {filename_prefix}")
            logger.add_log(f"Performing diffraction on objects given by {filename_prefix}")
            logger.add_log(f"Diffraction started at {time.asctime()}")
            
            DiffractionExperiment(filename_prefix=filename_prefix, nh = size, nk = size, nl = size, logger=logger, description=description, error_source_dict=error_source_dict)
            
            logger.add_log(f"Diffraction ended at {time.asctime()}\n")

        except Exception as e:
            print(f"\nError in diffraction experiment: {e}")
            print("Continuing with next object.")
            logger.add_log(f"Error in diffraction experiment: {e}")
            logger.add_log("Continuing with next object.\n")
            logger.save_log()
            continue
    


        for ind_e, error_source_dict in enumerate(error_source_dict_list):
            print(f"Object {ind_o} and error source combination {ind_e}")
            print(f"Combination {1 + ind_e + ind_o*len(error_source_dict_list)} of {len(object_dict_list)*len(error_source_dict_list)}")

            try:
                #TODO: Fix this
                #Now, the intensity reduction value comes first. Simpler. Should still be fixed properly.r
                filename_prefix = f"{group_filename_prefix}_{error_source_dict['index']}_{object_dict['index']}_{size}"
            except KeyError:
                filename_prefix = f"{group_filename_prefix}_unspecified_{i}_{size}"
                i += 1
             
            logger = Logger(filename_prefix)
            description = Description(filename_prefix)
            
            #Reconstruction
            #The reconstruction loads the diffraction data from the file, so it should not be nescary to store basically multiple copies. Can just apply the reduction after the loading.
            #Have to store the different f_hkls in that case. A lot of restructuring is needed. Can try to implement more of the information in the desciption file.
            try:
                print(f"\nReconstructing object from diffraction data given by {filename_prefix}")
                logger.add_log(f"Reconstructing object from diffraction data given by {filename_prefix}")
                logger.add_log(f"Reconstruction started at {time.asctime()}")
                
                Reconstruction(filename_prefix=filename_prefix, number_of_reconstructions=number_of_reconstructions, logger=logger, description=description, algorithm=None, mask=None, error_source_dict=error_source_dict, verbose=False)
                
                logger.add_log(f"Reconstruction ended at {time.asctime()}\n")
            
            except Exception as e:
                print(f"\nError in reconstruction: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in reconstruction: {e}")
                logger.add_log("Continuing with next object.\n")
                logger.save_log()
                continue
            
            #Error analysis. Done on all the reconstructions together.
            try:
                print(f"\nError analysis of reconstruction given by {filename_prefix}")
                logger.add_log(f"Error analysis of reconstruction given by {filename_prefix}")
                logger.add_log(f"Error analysis started at {time.asctime()}")

                ErrorAnalysis(group = group_filename_prefix, filename_prefix=filename_prefix, logger=logger, description=description)

                logger.add_log(f"Error analysis ended at {time.asctime()}\n")

            except Exception as e:
                print(f"\nError in error analysis: {e}")
                print("Continuing with next object.")
                logger.add_log(f"Error in error analysis: {e}")
                logger.add_log("Continuing with next object.\n")
                logger.save_log()
                continue

            
            logger.save_log()
            description.save_description()

    logger = Logger(group_filename_prefix)
    description = Description(group_filename_prefix)
    PlotResults(group = group_filename_prefix, logger=logger, description=description)
    logger.save_log()
    description.save_description()




if __name__ == "__main__":
    main()