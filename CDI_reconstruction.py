import numpy as np
import os
import time
# This imports all necessary operators. GPU will be auto-selected
from pynx.cdi import *
from CDI_logger import Logger
from CDI_dictionaries import error_source_dict_list
from CDI_description import Description
import raster_geometry as rg

class Reconstruction():
    """
        This class takes the diffraction pattern as input and reconstructs the object according to the given algorithm.
    """
    def __init__(self, filename_prefix, number_of_reconstructions, logger: Logger, description: Description, error_source_dict, algorithm = None, mask = None, verbose = False, test = False):
        self.filename_prefix = filename_prefix
        if test:
            self.filename_prefix += "_test"
            self.filename = self.filename_prefix + ".npy"
        else:
            self.filename = self.filename_prefix + ".npy"
        self.number_of_reconstructions = number_of_reconstructions
        
        self.logger = logger        
        self.description = description
        self.description.add_description("number_of_reconstructions", self.number_of_reconstructions)
        
        self.current_path = os.getcwd()
        if not os.path.isdir(os.path.join(self.current_path, "reconstructions")):
            os.mkdir(os.path.join(self.current_path, "reconstructions"))
            time.sleep(1)
        self.save_path = os.path.join(self.current_path, "reconstructions", self.filename_prefix)

        self.load_path = os.path.join(self.current_path, "diffraction_data")

        self.dif_pat = self.load_diffraction_pattern()
        self.algorithm = algorithm

        self.error_source_dict = error_source_dict
        self.unpack_error_source_dict()

        self.support = self.create_support()
        
        #TODO: This can obviously not be like this.
        self.mask = mask
        self.create_mask()

        self.reconstruction()
        
    # Have to base the parameters on the loaded diffraction pattern.
    # def set_standard_error_parameters(self):
    #     """
    #         Function to set the standard error parameters.
    #         If beamstop is given, but no radius, the radius is set to the largest possible radius.
    #     """        
    #     # if self.beamstop and self.beamstop_radius == None:
    #     #     self.beamstop_radius = np.sqrt(self.nh**2 + self.nk**2 + self.nl**2)/2
    #     if self.cylindrical_reciprocal_space:
    #             self.logger.add_log("\tUsing cylindrical reciprocal space.")
    #             if self.cylindrical_reciprocal_space_radius == None:
    #                 self.logger.add_log(f"\tNo radius for the cylindrical reciprocal space was given. Using the largest reciprocal space dimension as diameter.")
    #                 self.logger.add_log(f"\t\tI.e., radius: {self.nh//2}")
    #                 self.cylindrical_reciprocal_space_radius = self.nh//2
    #     if self.beamstop:
    #         self.logger.add_log("\tUsing a beamstop.")
    #         if self.beamstop_radius == None:
    #             self.logger.add_log(f"\t\tNo radius for the beamstop was given. Using standard values of h_range/10 = {self.h_range/10}.")
    #             self.beamstop_radius = self.nh//2

    def unpack_error_source_dict(self):
        """
            Saves the different parameters to member variables of the Reconstruction class.
            Parameter:
                beamstop
                beamstop_radius
                cylindrical_rec_space
                cylindrical_rec_space_radius
                + more to come
        """
        self.logger.add_log("\tUnpacking error source dictionary.")
        try:
            self.beamstop = self.error_source_dict["beamstop"]
            self.logger.add_log(f"\t\tBeamstop: {self.beamstop}")
        except KeyError:
            self.beamstop = False
            self.logger.add_log("\t\tNo beamstop in error source dictionary.")
            self.logger.add_log("\t\tBeamstop: False")
        try:
            #TODO: Fix magic number. self.nh in diffraction.
            self.beamstop_radius = self.dif_pat.shape[0]*self.error_source_dict["beamstop_radius_factor"]
            self.logger.add_log(f"\t\tBeamstop radius: {self.beamstop_radius}")
        except KeyError:
            self.beamstop_radius = None
            self.logger.add_log("\t\tNo beamstop radius in error source dictionary.")
            self.logger.add_log("\t\tBeamstop radius: None")
        try:
            self.cylindrical_rec_space = self.error_source_dict["cylindrical_rec_space"]
            self.logger.add_log(f"\t\tCylindrical reciprocal space: {self.cylindrical_rec_space}")
        except KeyError:
            self.cylindrical_rec_space = False
            self.logger.add_log("\t\tNo cylindrical reciprocal space in error source dictionary.")
            self.logger.add_log("\t\tCylindrical reciprocal space: False")
        try:   
            self.cylindrical_rec_space_radius = self.error_source_dict["cylindrical_rec_space_radius"]
            self.logger.add_log(f"\t\tCylindrical reciprocal space radius: {self.cylindrical_rec_space_radius}")
        except KeyError:
            self.cylindrical_rec_space_radius = None
            self.logger.add_log("\t\tNo cylindrical reciprocal space radius in error source dictionary.")
            self.logger.add_log("\t\tCylindrical reciprocal space radius: None")
        self.logger.add_log("\tUnpacking done.")

    def create_mask(self):
        """
            Creates a mask for the reconstruction.
        """
        self.logger.add_log("\tCreating mask based on error sources.")
        if self.beamstop and self.cylindrical_rec_space:
            self.mask = np.array(rg.sphere(self.dif_pat.shape, radius = self.beamstop_radius) | ~rg.cylinder(self.dif_pat.shape, radius = self.cylindrical_rec_space_radius, height = self.dif_pat.shape[0]), dtype = int)
        elif self.beamstop:
            self.mask = np.array(rg.sphere(self.dif_pat.shape, radius = self.beamstop_radius), dtype = int)
        elif self.cylindrical_rec_space:
            self.mask = np.array(~rg.cylinder(self.dif_pat.shape, radius = self.cylindrical_rec_space_radius, height = self.dif_pat.shape[0]), dtype = int)
        else:
            self.mask = np.zeros_like(self.dif_pat)

    def create_support(self):
        """
            Create support based on size of fhkl. 
            TODO: Find out what values are reasonable to use and why.
            TODO: Implement further parameters so that the support easily can be adjusted. Don't know if a circular support neccessarily(#?) is the best thing either.

            #? As of now 08.10 (8.10??). I'm only using some random values that worked nicely with the initial tests.
        """
        self.logger.add_log("\tCreating support based on size of diffraction pattern.")
        self.logger.add_log(f"\t\tShape of diffraction pattern: {self.dif_pat.shape}")
        self.logger.add_log(f"\t\tlen(self.dif_pat): {len(self.dif_pat)}")
        tmp = np.arange(-len(self.dif_pat)/2, len(self.dif_pat)/2)
        xx, yy, zz = np.meshgrid(tmp, tmp, tmp)
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        #Support should be approximately 1/4-5 of the size of the diffraction pattern. See Chapman/ask D.
        support = r < len(self.dif_pat)/4
        self.logger.add_log("\tSupport created.")
        self.logger.add_log(f"\t\tShape of support: {support.shape}")
        # Fourier shift is done when initializing the reconstruction.
        return support
    
    def load_diffraction_pattern(self):
        """
            Loads the diffraction pattern from the given filename.
        """
        self.logger.add_log(f"\tLoading diffraction pattern from {self.filename}")
        return np.load(os.path.join(self.load_path, self.filename))

    def reconstruction(self, object = None, wavelength = 1e-10, pixel_size_detector = 55e-6):
        """
            Detector: Danyial's detector
                - 75e-6 pixel size
                - 7.15 m sample detector distance (flere forskjellige)
                - Kanskje: Energy: 7.239983 = 7.24 keV - Wavelength: X1.7125Å
                - Later 8 keV.


            TODO: Find out if one can include real space object, and what this potentially does.
            TODO: Find suitable values for wavelength and pixel_size_detector. These should probably be calculated with regard to the values used in the simulation. 
                They might actually have to be dependent of each other.
            TODO: Alter the filename based on the settings that are used. Should not have to be changed manually. Have to be implemented in a main function of sorts.

            If no mask is given. Assumme no mask at all, i.e. zeros. Mask equals beam stop etc.(#?) 
        """
        # if self.mask == None:
        #     self.logger.add_log("\tNo mask given. Assuming no mask at all. That is, zeros.")
        #     mask = np.zeros_like(self.dif_pat)
        

        # Support update operator
        thr_rel = 0.28
        method = "rms"
        smooth_width = (2,0.5,600)
        force_shrink = False
        post_expand = None
        # AutoCorrelationSupport
        #Cheating. max_fraction should not be larger than 1. Don't know why all the points end up in the support, or what that even means.
        sup = SupportUpdate(threshold_relative=thr_rel, method=method, smooth_width=smooth_width,
                            force_shrink=force_shrink, post_expand=post_expand, verbose=True, max_fraction=1.1)
        self.logger.add_log("\tSupport update operator created. With the following parameters:")
        self.logger.add_log(f"\t\tthreshold_relative: {sup.threshold_relative}")
        self.logger.add_log(f"\t\tmethod: {sup.method}")
        self.logger.add_log(f"\t\tsmooth_width: {sup.smooth_width}")
        self.logger.add_log(f"\t\tforce_shrink: {sup.force_shrink}")
        self.logger.add_log(f"\t\tpost_expand: {sup.post_expand}")
        # TODO: I.e. the guys above here.

        # Just some temporary debugging stuff. Should be removed.
        # print(cdi.iobs.size/(self.dif_pat.shape[0]**3))
        for i in range(self.number_of_reconstructions):
            self.logger.add_log(f"\tStarting reconstruction {i+1}/{self.number_of_reconstructions}")
                    
            #Create CDI object.
            self.logger.add_log("\tCreating CDI object.")
            cdi = CDI(np.fft.fftshift(self.dif_pat), obj = object, support=np.fft.fftshift(self.support), mask = np.fft.fftshift(self.mask), wavelength=wavelength, pixel_size_detector=pixel_size_detector)

            #If log-likelihood is used. Have to read more about that.
            # TODO: Read more about log-likelihood.
            # cdi.init_free_pixels()

            # TODO: Figure out what all of these guys are doing.
            # Initial object
            cdi = InitObjRandom(src="support", amin=0, amax=1, phirange=0) * cdi
            self.logger.add_log("\tInitial random object created.")
            # Initial scaling, required by mask
            cdi = ScaleObj(method='F') * cdi
            self.logger.add_log("\tInitial scaling of object finished.")
            
            # TODO: Implement a smart way to controll the algorith. Can probably look at the script they have already written, i.e. run.py.
            # TODO: Learn more about the actual reconstruction algorithms.
            cdi = (sup * HIO(calc_llk=0, positivity=True, fig_num=-1, show_cdi=False, zero_mask=True) ** 25)**6 * cdi

            cdi = DetwinHIO(positivity=True, detwin_axis=0, zero_mask=True)**24 * cdi

            cdi = (sup * HIO(calc_llk=250, positivity=True, fig_num=-1, show_cdi=False, zero_mask=True) ** 25)**16 * cdi

            cdi = (sup * ER(calc_llk=150, positivity=True, fig_num=-1, show_cdi=False, zero_mask=True) ** 25)**16 * cdi

            cdi.save_obj_cxi(self.save_path + f"_{i}.cxi")
            self.logger.add_log(f"\tReconstruction {i+1}/{self.number_of_reconstructions} finished.")
        
            # TODO: What is the actual interesting object here. Should I look at the amplitude, the phase or the magnitude?
            # reconstructed_object = cdi.get_obj()
            del(cdi)
            # # TODO: Check how large these matrices are. Could be a problem if each matrix is several GB. Should not have to copy them all the time.
            # rec_obj_amplitude, rec_obj_phase = reconstructed_object.real, reconstructed_object.imag
            # # TODO: Check if this destroys the other ones in the function as well. Should not be a problem.
            # del(reconstructed_object)

# Currently not in use. Might be useful later on.
def main():
    logger = Logger("test_reconstruction")
    Reconstruction(filename_prefix = "3000000_0001", logger = logger, test = True, error_source_dict = error_source_dict_list[1])
    logger.save_log()

if __name__ == "__main__":
    main()

