"""
    Code to simulate scattering and reconstruction of a star-shaped object.
    Heavily based on the tutorial from TODO: Insert link to tutorial.

    When creating objects with different densities. It would be easier to implement it in a relative fashion were the largest one has the value 1.
    In that way one can just divide the reconstructed object by its largest value to normalize it. 

    TODO: modulo 2pi on the imaginary part of the reconstructed object? I.e. the phase. In that case one should only consider the real part when looking at the reconstructed electron denstity.

    TODO: Fix plotting of dif_pat. Should not start with 0 in the corner, but in the center. Could try to fftshift to coordinates without the value...
          That is, when plotting and the x-values have been separated from the y-values.

    TODO: Create functions to visualize the diffraction patterns. Does not have to be in the main run, but should be able to analyse them separately in this file.
                    Start by doing it in the test file. Then move it here.
                    Should see a circular diffraction pattern when looking in the kx ky plane. 
                    At the same time, each measurement is quadratic, so have to figure out what is actucally going to be cylindrical. Might be kl after all.
    TODO: Would be nice to apply cylindrical k-space/beamstop before doing the diffraction. Stupid to do many calculations just to ignore them later on, but seems to be difficult with regards to the structure of the Fhkl function.
          Probably not an issue on gamma, but might be on the laptop. Things seem to be going very slowly if the object is large.

    TODO: Difficult to implement multiple intensity reduction at once. Shouldn't multiply with 0.9 multiple times.
    
    TODO: Why is the lower border zero? 

    TODO: Normalize intensity to 1e10 after the reduction in order to be able to compare the reconstructions properly, now the ones with less reduction will have a higher value for the reconstructed object (at least it seems like that). This is not how they should differ.
"""

import raster_geometry as rg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pynx.scattering.fhkl import Fhkl_thread
# from fhkl_copy import Fhkl_thread
import os
from numpy.fft import fftshift
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from CDI_create_objects import *
from CDI_logger import Logger
from CDI_description import Description
from CDI_dictionaries import error_source_dict_list
import sys
from scipy.ndimage import rotate

# This imports all necessary operators. GPU will be auto-selected
from pynx.cdi import *




# The GPU name and language can be specified, or will be automatically selected
gpu_name=""
language=""


#TODO: Find function that calculates the relation between angles in real space and the coordinates in reciprocal space.
#Think it's hidden somewhere in the pynx code.

class DiffractionExperiment:
    """
        Class to simulate diffraction experiments.
    """
    def __init__(self, filename_prefix, logger: Logger, description: Description, error_source_dict, nh = 4**4, nk = 4**4, nl = 4**4, 
                 h_range = 1, k_range = 1, l_range = 1, h_offset = 0, k_offset = 0, l_offset = 0, 
                 test = False):
        
        """
            Constructor for the DiffractionExperiment class.
            Starts by storing all the necessary parameters.
            Then creates the reciprocal space grid.
            Then loads the object(s) and performs a diffraction simulation, by first looping through each individual object, before adding the calculated fhkls together and squaring to get the diffraction pattern intensity.
            Finally, it applies a beamstop or a cylindrical cut of the data if wanted, before saving the diffraction pattern.
        """

        # Store all the parameters
        self.logger = logger
        self.description = description
        self.error_source_dict = error_source_dict
        self.test = test

        self.nh = nh
        self.nk = nk
        self.nl = nl
        self.h_range = h_range
        self.k_range = k_range
        self.l_range = l_range
        self.h_offset = h_offset
        self.k_offset = k_offset
        self.l_offset = l_offset
        # #TODO: These functions are the same for the diffraction and reconstruction class. Find a way to make them the same.
        self.unpack_error_source_dict()
        # #This sets the beamstop radius if beamstop is given, but no radius.
        # #Obs: As things are set up now, one loops through all the different radii even though beamstop is False. Will have to change this.
        self.set_standard_error_parameters()

        self.current_path = os.getcwd()        
        if not os.path.isdir(os.path.join(self.current_path, "diffraction_data")):
            os.mkdir(os.path.join(self.current_path, "diffraction_data"))
            time.sleep(1)
        self.save_path = os.path.join(self.current_path, "diffraction_data")   
        self.object_path = os.path.join(self.current_path, "objects")
        self.filename_prefix = filename_prefix

        # Create the reciprocal space grid
        self.h, self.k, self.l = self.create_reciprocal_space()

        # Load the objects
        self.relevant_files = self.find_relevant_files()
        #List to organize the diffraction data from different objects.
        self.fhkls = []
        self.diffraction_pattern = None
        #This funciton calls the self.calculate_form_factors() function, which fills the self.fhkls list.
        self.load_coordinates_and_density()
        if self.int_red: self.apply_intensity_reductions()
        else: description.add_description("intensity_reduction", "none")
        self.calculate_intensity()
        
        # Apply a beamstop or a cylindrical cut of the data if wanted
        # if self.beamstop: self.apply_beamstop()
        # if self.cylindrical_reciprocal_space: self.apply_cylindrical_reciprocal_space()
        # if self.beamstop or self.cylindrical_reciprocal_space: self.save_mask()
        # TODO: Implement into error source dictionary.
        self.save_diffraction_pattern()
        self.change_filename_prefix()
        self.logger.add_log("\nDone with diffraction experiment.\n\n")

    def apply_intensity_reductions(self):
        #Rotation misalignment between the different part.
        #TODO: Have to implement this into the error source dictionary.
        self.misalignment_objects = 45 / 180 * np.pi
        self.misalignment_parts = 20 / 180 * np.pi
        self.logger.add_log(f"\n\t\tApplying intensity reduction with misalignment of {self.misalignment_objects*180/np.pi} and {self.misalignment_parts*180/np.pi} between objects and parts of same object.\n")
        
        total_intensity_pre_reduction = 0
        for fhkl in self.fhkls:
            if isinstance(fhkl, list):
                for fhkl_part in fhkl:
                    total_intensity_pre_reduction += np.sum(fhkl_part**2)
            else:
                total_intensity_pre_reduction += np.sum(fhkl**2)
        self.logger.add_log(f"\t\t\tTotal intensity before intensity reduction: {total_intensity_pre_reduction}")

        for ind_outer, fhkl in enumerate(self.fhkls):
            if isinstance(fhkl, list):
                for ind_inner, fhkl_object in enumerate(fhkl):
                    #TODO: Change to list, or change such that angle = self.int_red_angle + ind_outer + ... as a matrix operation.
                    if isinstance(self.int_red_angle, np.ndarray):
                        self.logger.add_log(f"\t\t\tApplying intensity reduction to object {ind_outer}, part {ind_inner}.")
                        for angle in self.int_red_angle:
                            #This doesn't really make sense. Can't apply reduction to the same place with different intensities.
                            if isinstance(self.int_red_pre, np.ndarray):
                                for pre in self.int_red_pre:
                                    fhkl_object *= self.apply_intensity_reduction(angle + ind_outer*self.misalignment_objects + ind_inner*self.misalignment_parts, pre)
                            else:
                                fhkl_object *= self.apply_intensity_reduction(angle + ind_outer*self.misalignment_objects + ind_inner*self.misalignment_parts, self.int_red_pre)
                    else:
                        self.logger.add_log(f"\t\t\tApplying intensity reduction to object {ind_outer}.")
                        if isinstance(self.int_red_pre, np.ndarray):
                            for pre in self.int_red_pre:
                                fhkl_object *= self.apply_intensity_reduction(self.int_red_angle + ind_outer*self.misalignment_objects + ind_inner*self.misalignment_parts, pre)
                        else:
                            fhkl_object *= self.apply_intensity_reduction(self.int_red_angle + ind_outer*self.misalignment_objects + ind_inner*self.misalignment_parts, self.int_red_pre)
            else:
                if isinstance(self.int_red_angle, np.ndarray):
                    for angle in self.int_red_angle:
                        if isinstance(self.int_red_pre, np.ndarray):
                            for pre in self.int_red_pre:
                                fhkl *= self.apply_intensity_reduction(angle + ind_outer*self.misalignment_objects, pre)
                        else:
                            fhkl *= self.apply_intensity_reduction(angle + ind_outer*self.misalignment_objects, self.int_red_pre)
                else:
                    if isinstance(self.int_red_pre, np.ndarray):
                        for pre in self.int_red_pre:
                            fhkl *= self.apply_intensity_reduction(self.int_red_angle + ind_outer*self.misalignment_objects, pre)
                    else:
                        fhkl *= self.apply_intensity_reduction(self.int_red_angle + ind_outer*self.misalignment_objects, self.int_red_pre)
        total_intensity_post_reduction = 0
        for fhkl in self.fhkls:
            if isinstance(fhkl, list):
                for fhkl_part in fhkl:
                    total_intensity_post_reduction += np.sum(fhkl_part**2)
            else:
                total_intensity_post_reduction += np.sum(fhkl**2)
        self.logger.add_log(f"\t\t\tTotal intensity after intensity reduction: {total_intensity_post_reduction}.\n\t\t\t\tCheck that this is carried to the diffraction pattern.")
                
            
    def apply_intensity_reduction(self, angle, intensity_reduction):
        """
            Applies intensity reductions to the diffraction pattern.
            TODO: Change name from filter to something else. filter is a python function.
        """
        self.logger.add_log(f"\t\t\t\t\tApplying intensity reduction of {intensity_reduction} at angle {angle}.")
        filter = np.ones_like(self.fhkls[0][:,:,0] if not isinstance(self.fhkls[0], list) else self.fhkls[0][0][:,:,0])
        self.logger.add_log(f"\t\t\t\t\t\tInitial filter created with shape: {filter.shape}")
        if intensity_reduction > 1:
            self.logger.add_log("\n\n\nIntensity reduction must be less than 1!!!\nAssumig that you forgot to divide by 100\n\n")
            intensity_reduction /= 100
        #TODO: The funciton above allows for multiple different intensity reductions, but this is not implemented, and makes it much more difficult to compare the different results, so it probably should not be implemented unless one wants to compare it more realistically up towards a real sample.
        intensity_reduction_string = str(int(float(intensity_reduction)*100))
        if len(intensity_reduction_string) == 1:
            intensity_reduction_string = "0" + intensity_reduction_string
        self.description.add_description("intensity_reduction", intensity_reduction_string)
        self.description.add_description("intensity_reduction_angle", angle)

        angle_max = angle * np.pi / 180
        #TODO: Find reasonable value for angle deviation based on number of diffraction patterns in real samples etc.
        #TODO: Use this to calculate the number of pixels to reduce intensity in. And hence a suitable gaussian distribution around the main angle.
        self.int_red_dev = 7
        dev = self.int_red_dev * np.pi / 180

        #TODO: Fix this.
        std = 0.04
        angle_plus_deviation  = angle_max + dev
        angle_minus_deviation = angle_max - dev
        angle_plus_deviation   = angle_plus_deviation % np.pi
        angle_plus_deviation  -= (angle_plus_deviation//(np.pi/2)) * np.pi
        angle_minus_deviation  = angle_minus_deviation % np.pi
        angle_minus_deviation -= (angle_minus_deviation//(np.pi/2)) * np.pi
        invert = False
        if angle_plus_deviation < angle_minus_deviation:
            invert = True
            angle_plus_deviation += np.pi
        for ind_x, row in enumerate(filter):
            for ind_y, value in enumerate(row):
                angle = np.arctan2(ind_y - filter.shape[1]//2, ind_x - filter.shape[0]//2)
                if not invert:
                    if angle_minus_deviation < angle < angle_plus_deviation:
                        #Where does 100 come from?
                        filter[ ind_x,  ind_y] += 100*1/np.sqrt(2*np.pi*std**2)*np.exp(-(np.abs(angle-angle_max)%(np.pi/2))**2/(2*std**2))
                        filter[-ind_x, -ind_y] += 100*1/np.sqrt(2*np.pi*std**2)*np.exp(-(np.abs(angle-angle_max)%(np.pi/2))**2/(2*std**2))
                else:
                    angle += np.pi
                    if angle_minus_deviation < angle < angle_plus_deviation:
                        filter[ ind_x,  ind_y] += 100*1/np.sqrt(2*np.pi*std**2)*np.exp(-(np.abs(angle-angle_max)%(np.pi/2))**2/(2*std**2))
                        filter[-ind_x, -ind_y] += 100*1/np.sqrt(2*np.pi*std**2)*np.exp(-(np.abs(angle-angle_max)%(np.pi/2))**2/(2*std**2))
        filter = filter / np.max(filter)
        filter = filter*intensity_reduction
        self.logger.add_log(f"Saving intensity reduction filter with angle {angle} and intensity reduction {intensity_reduction} to file.")
        np.save(os.path.join(self.save_path, f"{self.filename_prefix}_filter.npy"), filter)
        self.logger.add_log(f"Saving plot of intensity reduction filter with angle {angle} and intensity reduction {intensity_reduction} to file.")
        plt.figure(figsize=(10,10))
        self.logger.add_log(f"\t\t\tFilter dtype: {filter.dtype}")
        plt.imshow(np.abs(1-filter), extent = (self.h.min(), self.h.max(), self.k.min(), self.k.max()))
        plt.colorbar()
        plt.savefig(os.path.join(self.save_path, f"{self.filename_prefix}_filter.jpg"))
        plt.close()
        #?#?#?#?#?#?#?#?#?OBSOBS: Changed this from 1 - filter to account for the fact that it is the intensity and not the scattering amplitude that is supposed to be redused by x percents.
        return np.sqrt(1 - filter)
        # for fhkl in self.fhkls:
        #     fhkl = np.multiply(fhkl.transpose(1,2,0), 1 - filter).transpose(2,0,1)
        # self.diffraction_pattern = np.multiply(self.diffraction_pattern.transpose(1,2,0), (1-filter)).transpose(2,0,1)
    
    def set_standard_error_parameters(self):
        """
            Function to set the standard error parameters.
            If beamstop is given, but no radius, the radius is set to the largest possible radius.
        """        
        # if self.beamstop and self.beamstop_radius == None:
        #     self.beamstop_radius = np.sqrt(self.nh**2 + self.nk**2 + self.nl**2)/2
        if self.cylindrical_reciprocal_space:
                self.logger.add_log("\tUsing cylindrical reciprocal space.")
                if self.cylindrical_reciprocal_space_radius == None:
                    self.logger.add_log(f"\tNo radius for the cylindrical reciprocal space was given. Using the largest reciprocal space dimension as diameter.")
                    self.logger.add_log(f"\t\tI.e., radius: {self.nh//2}")
                    self.cylindrical_reciprocal_space_radius = self.nh//2
        if self.beamstop:
            self.logger.add_log("\tUsing a beamstop.")
            if self.beamstop_radius == None:
                self.logger.add_log(f"\t\tNo radius for the beamstop was given. Using standard values of h_range/10 = {self.h_range/10}.")
                self.beamstop_radius = self.nh//2
        if self.int_red:
            self.logger.add_log("\tApplying intensity reductions.")
            # if not isinstance(self.int_red, np.ndarray):  
            #     if self.int_red_angle == None:
            #         self.logger.add_log(f"\t\tNo angle for the intensity reduction was given. Using standard value of 45 degrees.")
            #         self.int_red_angle = 45                

            # if not isinstance(self.int_red_pre, np.ndarray):
            #     if self.int_red_pre == None:
            #         self.logger.add_log(f"\t\tNo precentage for the intensity reduction was given. Using standard value of 10%.")
            #         self.int_red_pre = 10
            # self.logger.add_log("CAME HERE")
    def unpack_error_source_dict(self):
        """
            Saves the different parameters to member variables of the Reconstruction class.
            Parameter:
                beamstop
                beamstop_radius
                cylindrical_reciprocal_space
                cylindrical_reciprocal_space_radius
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
            self.beamstop_radius = self.nh * self.error_source_dict["beamstop_radius_factor"]
            self.logger.add_log(f"\t\tBeamstop radius: {self.beamstop_radius}")
        except KeyError:
            self.beamstop_radius = None
            self.logger.add_log("\t\tNo beamstop radius in error source dictionary.")
            self.logger.add_log("\t\tBeamstop radius: None")
        except Exception as e:
            self.logger.add_log(f"\t\tError: {e}")
            if self.beamstop:
                self.logger.add_log("\t\t\tBeamstop radius set to three pixels.")
                self.beamstop_radius = 3
                self.logger.add_log(f"\t\t\tBeamstop radius: {self.beamstop_radius}")
            else:
                self.beamstop_radius = None
                self.logger.add_log("\t\tBeamstop radius: None")
        try:
            self.cylindrical_reciprocal_space = self.error_source_dict["cylindrical_reciprocal_space"]
            self.logger.add_log(f"\t\tCylindrical reciprocal space: {self.cylindrical_reciprocal_space}")
        except KeyError:
            self.cylindrical_reciprocal_space = False
            self.logger.add_log("\t\tNo cylindrical reconstruction space in error source dictionary.")
            self.logger.add_log("\t\tCylindrical reciprocal space: False")
        try:   
            self.cylindrical_reciprocal_space_radius = self.error_source_dict["cylindrical_reciprocal_space_radius"]
            self.logger.add_log(f"\t\tCylindrical reciprocal space radius: {self.cylindrical_reciprocal_space_radius}")
        except KeyError:
            self.cylindrical_reciprocal_space_radius = None
            self.logger.add_log("\t\tNo cylindrical reciprocal space radius in error source dictionary.")
            self.logger.add_log("\t\tCylindrical reciprocal space radius: None")
        try:
            self.int_red = self.error_source_dict["intensity_reduction"]
            self.logger.add_log(f"\t\tIntensity reduction: {self.int_red}")
        except KeyError:
            self.int_red = False
            self.logger.add_log("\t\tNo intensity reduction in error source dictionary.")
            self.logger.add_log("\t\tIntensity reduction: False")
        try:
            self.int_red_angle = self.error_source_dict["intensity_reduction_angle"]
            self.logger.add_log(f"\t\tIntensity reduction angle: {self.int_red_angle}")
            #Should work, but gamma doesn't accept it...
            # assert(isinstance(self.int_red_angle, (int | float | np.ndarray)))
            # assert(isinstance(self.int_red_angle, int) or isinstance(self.int_red_angle, float) or isinstance(self.int_red_angle, np.ndarray)) 
        except KeyError:
            self.int_red_angle = None
            self.logger.add_log("\t\tNo intensity reduction angle in error source dictionary.")
            self.logger.add_log("\t\tIntensity reduction angle: None")
        try:
            self.int_red_pre = self.error_source_dict["intensity_reduction_percentage"]
            self.logger.add_log(f"\t\tIntensity reduction percentage: {self.int_red_pre}")
            #Should work, but gamma doesn't accept it...
            # assert(isinstance(self.int_red_angle, (int | float | np.ndarray)))
            # assert(isinstance(self.int_red_pre, int) or isinstance(self.int_red_pre, float) or isinstance(self.int_red_pre, np.ndarray))
        except KeyError:
            self.int_red_pre = None
            self.logger.add_log("\t\tNo intensity reduction percentage in error source dictionary.")
            self.logger.add_log("\t\tIntensity reduction percentage: None")

        self.logger.add_log("\tUnpacking done.")

    def create_reciprocal_space(self):
        """
            TODO: Implement a way to choose the reciprocal space based on the angles over which the real space object is viewed.
            TODO: Implement the error sources that only effect which portion of the reciprocal space which is used, i.e. beam stop and missing angles.

            Function to create the grid in reciprocal space.
            To be used when simulating the scattering.

        Args:
            nh (int, optional): Number of points in the h direction. Defaults to 4**4.
            nk (int, optional): Number of points in the k direction. Defaults to 4**4.
            nl (int, optional): Number of points in the l direction. Defaults to 4**4.
            h_range (float, optional): Range in the h direction. Defaults to 1.
            k_range (float, optional): Range in the h direction. Defaults to 1.
            l_range (float, optional): Range in the h direction. Defaults to 1.
            h_offset (int, optional): Offset in k-space. Defaults to 0, since we're working with SAXS.
            k_offset (int, optional): Offset in k-space. Defaults to 0, since we're working with SAXS.
            l_offset (int, optional): Offset in k-space. Defaults to 0, since we're working with SAXS.

        Returns:
            3 np.ndarrays: Arrays containing the h, k and l values. These are later put together to create the 3D reciprocal space grid.
        
        Notes:
            nh, nk and nl should be a multiplum(#?) of 4. 
            Ref. PyNX documentation -> Fhkl_thread (#? I think) 
        
        TODO: Figure out a suitable range. Should probably not use the entire first Brilloin zone since we're working with SAXS.
            I.e. one should probably not use 1 as range.
        """    
        self.logger.add_log("\tCreating reciprocal space grid with the following parameters:")
        self.logger.add_log(f"\t\tnh: {self.nh}, h_range: {self.h_range}, h_offset: {self.h_offset}")
        self.logger.add_log(f"\t\tnk: {self.nk}, k_range: {self.k_range}, k_offset: {self.k_offset}")
        self.logger.add_log(f"\t\tnl: {self.nl}, l_range: {self.l_range}, l_offset: {self.l_offset}")
        h = self.h_offset + np.linspace(-self.h_range/2, self.h_range/2, self.nh)[:, np.newaxis, np.newaxis]
        k = self.k_offset + np.linspace(-self.k_range/2, self.k_range/2, self.nh)[np.newaxis, :, np.newaxis]
        l = self.l_offset + np.linspace(-self.l_range/2, self.l_range/2, self.nh)[np.newaxis, np.newaxis, :]
        return h, k, l

    def find_relevant_files(self):
        """
            Function to find the relevant files in the object_path.
            The files should be .npy files.

        Returns:
            list: List containing the paths to the relevant files.
        """    
        relevant_files = []
        self.logger.add_log(f"\tFinding relevant object files by looping through the directory where the objects are saved ({self.object_path})")
        for file in os.listdir(self.object_path):
            if file.find(self.filename_prefix + "_") != -1 and os.path.basename(file).startswith("old") == False:
                relevant_files.append(os.path.join(self.object_path, file))
            if file.find(self.filename_prefix + "_") != -1 and os.path.basename(file).startswith("old") == True and self.test == True:
                relevant_files.append(os.path.join(self.object_path, file))
        return relevant_files
        
    def calculate_form_factor(self, x, y, z, density, gpu_name = "", language = ""):
        """ 
            Calculates form factor for a given object/part of object.
            TODO: Figure out if form factor is the correct term to use.

        Args:
            h (np.ndarray): Array containing the h values.
            k (np.ndarray): Array containing the k values.
            l (np.ndarray): Array containing the l values.
            x (np.ndarray): Array containing the x values.
            y (np.ndarray): Array containing the y values.
            z (np.ndarray): Array containing the z values.
            gpu_name (str, optional): Should be chosen automatically by pynx. Defaults to "".
            language (str, optional): Should be chosen automatically by pynx. Defaults to "".
            
        Returns:
            np.ndarray: Returns a three dimensional array containing the "diffraction pattern" (not diffraction pattern since it is the "unsquared" form factor). This should be squared to get the intensity.
                        This should happen outside this function to easier add together diffraction parts with different densities.

            TODO: Figure out what to set as standard gpu_name and language. Have only used "" before. Should work. Think it's able to fix it itself.
        """
        try:
            self.logger.add_log(f"\tCalculating form factor for {self.filename_prefix}.")
            fhkl, dt = Fhkl_thread(self.h, self.k, self.l, x, y, z, gpu_name=gpu_name, language=language)
            self.logger.add_log(f"\tForm factor calculation for {self.filename_prefix} took {dt} seconds.")
            #From the tutorial. Slightly modified (nx*ny*nz --> x.size) to account for the changes made to represent the objects. I.e. only keep the nonzero points.
            # if not self.cylindrical_reciprocal_space:
            #     print("fhkl: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s" %
            #     (self.h.size * self.k.size * self.l.size // 1000, x.size // 1000, x.size * self.h.size * self.k.size * self.l.size / dt / 1e9))
            # else:
            #     print("fhkl: %5d 10^3 reflections, %5d 10^3 atoms, speed=%7.3f 10^9 reflections.atoms/s" %
            #     (self.h.size // 1000, x.size // 1000, x.size * self.h.size / dt / 1e9))
            #?Not quite sure where or if this should be implemented, but I think it's needed.
            #TODO: Find backing for this.
            #TODO: Potentially change to return fhkl*density and do the appending to self.fhkls outside this function.
            self.logger.add_log(f"\tAppending form factor for {self.filename_prefix} to self.fhkls.")
            self.logger.add_log(f"\t\tfhkl shape: {fhkl.shape}")
            self.logger.add_log(f"\tForm factor for {self.filename_prefix} appended to self.fhkls.")
            return fhkl*density
            
        except Exception as e:
            print("Error while calculating form factor. Check if the object is too large.")
            self.logger.add_log(f"\tError {e} while calculating form factor for {self.filename_prefix}. Check if the object is too large.")
            # sys.exit(1)

    def apply_beamstop(self):
        """ 
            Applies a beamstop to the diffraction pattern.
        """
        self.logger.add_log(f"\tApplying spherical beamstop to {self.filename_prefix}.")
        mask = rg.sphere(self.nh, radius = self.beamstop_radius)
        self.diffraction_pattern[mask] = 0
             
    def apply_cylindrical_reciprocal_space(self):
        self.logger.add_log(f"\tCreating cylindrical reciprocal space with radius {self.cylindrical_reciprocal_space_radius}.")
        mask = ~rg.cylinder(shape = len(self.diffraction_pattern), height = len(self.diffraction_pattern), radius = self.cylindrical_reciprocal_space_radius)
        self.diffraction_pattern[mask] = 0
        #  *= np.array(rg.cylinder(shape = len(self.diffraction_pattern), height = len(self.diffraction_pattern), radius= self.cylindrical_reciprocal_space_radius), dtype = int)

    def save_mask(self):
        #TODO: Implement this.
        self.logger.add_log(f"\t\tSaving mask for {self.filename_prefix}.")
        if self.beamstop and self.cylindrical_reciprocal_space:
            mask = ~rg.sphere(self.nh, radius = self.beamstop_radius) & rg.cylinder(shape = len(self.diffraction_pattern), height = len(self.diffraction_pattern), radius = self.cylindrical_reciprocal_space_radius)
        elif self.beamstop:
            mask = ~rg.sphere(self.nh, radius = self.beamstop_radius)
        elif self.cylindrical_reciprocal_space:
            mask = rg.cylinder(shape = len(self.diffraction_pattern), height = len(self.diffraction_pattern), radius= self.cylindrical_reciprocal_space_radius)
        else:
            self.logger("No mask to save. Saving zeros.")
            mask = np.zeros_like(self.diffraction_pattern)
        np.save(os.path.join(self.save_path, f"mask_{self.filename_prefix}.npy"), mask)

#TODO: Merge the two functions below into one.
    def load_coordinates_and_density(self):
        """
            TODO: Implement a way to include the possibility of having multiple densities per object. As of now, it's only possible to have one or two densities per object.

            Function to load the coordinates and density of the object.
            This function is used when the object is stored as a .npy file.

        Returns:
            TODO: Low Prio. Describe this in a proper way.
            np.ndarray: Returns an array with the calculated value of the form factor for different possitions in reciprocal space.
        """    
        self.logger.add_log(f"\tLoading coordinates and density for {self.filename_prefix}.")
        self.logger.add_log("\tLooping through all files in the directory with the correct filename prefix.")
        self.logger.add_log(f"\tNumber of relevant files: {len(self.relevant_files)}")
        
        for ind, file in enumerate(self.relevant_files):
            filename = os.path.basename(file)
            self.logger.add_log(f"\tLoading coordinates and density from {filename}")
            coordinates_and_density = np.load(file)
            self.logger.add_log(f"\tCoordinates and density loaded from {filename}")

            fhkl_object = []

            self.logger.add_log(f"\tThe file constists of {len(coordinates_and_density)//4} objects. (Each object is counted as a part with different density")
            print(len(coordinates_and_density))
            if len(coordinates_and_density) == 4:
                self.logger.add_log(f"\tCalculating form factor for {filename}. Object has one density.")
                x = coordinates_and_density["x"]
                y = coordinates_and_density["y"]
                z = coordinates_and_density["z"]
                d = coordinates_and_density["d"]
                fhkl_object.append(self.calculate_form_factor(x, y, z, d))

            elif len(coordinates_and_density) == 8:
                self.logger.add_log(f"\tCalculating form factor for {filename}. Object has two densities.")
                xd = coordinates_and_density["xd"]
                yd = coordinates_and_density["yd"]
                zd = coordinates_and_density["zd"]
                
                xl = coordinates_and_density["xl"]
                yl = coordinates_and_density["yl"]
                zl = coordinates_and_density["zl"]
                
                dd = coordinates_and_density["dd"]
                dl = coordinates_and_density["dl"]
                fhkl_object.append(self.calculate_form_factor(xd, yd, zd, dd))
                fhkl_object.append(self.calculate_form_factor(xl, yl, zl, dl))
            else:
                print("Something went wrong while loading the coordinates and density. Check the file.")
                print("Error: The number of coordinates and densities is not supported.")
                self.logger.add_log(f"\tSomething went wrong while loading the coordinates and density for {filename}. Check the file.")
                self.logger.add_log(f"\tError: The number of coordinates and densities is not supported.")
                # sys.exit(1)
            self.fhkls.append(fhkl_object)

    def calculate_intensity(self):
        """
            Function to calculate the intensity of the diffraction pattern. I.e. the diffraction pattern...
            Have to add together the form factors of the different objects before squaring to get the intesity pattern.

        Args:
            objects (dict): Dictionary containing the objects to be simulated. Either a single object or multiple objects.

        Returns:
            _type_: _description_
        """
        self.logger.add_log(f"\tCalculating intensity for {self.filename_prefix}.")
        #TODO: Figure out which dtype to use. fhkl is automatically set to complex128. Should I use that?
        
        self.diffraction_pattern = np.zeros((self.h.size, self.k.size, self.l.size), dtype = np.complex128)
        
        self.logger.add_log("Lopping through all form factors, adding them together, before squaring to get the intensity.")
        self.logger.add_log(f"\tNumber of form factors/objects: {len(self.fhkls)}")
        for fhkl in self.fhkls:
            if isinstance(fhkl, list):
                self.logger.add_log(f"\t\tForm factor is a list with {len(fhkl)} elements. Meaning multiple densities.")
                for fhkl_object in fhkl:
                    self.diffraction_pattern += fhkl_object
            else:
                self.logger.add_log("\tForm factor is not a list.")
                self.diffraction_pattern += fhkl
            

        #TODO: Find out if this is the correct way to calculate the intensity. I.e. is it the correct way to add the form factors together.
        #TODO: Find out if fftshift is necessary or even correct. The tutorial uses it, but I'm not sure if it's necessary.
        self.diffraction_pattern = np.abs(self.diffraction_pattern)**2
        #Normalize the diffraction pattern such that all the intensities, regardless of reduction applied, add up to 1e10, as suggested by Favre.
        self.logger.add_log(f"\tIntensity calculated for {self.filename_prefix} before normalization. Intensity: {np.sum(self.diffraction_pattern)}")
        self.diffraction_pattern = self.diffraction_pattern / np.sum(self.diffraction_pattern) 
        self.diffraction_pattern = self.diffraction_pattern * 1e10
        self.logger.add_log(f"\tIntensity calculated for {self.filename_prefix} after normalization. Intensity: {np.sum(self.diffraction_pattern)}")
        self.logger.add_log(f"\n\tIntensity calculated for {self.filename_prefix}. Intensity: {np.sum(self.diffraction_pattern)}\n")

    def save_diffraction_pattern(self):
        """
            Function to save the diffraction pattern to a file.
        """
        if self.test:
            np.save(os.path.join(self.save_path, self.filename_prefix + "_test.npy"), self.diffraction_pattern)#, dif_pat = self.diffraction_pattern)
        else:
            np.save(os.path.join(self.save_path, self.filename_prefix + ".npy"), self.diffraction_pattern)
    
    def change_filename_prefix(self):
        """
            Function to change the filename prefix of the object to mark it as used.

        Args:
            None
        """
        for file in self.relevant_files:
            filename = os.path.basename(file)
            try:
                self.logger.add_log(f"\tChanging name of {filename} to old_{filename}")
                os.rename(file, os.path.join(self.object_path, f"old_{filename}"))
                self.logger.add_log(f"\tName of {filename} changed to old_{filename}")
            except:
                self.logger.add_log(f"\t\tFailed to change name of {filename} to old_{filename}. Probably because it already exists.")
                self.logger.add_log(f"\t\tTrying to overwrite the file.")
                try: 
                    os.remove(os.path.join(self.object_path, f"old_{filename}"))
                    os.rename(file, os.path.join(self.object_path, f"old_{filename}"))
                    self.logger.add_log("\t\tFile successfully overwritten.")
                except:
                    self.logger.add_log(f"\tFailed to overwrite the file.")
                
# Currently not used. Can be used when the code is ready to be used in console mode.

class ErrorSources:
    """Class to apply error sources to the diffraction data, e.g. missing intensities.
    """
    pass


def main():
    logger = Logger("test_diffraction")
    DiffractionExperiment(filename_prefix="3000000_0001", logger=logger, error_source_dict=error_source_dict_list[1], test = True)
    logger.save_log()

if __name__ == "__main__":
    main()

"""Legacy code. Not used anymore, but kept for reference."""



"""
def main():
    
        # Have to loop over a set of error sources and their combination to see how large the impact is.

        # missing_angles: How many "projection degrees" are not measured. Remove them symmetrically.
        # beam_stop: Remove a cube/cyllinder (what makes the most sense#?) of the innermost diffraction data.
        # sub_res_diffraction: Remove data from main angle, with a width included (how many other angles it is supposed to impact) and how severly it is supposed to impact them, i.e. the intensity_reduction.
    
        # TODO: Find a way to pass information to the main function when used in console mode.
    

    # TODO: Should also be able to pass the error sources as an argument. If err_sou == None: use standard below.
    error_sources = {"missing_angles": np.array([0,15,30]),
                     "beam_stop": np.array([0,10,20]),
                     "sub_res_diffraction": {"main_angle": np.array([20,30,40]), "width": np.array([3,5,7]), "intensity_reduction": np.array([0.01,0.1,1,10])}}
    
    
    #? This should always be the same.
    obj = create_star()

    # TODO: This should be done in a much cleaner way
    try:
        for missing_angles in error_sources["missing_angles"]:
            try:
                for beam_stop in error_sources["beam_stop"]:               
                    # TODO: Find out how to change this according to the missing_angles and beam_stop arguments.
                    h, k, l = create_reciprocal_space()
                    dif_pat = create_diffraction_pattern(h, k, l, obj[0], obj[1], obj[2])  
            except:
                # TODO: Find out how to change this according to the missing_angles argument.
                h, k, l = create_reciprocal_space()
                dif_pat = create_diffraction_pattern(h, k, l, obj[0], obj[1], obj[2])
    except:
        try:
            for beam_stop in error_sources["beam_stop"]:               
                # TODO: Find out how to change this according to the beam_stop argument.
                h, k, l = create_reciprocal_space()
                dif_pat = create_diffraction_pattern(h, k, l, obj[0], obj[1], obj[2])
        except:
            h, k, l = create_reciprocal_space()
            dif_pat = create_diffraction_pattern(h, k, l, obj[0], obj[1], obj[2])
    try:
        sub_res_dif_dict = error_sources["sub_res_diffraction"]
        # TODO: Remove different parts of the diffraction pattern accordingly
    except:
        print("This has to be done within the other try-except structure as it is setup now. Not feasable in the long run.")

    try:
        missing_angles, beam_stop, sub_res_dif = error_sources["missing_angles"], error_sources["beam_stop"], error_sources["sub_res_diffraction"]
        assert(isinstance(missing_angles, (np.ndarray, int)))
        assert(isinstance(beam_stop, (np.ndarray, int)))
        assert(isinstance(sub_res_dif, (dict, int)))
    except:
        print("Missing mandatory information on the error sources. Either 'missing_angles', 'beam_stop' or 'sub_resolution_diffraction'. These can be set to 0")
def create_cylindrical_reciprocal_space(self, r):
    
        Function to create a cylindrical k-space.

    Args:
        r (float): Radius of the cylinder.

    Returns:
        np.ndarray: Array containing the hkl values.
        
    self.logger.add_log(f"\tCreating cylindrical reciprocal space with radius {r}.")
    #nl gives z-values. Should be the height of the cylinder, since it is constant throughout the experiment. I think.
    print(f"Shapes: h: {self.h.shape}, k: {self.k.shape}, l: {self.l.shape}")
    self.h = self.h * rg.cylinder(self.nh, height=self.nl, radius=r)
    self.k = self.k * rg.cylinder(self.nk, height=self.nl, radius=r)
    self.l = self.l * rg.cylinder(self.nl, height=self.nl, radius=r)
    print(f"Shapes: h: {self.h.shape}, k: {self.k.shape}, l: {self.l.shape}")
    self.h = self.h.ravel()
    self.k = self.k.ravel()
    self.l = self.l.ravel()
    self.h = self.h.astype(np.float32)
    self.k = self.k.astype(np.float32)
    self.l = self.l.astype(np.float32)
    print(f"Shapes: h: {self.h.shape}, k: {self.k.shape}, l: {self.l.shape}")
    self.cylindrical_reciprocal_space = True

        def calculate_gaussian_filter(self):
        
            Function to calculate the gaussian filter.
            The gaussian filter is calculated by first creating a meshgrid of the reciprocal space.
            Then, the gaussian filter is calculated by using the scipy.ndimage.gaussian_filter function.
            The standard deviation of the gaussian filter is calculated by using the self.calculate_gaussian_filter_standard_deviation() function.
            The gaussian filter is then normalized by dividing by its maximum value.
        
        self.logger.add_log("Calculating the gaussian filter.")
        h, k, l = np.meshgrid(self.h, self.k, self.l, indexing = "ij")
        gaussian_filter = gaussian_filter(np.ones_like(h), sigma = self.calculate_gaussian_filter_standard_deviation())
        gaussian_filter /= np.max(gaussian_filter)
        self.logger.add_log("Done calculating the gaussian filter.")
        return gaussian_filter
    
    def calculate_gaussian_filter_standard_deviation(self):
        
            Function to calculate the standard deviation of the gaussian filter.
            The standard deviation is calculated by using the self.calculate_gaussian_filter_standard_deviation() function.
            The standard deviation is then normalized by dividing by the reciprocal space range.
        
        self.logger.add_log("Calculating the standard deviation of the gaussian filter.")
        standard_deviation = self.calculate_gaussian_filter_standard_deviation()
        standard_deviation /= self.h_range
        self.logger.add_log("Done calculating the standard deviation of the gaussian filter.")
        return standard_deviation
    
    def calculate_gaussian_filter_standard_deviation(self):
        
            Function to calculate the standard deviation of the gaussian filter.
            The standard deviation is calculated by using the self.calculate_gaussian_filter_standard_deviation() function.
            The standard deviation is then normalized by dividing by the reciprocal space range.
        
        self.logger.add_log("Calculating the standard deviation of the gaussian filter.")
        standard_deviation = self.calculate_gaussian_filter_standard_deviation()
        standard_deviation /= self.h_range
        self.logger.add_log("Done calculating the standard deviation of the gaussian filter.")
        return standard_deviation

    def apply_intensity_reductions(self):
        '''
            Function to apply intensity reductions to the diffraction pattern.
            This is done by multiplying the diffraction pattern with a 2D matrix in the k_x-k_y-plane.
        '''
        self.logger.add_log("Applying intensity reductions to the diffraction pattern.")
        # Size of intensity reduction matrix
        # Multiplying the size of the diffraction pattern with sqrt(2) (+ some extra) to make sure that the matrix won't get any non-zero values when rotating the matrix.
        N = int(self.diffraction_pattern.shape[0]*np.sqrt(2)) + 3
        intensity_reduction_matrix = np.ones((N,N), dtype=np.float32)
        
        # Creating line in the middle of the matrix, which will be rotated, corresponding to a projection where the intensity has been reduced.
        std = 0.01
        # Only mean = 0 makes sense. The line have to go through zero.
        mean = 0
        reduction_distribution = np.linspace(-1,1,N)
        # Using a gaussian to create the line. 
        # TODO: Choose reasonable values for std and mean.
        # TODO: Find out if gaussian even makes sense here.
        reduction_distribution = 1/np.sqrt(2*np.pi*std**2)*np.exp(-(reduction_distribution-mean)**2/(2*std**2))

        reduction_distribution = reduction_distribution/reduction_distribution.max() 
        # Percentage reduction:
        # TODO: Find easy formula to do this.
        reduction_distribution = reduction_distribution * self.int_red_pre / 100
        w = np.ones((N,N), dtype=np.float32)
        w = w*reduction_distribution
        intensity_reduction_matrix = intensity_reduction_matrix - w
        intensity_reduction_matrix = rotate(intensity_reduction_matrix, angle=self.int_red_angle, reshape = False, order = 3)
        intensity_reduction_matrix[intensity_reduction_matrix>1] = 1
        # Method to make sure that the matrix is symmetric around center and correct size.
        # TODO: Explain each step
        x, y = intensity_reduction_matrix.shape
        xo = x%self.diffraction_pattern.shape[0]
        yo = y%self.diffraction_pattern.shape[1]
        xs = xo//2
        ys = yo//2
        xe = xs + self.diffraction_pattern.shape[0]
        ye = ys + self.diffraction_pattern.shape[1]
        sl = slice(xs, xe), slice(ys, ye)  
        #Skummelt med reshape av diffraksjonsmønsteret. Må finne ut noe annet. Gange over andre akser eller noe sånt.
        #Now, the diffraction pattern should be reduced by the wanted amount.
        #The line will be exactly the same for all z-values.
        #When plotting diffraction_pattern[:,:,z], the diffraction pattern is viewed from above, and a line should appear.
        #When plotting diffraction_pattern[x,:,:] or diffraction_pattern[:,y,:], the diffraction pattern is viewed from the side, and a line should appear (given that it is at an angle and not in the x or y direction).
        self.diffraction_pattern = np.multiply(self.diffraction_pattern.transpose(2,1,0),intensity_reduction_matrix[sl]).transpose(2,1,0)

        # self.diffraction_pattern *= self.calculate_gaussian_filter()
        self.logger.add_log("Done applying intensity reductions to the diffraction pattern.")
    

"""