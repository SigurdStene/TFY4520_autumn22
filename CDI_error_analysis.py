import os
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from CDI_logger import Logger
from CDI_description import Description
from CDI_plotting import *
from CDI_dictionaries import grid_size

#TODO: Should the reconstructed objects be extrapolated to the same size as the original objects?
#TODO: Create plots. Distribution around the mean, mean, standard deviation, etc.

#TODO: Handle the case where the reconstructed object is rotated compared to the initial object. Allowed for symmerty reasons in the reconstruction.

#TODO: NB. Because of the rotation of the matrices, they should not have approximately the same sizes, as this might lead to some unwanted rotations.

#TODO: Save all the objects (histogram values (have to specify the bins), mean, standard deviation, etc.) to a file and load them later to compare them together in one plot.
#Have to be done outside the loop which is already in the main function, so that one can compare the results from the different initial conditions. Look to test file to see how they are to be compared. Should also include some line plots in a reasonable way.


debug = True


plt.rcParams.update({'font.size': 28})
# plt.rcParams.update({'figure.autolayout': True})


class ErrorAnalysis:
    """
    Class to analyse the errors of the reconstructed objects.
    Have to load the objects from the cxi-files and compare them to the initial objects.
    The reconstructed and initial objects might have different sizes and grid sizes, so the nonzero part of the objects have to be cut out, before they are placed on the same grid.
    Have to find the deviation in intensity of the reconstructed object from the initial. The maximum intensity should be one.
        Could divide by the maximum value to normalize it, but this might place a too large weight on a potentially deviating value.
    """
    def __init__(self, group, filename_prefix, logger : Logger, description: Description, plot_gifs = False):
        
        #TODO: Describe group better. Just a collection of measurements that are to be analyzed together, i.e. stemming from the same run.
        self.group = group
        self.filename_prefix = filename_prefix
        self.logger = logger
        self.description = description
        self.plot_gifs = plot_gifs

        self.intensity_reduction_value = self.description.get_description("intensity_reduction")
        # if self.intensity_reduction_value != "none":
        #     #To get it back into percentage.
        #     self.intensity_reduction_value = str(int(self.intensity_reduction_value)*100))

        #Function to create directories that will be needed to save the error analysis data.
        self.create_directories()
        
        #TODO: Find out if it should be called something like self.save_path_folder, since it's just reffering to the directory.
        #TODO: When searching for the files. Use np.find to find the files with the correct prefix, and check that the length of the filename is equal to the filename prefix plus the file ending. At least for the diffraction data and the reconstructed objects.
        # self.save_path = os.path.join(self.current_path, "error_analysis", self.filename_prefix)   
        
        self.initial_object_path = os.path.join(self.current_path, "objects")
        self.reconstructed_object_path = os.path.join(self.current_path, "reconstructions")


        #TODO: Have to save the initial object(s) in a way that is easy to load.
        self.relevant_object_files = self.find_relevant_object_files()
        self.initial_object = self.find_nonzero(self.construct_initial_object())
        self.initial_object = self.find_correct_rotation(self.initial_object)
        self.logger.add_log(f"Shape of initial object after rotation: {self.initial_object.shape}")



        # self.initial_object_cropped = self.find_nonzero(self.initial_object)
        # self.initial_object_filename = os.path.join(self.initial_object_path, "old_", self.filename_prefix + ".cxi")
        self.relevant_reconstruction_files = self.find_relevant_reconstruction_files()
        # self.reconstructed_object_filename = os.path.join(self.reconstructed_object_path, self.filename_prefix + ".cxi")
        #Only to collect the density values of the reconstructed objects. Not interested in the placement etc.
        self.reconstructed_raw_object_mean = np.array([])
        self.reconstructed_object_mean_norm = None
        self.reconstructed_object_mean = None
        for DB, file in enumerate(self.relevant_reconstruction_files):
            #Find index manually in case the files are stored in an unordered way.
            
            #If rotated (cannot look at shape) try to flip and see if the deviation gets lower.
            #See how trhe shape is chsangedwith dlip
            
            basename = os.path.basename(file)
            print(f"Error analysis: {basename}")
            ind_stop = basename.find(".cxi")
            ind_len = 0
            for i in basename[ind_stop-1::-1]:
                if i == "_":
                    break
                else:
                    ind_len += 1
            ind_start = ind_stop - ind_len
            index = basename[ind_start:ind_stop]
            self.logger.add_log(f"Loading reconstructed object with index {index}")
            self.recon_index = index

            
            self.logger.add_log(f"Starting error analysis for object {index}.")            
            #TODO: Figure out what to actually collect from this file. Many keys to look through. Ref. test file.
            self.reconstructed_object = self.find_nonzero(np.asarray(h5py.File(file, 'r')["entry_1"]["image_1"]["data"]))
            self.reconstructed_object = self.find_correct_rotation(self.reconstructed_object)
            self.reconstructed_raw_object_mean = np.append(self.reconstructed_raw_object_mean, self.reconstructed_object.flatten())
            # self.save_raw_data()
            try:
                if self.reconstructed_object_mean == None:
                    self.reconstructed_object_mean = self.reconstructed_object
            except:
                #Have to check that the sizes are the same.
                self.add_reconstructed_object()
            self.reconstructed_object = self.normalize_object(self.reconstructed_object)
            #Added ugly fix to make sure that the flip check can be done even when it doesn't have the same shape as the initial object. The two functions below should probably be combined, but then the possible combinations beome large.
            self.reconstructed_object = self.flip_if_inverted()
            self.logger.add_log("Starting size check.")
            #Checks the sizes of the initial and reconstructed objects.
            #Potentially resizes the objects to the same size.
            self.initial_object, self.reconstructed_object = self.check_sizes()
            self.logger.add_log("Size check done.")

            # self.logger.add_log("Starting deviation check.")
            
            # self.deviation = self.find_deviation(self.reconstructed_object)
        
            # self.save_normalized_data()    
        
            self.logger.add_log(f"Error analysis finished for index {index}.")
            try:
                if self.reconstructed_object_mean_norm == None:
                    self.reconstructed_object_mean_norm = self.reconstructed_object
            except:
                #Have to check that the sizes are the same.
                self.add_reconstructed_object_norm()

        #Terrible way to do this again, but will ensure that all the files have the same shape.
        self.reconstructed_raw_object_mean = np.array([])
        self.reconstructed_object_mean_norm = np.zeros_like(self.reconstructed_object_mean_norm)
        self.reconstructed_object_mean = np.zeros_like(self.reconstructed_object_mean)
        for DB, file in enumerate(self.relevant_reconstruction_files):
            #Find index manually in case the files are stored in an unordered way.
            
            #If rotated (cannot look at shape) try to flip and see if the deviation gets lower.
            #See how trhe shape is chsangedwith dlip
            
            basename = os.path.basename(file)
            print(f"Error analysis: {basename}")
            ind_stop = basename.find(".cxi")
            ind_len = 0
            for i in basename[ind_stop-1::-1]:
                if i == "_":
                    break
                else:
                    ind_len += 1
            ind_start = ind_stop - ind_len
            index = basename[ind_start:ind_stop]
            self.logger.add_log(f"Loading reconstructed object with index {index}")
            self.recon_index = index

            
            self.logger.add_log(f"Starting error analysis for object {index}.")            
            #TODO: Figure out what to actually collect from this file. Many keys to look through. Ref. test file.
            self.reconstructed_object = self.find_nonzero(np.asarray(h5py.File(file, 'r')["entry_1"]["image_1"]["data"]))
            self.reconstructed_object = self.find_correct_rotation(self.reconstructed_object)
            self.reconstructed_raw_object_mean = np.append(self.reconstructed_raw_object_mean, self.reconstructed_object[self.reconstructed_object.nonzero()])
            #Added ugly fix to make sure that the flip check can be done even when it doesn't have the same shape as the initial object. The two functions below should probably be combined, but then the possible combinations beome large.
            self.reconstructed_object = self.flip_if_inverted()
            try:
                if self.reconstructed_object_mean == None:
                    self.reconstructed_object_mean = self.reconstructed_object
            except:
                #Have to check that the sizes are the same.
                self.add_reconstructed_object()
            self.save_raw_data()
            
            self.reconstructed_object = self.normalize_object(self.reconstructed_object)
            # #Added ugly fix to make sure that the flip check can be done even when it doesn't have the same shape as the initial object. The two functions below should probably be combined, but then the possible combinations beome large.
            self.reconstructed_object = self.flip_if_inverted()
            self.logger.add_log("Starting size check.")
            #Checks the sizes of the initial and reconstructed objects.
            #Potentially resizes the objects to the same size.
            self.initial_object, self.reconstructed_object = self.check_sizes()
            self.logger.add_log("Size check done.")

            self.logger.add_log("Starting deviation check.")
            
            self.deviation = self.find_deviation(self.reconstructed_object)
        
            self.save_normalized_data()    
        
            self.logger.add_log(f"Error analysis finished for index {index}.")
            try:
                if self.reconstructed_object_mean_norm == None:
                    self.reconstructed_object_mean_norm = self.reconstructed_object
            except:
                #Have to check that the sizes are the same.
                self.add_reconstructed_object_norm()
        
        self.logger.add_log("Mean_shape: " + str(self.reconstructed_object_mean.shape))
        self.reconstructed_object = self.reconstructed_object_mean / len(self.relevant_reconstruction_files)
        self.logger.add_log("Mean_shape: " + str(self.reconstructed_object.shape))
        self.recon_index = "mean"
        self.initial_object, self.reconstructed_object = self.check_sizes()
        self.save_raw_data()
        self.logger.add_log(f"Starting error analysis for object {self.recon_index}.")
        self.deviation = self.find_deviation(self.reconstructed_object)
        

        # self.plot_histogram(self.deviation)
        # self.plot_histogram(self.deviation_abs, filename_extra="abs")
        # # self.plot_histogram(self.initial_object, filename_extra="initial")
        # self.plot_histogram(np.abs(self.reconstructed_object), filename_extra="reconstructed")
        # self.plot_histogram(np.abs(self.reconstructed_object)/np.median(np.abs(self.reconstructed_object[self.reconstructed_object.nonzero()])), filename_extra="reconstructed_norm")
        # self.deviation = self.find_deviation()
        # self.error = self.find_error()
        self.save_normalized_mean()
        
        self.save_raw_mean()
        # np.save(os.path.join(self.save_path, f"{self.filename_prefix}_reconstructed_{self.recon_index}.npy"), np.abs(self.reconstructed_object))
        # np.save(os.path.join(self.save_path, f"{self.filename_prefix}_deviation.npy"), self.deviation)
        # Save the initial object
        self.save_initial_data()
        self.logger.add_log(f"Error analysis finished for index {self.recon_index}.")

    def save_raw_data(self):
        self.logger.add_log(f"Saving data for index: {self.recon_index}.")
        np.save(os.path.join(self.save_path, "reconstructed_objects", self.description.get_description("intensity_reduction"), f"{self.recon_index}.npy"), np.abs(self.reconstructed_object))
        if self.plot_gifs:
            self.plot_gif_imshow(data = np.abs(self.reconstructed_object), folder = os.path.join(self.save_path, "gifs_reconstructed", self.description.get_description("intensity_reduction")), filename = f"{self.recon_index}")

    def save_raw_mean(self):
        self.logger.add_log(f"Saving data for index: {self.recon_index}.")
        np.save(os.path.join(self.save_path, "reconstructed_objects", self.description.get_description("intensity_reduction"), f"mean_flattened.npy"), np.abs(self.reconstructed_raw_object_mean))

    def save_normalized_mean(self):
        self.logger.add_log(f"Saving data for index: {self.recon_index}.")
        np.save(os.path.join(self.save_path, "reconstructed_objects_norm", self.description.get_description("intensity_reduction"), f"mean.npy"), np.abs(self.reconstructed_object_mean_norm))
        

    def save_normalized_data(self):
        self.logger.add_log(f"Saving data for index: {self.recon_index}.")
        np.save(os.path.join(self.save_path, "reconstructed_objects_norm", self.description.get_description("intensity_reduction"), f"{self.recon_index}.npy"), np.abs(self.reconstructed_object))
        np.save(os.path.join(self.save_path, "deviations", self.description.get_description("intensity_reduction"), f"{self.recon_index}.npy"), self.deviation)
        np.save(os.path.join(self.save_path, "deviations_abs", self.description.get_description("intensity_reduction"), f"{self.recon_index}.npy"), np.abs(self.deviation))
        
        # Save the initial object
        
        if self.plot_gifs:
            self.plot_gif_imshow(data = self.deviation, folder = os.path.join(self.save_path, "gifs_deviation", self.description.get_description("intensity_reduction")), filename = f"{self.recon_index}")
            self.plot_gif_imshow(data = self.deviation_abs, folder = os.path.join(self.save_path, "gifs_deviation_abs", self.description.get_description("intensity_reduction")), filename = f"{self.recon_index}")
            self.plot_gif_imshow(data = np.abs(self.reconstructed_object), folder = os.path.join(self.save_path, "gifs_reconstructed_norm", self.description.get_description("intensity_reduction")), filename = f"{self.recon_index}")
        self.logger.add_log(f"Data saved for index: {self.recon_index}")

    def save_initial_data(self):
        """
        This function should be called after all the reconstructed objects have been loaded, so that is has the correct shape as compared to the mean of the reconstructions.
        """
        self.logger.add_log(f"Saving initial data.")
        np.save(os.path.join(self.save_path, "initial_object", "initial.npy"), self.initial_object)
        if self.plot_gifs:
            self.plot_gif_imshow(data = self.initial_object, folder = os.path.join(self.save_path, "gifs_initial"), filename = "initial")
        self.logger.add_log(f"Initial data saved.")
       
    def create_directories(self):
        """
        Create the directories for the error analysis.
        """
        self.current_path = os.getcwd()
        if not os.path.isdir(os.path.join(self.current_path, "error_analysis")):
            os.mkdir(os.path.join(self.current_path, "error_analysis"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.current_path, "error_analysis", self.group)):
            os.mkdir(os.path.join(self.current_path, "error_analysis", self.group))
            time.sleep(0.1)
        self.save_path = os.path.join(self.current_path, "error_analysis", self.group)
        if not os.path.isdir(os.path.join(self.save_path, "initial_object")):
            os.mkdir(os.path.join(self.save_path, "initial_object"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "reconstructed_objects")):
            os.mkdir(os.path.join(self.save_path, "reconstructed_objects"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "reconstructed_objects_norm")):
            os.mkdir(os.path.join(self.save_path, "reconstructed_objects_norm"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "deviations")):
            os.mkdir(os.path.join(self.save_path, "deviations"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "deviations_abs")):
            os.mkdir(os.path.join(self.save_path, "deviations_abs"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "gifs_reconstructed")):
            os.mkdir(os.path.join(self.save_path, "gifs_reconstructed"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "gifs_reconstructed_norm")):
            os.mkdir(os.path.join(self.save_path, "gifs_reconstructed_norm"))
            time.sleep(0.1)
        
        if not os.path.isdir(os.path.join(self.save_path, "gifs_initial")):
            os.mkdir(os.path.join(self.save_path, "gifs_initial"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "gifs_deviation")):
            os.mkdir(os.path.join(self.save_path, "gifs_deviation"))
            time.sleep(0.1)
        if not os.path.isdir(os.path.join(self.save_path, "gifs_deviation_abs")):
            os.mkdir(os.path.join(self.save_path, "gifs_deviation_abs"))
            time.sleep(0.1)
        self.create_sub_directories()

    def create_sub_directories(self):
        """
        Loop through the directories in self.save_path and create subdirectories correpesponding to each intensity reduction value for each. Also create a directory for no intensity reduction.
        Would be nice, but this is only run once per reduction value, so makes no sense. Might be kept to the other file though.
        Or can be kept, just change meaning. Have to create the same subdirectory in many folders. 
        """
        for directory in os.listdir(self.save_path):
            if not "initial" in directory:
                if not os.path.isdir(os.path.join(self.save_path, directory, self.description.get_description("intensity_reduction"))):
                    os.mkdir(os.path.join(self.save_path, directory, self.description.get_description("intensity_reduction")))
                    time.sleep(0.1)

   

    def add_reconstructed_object(self):
        """
        Adds the reconstructed object to the mean of the reconstructed objects.
        Have to check that the sizes are the same.
        If not, have to resize the objects to the same size.
        """
        if self.reconstructed_object.shape != self.reconstructed_object_mean.shape:
            self.logger.add_log("Reconstructed objects have different sizes. Resizing.")
            self.reconstructed_object, self.reconstructed_object_mean = self.resize_objects(self.reconstructed_object, self.reconstructed_object_mean)
            self.logger.add_log("Reconstructed objects resized.")
        
        self.reconstructed_object_mean += self.reconstructed_object
    
    def add_reconstructed_object_norm(self):
        """
        Copied function above for quick fix. Will have to be accounted for later.
        Terrible way to do it.
        Adds the reconstructed object to the mean of the reconstructed objects.
        Have to check that the sizes are the same.
        If not, have to resize the objects to the same size.
        """
        if self.reconstructed_object.shape != self.reconstructed_object_mean_norm.shape:
            self.logger.add_log("Reconstructed objects have different sizes. Resizing.")
            self.reconstructed_object, self.reconstructed_object_mean_norm = self.resize_objects(self.reconstructed_object, self.reconstructed_object_mean_norm)
            self.logger.add_log("Reconstructed objects resized.")
        
        self.reconstructed_object_mean_norm += self.reconstructed_object
    

    def normalize_object(self, object):
        """
        Normalizes the object based on the most common value in a histogram of the object with 100 bins.
        Have to make sure that the last peak is located around 1. Might end up with that the less dense part is bigger. Have to multiply with that value in this case, but might be difficult to extract.
        Also takes the abolute value of the object to get rid of potential imaginary values.
        """
        self.logger.add_log("Starting normalization.")
        hist, bin_edges = np.histogram(np.abs(object[object.nonzero()]), bins=200)
        # hist, bin_edges = plt.hist(object[object.nonzero()], bins=200)
        self.logger.add_log("Histogram calculated. Finding peaks.")
        most_common_value = bin_edges[np.argmax(hist)]
        self.logger.add_log(f"Most common value: {most_common_value}")
        return np.abs(object) / most_common_value

    def construct_initial_object(self):
        """
            Loop through self.relevant_object_files and construct the initial object from them by using logical operators.
            Checking for overlap in the same way as in CDI_diffraction
        """
        total_object = np.zeros((grid_size, grid_size, grid_size))
        
        self.logger.add_log(f"\tLoading coordinates and density for {self.filename_prefix}.")
        self.logger.add_log("\tLooping through all files in the directory with the correct filename prefix.")
        self.logger.add_log(f"\tNumber of relevant files: {len(self.relevant_object_files)}")
        
        for file in self.relevant_object_files:
            filename = os.path.basename(file)
            self.logger.add_log(f"\tLoading coordinates and density from {filename}")
            coordinates_and_density = np.load(file)
            self.logger.add_log(f"\tCoordinates and density loaded from {filename}")

            self.logger.add_log(f"\tThe file constists of {len(coordinates_and_density)//4} object(s). (Each object is counted as a part with different density")
            print(len(coordinates_and_density))
            if len(coordinates_and_density) == 4:
                self.logger.add_log(f"\tExtracting coordinates for {filename}. Object has one density.")
                x = coordinates_and_density["x"]
                y = coordinates_and_density["y"]
                z = coordinates_and_density["z"]
                d = coordinates_and_density["d"]
                
                total_object[x,y,z] = 1*d

            elif len(coordinates_and_density) == 8:
                self.logger.add_log(f"\tExtracting coordinates for {filename}. Object has two densities.")
                xd = coordinates_and_density["xd"]
                yd = coordinates_and_density["yd"]
                zd = coordinates_and_density["zd"]
                
                xl = coordinates_and_density["xl"]
                yl = coordinates_and_density["yl"]
                zl = coordinates_and_density["zl"]
                
                dd = coordinates_and_density["dd"]
                dl = coordinates_and_density["dl"]
                
                total_object[xd,yd,zd] = 1*dd
                total_object[xl,yl,zl] = 1*dl
                
            else:
                print("Something went wrong while loading the coordinates and density. Check the file.")
                print("Error: The number of coordinates and densities is not supported.")
                self.logger.add_log(f"\tSomething went wrong while loading the coordinates and density for {filename}. Check the file.")
                self.logger.add_log(f"\tError: The number of coordinates and densities is not supported.")
            #To avoid troubles of old files. Might have to be done somewhere else.
            #Smarter to just have function that deletes all object files afterwards?
            #Or perhaps move them to a folder for old objects.
            os.remove(file)
        # if debug:
        #     self.logger.add_log("Saving the initial object.")
        #     np.save(os.path.join(self.save_path, f"{self.filename_prefix}_initial_object.npy"), total_object)
        return total_object 

    def find_nonzero(self, object):
        """
        Find the nonzero part of the objects. This is the part that is compared.
        Object is a 3f numpy array.
        TODO: Fix name. It's not really finding the nonzero part, but the part that is to be compared.
        """
        self.logger.add_log("\n\nDEBUG\n")
        x, y, z = object.nonzero()
        slices = slice(x.min(), x.max() + 1), slice(y.min(), y.max() + 1), slice(z.min(), z.max() + 1)
        # If the object is wrapping around the edges, the slices will include 0 and the size of the grid given by the shape before it is cut.
        # Will then align it with the center of the grid and to the same procedure again.
        for ind, sl in enumerate(slices):
            if sl.start == 0 and sl.stop == object.shape[ind]:
                self.logger.add_log(f"\n\n\n\t\t\tObject is wqraping around the edges (object[slice].shape = {object[slices].shape}). Aligning the object with the center of the grid.\n\n\n")
                return self.find_nonzero(self.center_object(object))
        return object[slices]

    def check_sizes(self):
        """
        Check the sizes of the initial and reconstructed objects.
        They should preferably be the same.
        If not, the objects might be resized to the same size, or just do it in some other clever way.
            Easy for cube, but might be very different for the star shape etc. But to have it in the same grid might be a good idea either way.    
        """
        self.logger.add_log("Checking sizes of initial and reconstructed objects.")
        self.logger.add_log("\tInitial object shape: " + str(self.initial_object.shape))
        self.logger.add_log("\tReconstructed object shape: " + str(self.reconstructed_object.shape))
        if self.initial_object.shape == self.reconstructed_object.shape:
            self.logger.add_log("\t\tSizes are the same.")
            return self.initial_object, self.reconstructed_object
        else:
            self.logger.add_log("\t\tSizes are not the same.")
            self.logger.add_log("\t\t\tResizing the objects to the same size.")
            return self.resize_reconstructed_and_initial_object()

    def resize_reconstructed_and_initial_object(self):
        """
        Resize the reconstructed object to the same size as the initial object.
        Choose the padding which gives the smallest deviation from the initial object.
        Should give the most "true" reconstructed object.
        """
        # changes = {}
        changes_init = [[]]
        changes_rec  = [[]]
        #Create the list with the possible changes.
        for i in range(len(self.initial_object.shape)):
            sh_i = self.initial_object.shape[i]
            sh_r = self.reconstructed_object.shape[i]
            # if sh_i > sh_r:
            #     changes[i] = ["initial"]
            # elif sh_i < sh_r:
            #     changes[i] = ["reconstructed"]
            # else:
            #     changes[i] = ["none"]

            dif = np.abs(sh_i - sh_r)
            if dif == 0:
                self.logger.add_log("\t\t\t\tNo need to resize axis " + str(i))
                for j in range(len(changes_init)):
                    changes_init[j].append((0,0))
                    changes_rec[j].append((0,0))
            elif dif%2 == 0:
                change = (dif//2, dif//2)
                self.logger.add_log(f"\t\t\t\tResizing axis {i} by {change[0]} on each side.")
                if sh_i > sh_r:
                    self.logger.add_log(f"\t\t\t\tResizing axis {i} of reconstructed object.")
                    for j in range(len(changes_init)):
                        changes_rec[j].append(change)
                        changes_init[j].append((0,0))
                    # self.reconstructed_object = np.pad(self.reconstructed_object, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
                else:
                    self.logger.add_log(f"\t\t\t\tResizing axis {i} of initial object")
                    for j in range(len(changes_init)):
                        changes_rec[j].append((0,0))
                        changes_init[j].append(change)
                    # self.initial_object = np.pad(self.initial_object, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
            else:
                for j in range(len(changes_init)):
                    changes_init.append(changes_init[j].copy())
                    changes_rec.append(changes_rec[j].copy())
                
                change_list = [(dif//2, dif//2 + 1), (dif//2 + 1, dif//2)]
                for j in range(len(changes_init)):
                    #Trust me, it's bad, but it works
                    change = change_list[(2*j//len(changes_init))%2]
                    if sh_i > sh_r:
                        changes_rec[j].append(change)
                        changes_init[j].append((0,0))
                        # self.reconstructed_object = np.pad(self.reconstructed_object, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
                    else:
                        changes_init[j].append(change)
                        changes_rec[j].append((0,0))
                        # self.initial_object = np.pad(self.initial_object, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
        self.logger.add_log("\t\t\t\tPossible changes:")
        self.logger.add_log(f"\t\t\t\t\tInitial object: {changes_init}")
        self.logger.add_log(f"\t\t\t\t\tReconstructed object: {changes_rec}")
        
        dev_keep = -1
        ind_keep = -1
        self.logger.add_log(f"\t\t\t\tLooping through the possible changes for the axes.")
        for index, change in enumerate(zip(changes_init, changes_rec)):
            self.logger.add_log(f"\t\t\t\t\tChange {index}: {change}")
            change_init = change[0]
            change_rec = change[1]
            dev = np.sum(np.abs(np.pad(self.reconstructed_object, change_rec, 'constant', constant_values=0))-np.pad(self.initial_object, change_init, 'constant', constant_values=0))
            
            if dev < dev_keep or dev_keep == -1:
                dev_keep = dev
                ind_keep = index
        self.reconstructed_object = np.pad(self.reconstructed_object, changes_rec[ind_keep], 'constant', constant_values=0)
        self.initial_object = np.pad(self.initial_object, changes_init[ind_keep], 'constant', constant_values=0)
        return self.initial_object, self.reconstructed_object
           
    def resize_objects(self, object1, object2):
        """
        Resize the object matrices to the same size by zero-padding the smallest.
        Have to be done for every axis induvidually, as their sizes might vary.
        This assumes that the objects have approximately the same shape, if not, further considerations have to be taken into account.
        """
        #TODO: 17.11
        #TODO: Check which direction the append should be done in by checking which direction is giving the smallest deviation.
        #TODO: Might be a slight cheat, but should be ok. Now they sometime seem to be shifted by one compared to the optimal solution.
        for i in range(len(object1.shape)):
            o1_i = object1.shape[i]
            o2_i = object2.shape[i]
            
            dif = np.abs(o1_i - o2_i)
            if dif == 0:
                self.logger.add_log("\t\t\t\tNo need to resize axis " + str(i))
                continue
            
            if dif%2 == 0:
                change = (dif//2, dif//2)
                self.logger.add_log(f"\t\t\t\tResizing axis {i} by {change[0]} on each side.")
            else:
                change = (dif//2, dif//2 + 1)
                self.logger.add_log(f"\t\t\t\tResizing axis {i} by {change[0]} on one side and {change[1]} on the other side.")

            if dif/o1_i > 0.1:
                #TODO: Figure out how much this impacts the results.
                self.logger.add_log("Warning: The difference in size is more than 10 percent. This might be a problem. Objects might be rotated compared to each other.")


            #TODO: Zero-padding seems to be the best option, but maybe there are better ways to do it.
            #TODO: Check if the objects are rotated compared to each other. If so, rotate them.
            #TODO: Find a way to find the name of the objects.
            if o1_i > o2_i:
                self.logger.add_log(f"\t\t\t\tResizing axis {i} of object 2.")
                object2 = np.pad(object2, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
            else:
                self.logger.add_log(f"\t\t\t\tResizing axis {i} of object 1")
                object1 = np.pad(object1, np.roll(np.array([change, (0,0), (0,0)]), i, axis = 0), 'constant', constant_values=0)
        return object1, object2
    def find_center_of_mass(self, object):
        """
        Find the center of mass of the object.
        Returns the x, y, and z coordinate of the center of mass of the 3D numpy array.
        Strictly speaking, this returns some sort of center of position, as the density/mass is not taken into account.
        The purpose of the function is just find coordinates that will later be used to align the objects.

        TODO: Is this necessary after I've already stripped them of their nonzero part? Might be useful in less perfect examples.
        """
        x, y, z = object.nonzero()
        return x.mean(), y.mean(), z.mean()

    def center_object(self, object):
        """
        Center the object around the origin.
        """
        x, y, z = self.find_center_of_mass(object)
        return np.roll(object, (int(object.shape[0]//2 - x), int(object.shape[1]//2 - y), int(object.shape[2]//2 - z)), axis = (0,1,2))

    def align_objects(self):
        """
        Align the objects by shifting them so that their center of mass is the same, i.e. in the middle of the matrix.
        Using np.roll to roll the matrices with the difference in center of mass from (len(array_dimension) - 1)/2.
        """
        self.logger.add_log("Aligning the objects.")
        cm_i_x, cm_i_y, cm_i_z = self.find_center_of_mass(self.initial_object)
        cm_r_x, cm_r_y, cm_r_z = self.find_center_of_mass(self.reconstructed_object)
        h_i, w_i, d_i = self.initial_object.shape
        self.initial_object = np.roll(self.initial_object, (int((h_i-1)/2 - cm_i_x), int((w_i-1)/2 - cm_i_y), int((d_i-1)/2 - cm_i_z)), axis=(0,1,2))
        h_r, w_r, d_r = self.reconstructed_object.shape
        self.reconstructed_object = np.roll(self.reconstructed_object, (int((h_r-1)/2 - cm_r_x), int((w_r-1)/2 - cm_r_y), int((d_r-1)/2 - cm_r_z)), axis=(0,1,2))
        cm_i_x_new, cm_i_y_new, cm_i_z_new = self.find_center_of_mass(self.initial_object)
        cm_r_x_new, cm_r_y_new, cm_r_z_new = self.find_center_of_mass(self.reconstructed_object)
        if cm_i_x_new == cm_r_x_new and cm_i_y_new == cm_r_y_new and cm_i_z_new == cm_r_z_new:
            self.logger.add_log("Objects aligned.")
        elif np.abs(cm_i_x_new - cm_r_x_new) < 2 and np.abs(cm_i_y_new - cm_r_y_new) < 2 and np.abs(cm_i_z_new - cm_r_z_new) < 2:
            self.logger.add_log("Objects aligned, but not perfectly.")
        else:
            self.logger.add_log("Objects not aligned. Check what has gone wrong.")
            self.logger.add_log(f"Inital object center of mass: {cm_i_x_new}, {cm_i_y_new}, {cm_i_z_new}. Initial object shape: {self.initial_object.shape}")
            self.logger.add_log(f"Reconstructed object center of mass: {cm_r_x_new}, {cm_r_y_new}, {cm_r_z_new}. Reconstructed object shape: {self.reconstructed_object.shape}")

    def find_relevant_object_files(self):
        """
            Function to find the relevant files in the object_path.
            The files should be .npy files.

        Returns:
            list: List containing the paths to the relevant files.
        """    
        relevant_files = []
        self.logger.add_log(f"\tFinding relevant object files by looping through the directory where the objects are saved ({self.initial_object_path})")
        for file in os.listdir(self.initial_object_path):
            #TODO: Make this more robust. Unsure if it should be moved to utility fuction 
            if file.find(self.filename_prefix + "_") != -1 and os.path.basename(file).startswith("old") == True:
                relevant_files.append(os.path.join(self.initial_object_path, file))
                self.logger.add_log(f"\t\tFound relevant file: {file}")
        self.logger.add_log(f"\tFound {len(relevant_files)} relevant files.")
        return relevant_files
                     
    def find_relevant_reconstruction_files(self):
        """
            Function to find the relevant files in the reconstructed_object_path.
            The files should be .cxi files.

        Returns:
            list: List containing the paths to the relevant files.
        """    
        relevant_files = []
        self.logger.add_log(f"\tFinding relevant reconstruction files by looping through the directory where the reconstructions are saved ({self.reconstructed_object_path})")
        for file in os.listdir(self.reconstructed_object_path):
            if file.find(self.filename_prefix + "_") != -1:
                relevant_files.append(os.path.join(self.reconstructed_object_path, file))
                self.logger.add_log(f"\t\tFound relevant file: {file}")
        self.logger.add_log(f"\tFound {len(relevant_files)} relevant files.")
        return relevant_files

    def find_correct_rotation(self, object):
        """
        Find the correct rotation between the initial object and the reconstructed object.
        Look at the shapes of the matrices. Align them such that the shapes are in rising order.
        """
        self.logger.add_log("Finding the correct rotation.")
        object_shape = object.shape
        self.logger.add_log(f"Object shape: {object_shape}")
        shape_array = np.array([object_shape[i] for i in range(len(object_shape))])
        self.logger.add_log(f"Object shape after transposing based on sorted shape list: {shape_array.argsort()}")
        return np.transpose(object, shape_array.argsort())

    def flip_if_inverted(self):
        """
        Check if the reconstructed object is flipped, i.e. it is the centrocymmetric inversion of the initial object. 
        """
        self.logger.add_log("\tChecking if the reconstructed object is flipped.")
        temp = None
        if self.reconstructed_object.shape != self.initial_object.shape:
            #SORRY
            self.logger.add_log("\t\tThe reconstructed object is not the same shape as the initial object. Have to pad or strip reconstructed object to match the initial object. Fixing correct size of reconstructed object will be dealt with later.")
            temp = self.reconstructed_object.copy()
            #Quick fix to account for the fact that I didn't flip them before adding them to the mean after reorganizing. Have to be fixed.
            if temp.max() > 40:
                temp = temp / 61.28
            if self.reconstructed_object.shape[0] > self.initial_object.shape[0]:
                temp = temp[:self.initial_object.shape[0],:,:]
            elif self.reconstructed_object.shape[0] < self.initial_object.shape[0]:
                temp = np.pad(temp, ((0, self.initial_object.shape[0] - self.reconstructed_object.shape[0]), (0,0), (0,0)), mode = 'constant', constant_values = 0)
            if self.reconstructed_object.shape[1] > self.initial_object.shape[1]:
                temp = temp[:,:self.initial_object.shape[1],:]
            elif self.reconstructed_object.shape[1] < self.initial_object.shape[1]:
                temp = np.pad(temp, ((0,0), (0, self.initial_object.shape[1] - self.reconstructed_object.shape[1]), (0,0)), mode = 'constant', constant_values = 0)
            if self.reconstructed_object.shape[2] > self.initial_object.shape[2]:
                temp = temp[:,:,:self.initial_object.shape[2]]
            elif self.reconstructed_object.shape[2] < self.initial_object.shape[2]:
                temp = np.pad(temp, ((0,0), (0,0), (0, self.initial_object.shape[2] - self.reconstructed_object.shape[2])), mode = 'constant', constant_values = 0)
        if isinstance(temp, np.ndarray):
            deviation = np.sum(np.abs(self.find_deviation(temp)))
            deviation_flipped = np.sum(np.abs(self.find_deviation(np.flip(temp))))
        else:
            deviation = np.sum(np.abs(self.find_deviation(self.reconstructed_object)))
            deviation_flipped = np.sum(np.abs(self.find_deviation(np.flip(self.reconstructed_object))))
        if deviation_flipped < deviation:
            self.logger.add_log("The reconstructed object is flipped. Flipping it back.")
            self.reconstructed_object = np.flip(self.reconstructed_object)
        else:
            self.logger.add_log("The reconstructed object is not flipped.")
        return self.reconstructed_object

            
    def find_deviation(self, object):
        """
        Find the deviation in intensity of the reconstructed object from the initial. The maximum intensity should be one.
        Could divide by the maximum value to normalize it, but this might place a too large weight on a potentially deviating value.
        Should probably divide by the mean.
        TODO: DW wanted me to find out the importance of the values in the reconstruction before I divide by anything. Don't think it's too important though.
        Assume reconstructed object already normalized.
        # d = 4
        # pos = self.initial_object.shape[2]//d
        # #median = np.median(np.abs(self.reconstructed_object[self.reconstructed_object.nonzero()]))
        # deviation = np.sum(np.abs(np.abs(self.reconstructed_object) - self.initial_object)) #[:,:,pos]
        # self.logger.add_log(f"\n\n\nDEBUG: Deviation: {deviation}")
        # axis = None
        # for ax in [0,1,2,(0,1),(0,2),(1,2),(0,1,2)]: 
        #     deviation_flipped = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=ax)) - self.initial_object))
        #     self.logger.add_log(f"DEBUG: Deviation for axis {ax}: {deviation_flipped}")
        #     if deviation_flipped < deviation and ax == (0,1,2):
        #         deviation = deviation_flipped
        #         axis = ax
        # if axis != None:
        #     #Flip reconstructed object to the orientation that gave the smallest deviation
        #     self.reconstructed_object = np.flip(self.reconstructed_object, axis=axis)
        #         #TODO: Figure out what to divide on.
        
        """
        return (np.abs(object) - self.initial_object)/np.where(self.initial_object, self.initial_object, 1)  #.sum()/np.abs(self.initial_object).sum()

    def find_error(self):
        """
        Find the error of the deviation.
        """
        return np.sum(self.deviation)


    def save_error(self, error):
        """
        Save the error to a file.
        """
        with open(os.path.join(self.save_path, "error.txt"), "w") as f:
            f.write(str(error))

    def plot_error(self):
        """
        Plot the error.
        """

        plt.imshow(self.deviation[0])
        plt.savefig(os.path.join(self.save_path, "error.png"))

    def plot_gif_imshow(self, data, folder = None, filename = None, axis = 2, filename_extra = None):
        """_summary_

        Args:
            data (_type_): _description_
            filename (_type_): _description_
            axis (int, optional): Axis to plot along. Defaults to 2, corresponding to the z-axis.
        """ 
        #Might be very inefficient for large data sets, but they should not be very large for the reconstructed objects.
        data = np.transpose(data, np.roll((0,1,2), -axis))
        if filename == None:
            filename = f"{self.filename_prefix}_{self.recon_index}"

        if filename_extra != None:
            filename += f"{filename_extra}"
        

        print('building plots\n')
        filenames = []
        for i, data_slice in enumerate(data):
            filename_image = f"{filename}_{i}.png"
            filenames.append(filename_image)
            fig, ax = plt.subplots(figsize=(9,9))
            plt.title(f"Slice {i+1} of {len(data)}")
            min = np.floor(data.min()*100)/100
            max = np.ceil(data.max()*100)/100
            largest = np.max([np.abs(min), np.abs(max)])
            lower = -largest if min < 0 else min
            upper = largest
            if lower < 0:
                cmap = plt.cm.get_cmap('RdBu', 100)
            else:
                cmap = plt.cm.get_cmap('Blues',100)
            im = plt.imshow(data_slice, vmin = lower, vmax= upper, cmap=cmap)
            cax = set_colorbar_axes(fig, ax)
            if lower < 0:
                plt.colorbar(im, cax, extend='both')
                plt.clim(-0.3,0.3)
            else:
                plt.colorbar(im, cax)
                plt.clim(0.3)    
                
            if folder == None:
                plt.savefig(os.path.join(self.save_path, filename_image),bbox_inches='tight', pad_inches=1, transparent=True)
            else:
                plt.savefig(os.path.join(folder, filename_image),bbox_inches='tight', pad_inches=1, transparent=True)
            plt.close()

        #Yields 10 second gif. Should be able to give as a parameter. TODO.
        duration = 10/len(data)
        
        with imageio.get_writer(os.path.join(self.save_path, filename + '.gif'), mode='I', duration=duration) as writer:
            for fn in filenames:
                image = imageio.imread(os.path.join(self.save_path,fn))
                writer.append_data(image)
            # image = imageio.imread(filename)
            # for _ in range(n_frames):
            #     writer.append_data(image)
        #Using set in case there are duplicates. Should not be duplicates.
        for fn in set(filenames):
            os.remove(os.path.join(self.save_path,fn))

    def plot_histogram(self, data, filename_extra = None):
        """
        Plot the histogram of the deviation.
        """
        if filename_extra != None:
            filename = f"{self.filename_prefix}_{filename_extra}_{self.recon_index}"
        else:
            filename = f"{self.filename_prefix}_{self.recon_index}"
        counts, bins = np.histogram(data[data.nonzero()], bins = 100)
        fig, ax = plt.subplots(figsize = (9,9))
        plt.stairs(counts, bins, fill=True)
        # plt.hist(data[data.nonzero()].flatten(), bins = 100)
        plt.savefig(os.path.join(self.save_path, filename + "_histogram.png"))
        plt.close()

    
def main():
    fnp = "0000000_12_64"
    log = Logger(filename_prefix=fnp)
    try:
        ea = ErrorAnalysis(filename_prefix=fnp, logger = log)
    except Exception as e:
        print(e)
        log.save_log()
    log.save_log()

if __name__ == "__main__":
    main()


"""
    def reduce_intensity_cylindrical_coordinates(self, data, axis = 2, radius = 0.5, intensity = 0.5):
    '''
    Reduce the intensity of the object in a cylindrical region around the center. 
    This is done to avoid the problem of the reconstruction being too bright in the center.
    '''
    data = np.transpose(data, np.roll((0,1,2), -axis))
    center = np.array(data.shape)/2
    center = center.astype(int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i-center[0])**2 + (j-center[1])**2) < radius:
                data[i,j] = data[i,j]*intensity
    return np.transpose(data, np.roll((0,1,2), axis))

    # self.logger.add_log("Finding the correct rotation between the initial object and the reconstructed object.")
        # deviation = np.zeros((3, 3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             deviation[i,j,k] = np.sum(np.abs(np.transpose(self.reconstructed_object, (i,j,k)) - self.initial_object))
        # self.logger.add_log(f"Deviation matrix: {deviation}")
        # self.logger.add_log(f"Deviation matrix shape: {deviation.shape}")
        # self.logger.add_log(f"Deviation matrix min: {np.min(deviation)}")
        # self.logger.add_log(f"Deviation matrix max: {np.max(deviation)}")
        # self.logger.add_log(f"Deviation matrix mean: {np.mean(deviation)}")
        # self.logger.add_log(f"Deviation matrix std: {np.std(deviation)}")
        # self.logger.add_log(f"Deviation matrix argmin: {np.argmin(deviation)}")
        # self.logger.add_log(f"Deviation matrix argmax: {np.argmax(deviation)}")
        # self.logger.add_log(f"Deviation matrix argmin shape: {np.argmin(deviation).shape}")
        # self.logger.add_log(f"Deviation matrix argmax shape: {np.argmax(deviation).shape}")
        # self.logger.add_log(f"Deviation matrix argmin type: {type(np.argmin(deviation))}")
        # self.logger.add_log(f"Deviation matrix argmax type: {type(np.argmax(deviation))}")
        # self.logger.add_log(f"Deviation matrix argmin dtype: {np.argmin(deviation).dtype}")
        # self.logger.add_log(f"Deviation matrix argmax dtype: {np.argmax(deviation).dtype}")
        # self.logger.add_log(f"Deviation matrix argmin ndim: {np.argmin(deviation).ndim}")
        # self.logger.add_log(f"Deviation matrix argmax ndim: {np.argmax(deviation).ndim}")
        # self.logger.add_log(f"Deviation matrix argmin size: {np.argmin(deviation).size}")
        # self.logger.add_log(f"Deviation matrix argmax size: {np.argmax(deviation).size}")
        # self.logger.add_log(f"Deviation matrix argmin itemsize: {np.argmin(deviation).itemsize}")
        # # self.logger.add_log(f"Deviation matrix arg

            # deviation_flipped = np.sum(np.abs(np.flip(self.reconstructed_object))/median - self.initial_object)#np.sum(np.abs(np.flip(self.reconstructed_object[:,:,self.recon
        # deviation_flipped_axis2 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=2))/median - self.initial_object)#np.sum(np.abs(np.flip(self.reconstructed_object[
        # deviation_flipped_axis1 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=1))/median - self.initial_object)#np.sum(np.abs(np.flip(self.reconstructed_object[
        # deviation_flipped_axis0 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=0))/median - self.initial_object)#np.sum(np.abs(np.flip(self.reconstructed_object[
        # deviation_flipped_axis01 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(0,1)))/median - self.initial_object)#
        # deviation_flipped_axis02 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(0,2)))/median - self.initial_object)#
        # deviation_flipped_axis12 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(1,2)))/median - self.initial_object)#

        # if axis != (0,1):
        #     print(f"\nDEBUG: The axis that gave the smallest deviation was not (0,1), but {axis}.\n")

        # deviation_flipped = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object)[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis0 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=0)[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis1 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=1)[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis2 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=2)[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis01 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(0,1))[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis02 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(0,2))[:,:,pos])/median - self.initial_object[:,:,pos]))
        # deviation_flipped_axis12 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(1,2))[:,:,pos])/median - self.initial_object[:,:,pos]))

        # deviation_flipped = np.sum(np.abs(np.flip(self.reconstructed_object)[:,:,pos])/median - self.initial_object[:,:,pos])#np.sum(np.abs(np.fl
        # deviation_flipped_axis2 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=2)[:,:,pos])/median - self.initial_object[:,:,pos])#np.su
        # deviation_flipped_axis1 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=1)[:,:,pos])/median - self.initial_object[:,:,pos])#np.su
        # deviation_flipped_axis0 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=0)[:,:,pos])/median - self.initial_object[:,:,pos])#np.su
        # deviation_flipped_axis01 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(0,1))[:,:,pos])/median - self.initial_object[:,:,pos])#
        # deviation_flipped_axis02 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(0,2))[:,:,pos])/median - self.initial_object[:,:,pos])#
        # deviation_flipped_axis12 = np.sum(np.abs(np.flip(self.reconstructed_object, axis=(1,2))[:,:,pos])/median - self.initial_object[:,:,pos])#
        


        # deviation = np.sum(np.abs(np.abs(self.reconstructed_object)/median - self.initial_object))
        # deviation_flipped = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object))/median - self.initial_object))#np.sum(np.abs(np.fl
        # deviation_flipped_axis2 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=2))/median - self.initial_object))#np.sum(np.abs(np.flip(self.reconstructed_object, axis=2))/median - self.initial_obj
        # deviation_flipped_axis1 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=1))/median - self.initial_object))#np.sum(np.abs(np.flip(self.reconstructed_object, axis=1))/median - self.initial_obj
        # deviation_flipped_axis0 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=0))/median - self.initial_object))#np.sum(np.abs(np.flip(self.reconstructed_object, axis=0))/median - self.initial_obj
        # deviation_flipped_axis01 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(0,1)))/median - self.initial_object))#
        # deviation_flipped_axis02 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(0,2)))/median - self.initial_object))#
        # deviation_flipped_axis12 = np.sum(np.abs(np.abs(np.flip(self.reconstructed_object, axis=(1,2)))/median - self.initial_object))#




        # if True:
        #     old = self.reconstructed_object.copy()
        #     self.reconstructed_object = np.flip(self.reconstructed_object)
        #     self.logger.add_log("\n\n\nDEBUG: Flipped the reconstructed object.")
        #     self.logger.add_log(f"DEBUG: Difference between old and new. Should not be 0: {np.sum(np.abs(old - self.reconstructed_object))}")
        #     self.logger.add_log(f"\tDeviation: {deviation}")
        #     self.logger.add_log(f"\tDeviation flipped: {deviation_flipped}\n\n\n")
        #     self.logger.add_log(f"\tDeviation flipped axis 0: {deviation_flipped_axis0}")
        #     self.logger.add_log(f"\tDeviation flipped axis 1: {deviation_flipped_axis1}")
        #     self.logger.add_log(f"\tDeviation flipped axis 2: {deviation_flipped_axis2}")
        #     self.logger.add_log(f"\tDeviation flipped axis 01: {deviation_flipped_axis01}")
        #     self.logger.add_log(f"\tDeviation flipped axis 02: {deviation_flipped_axis02}")
        #     self.logger.add_log(f"\tDeviation flipped axis 12: {deviation_flipped_axis12}")

        #Divides by the original object where it has values. Else just divide by the median/mean/one.

    Nice, men ikke helt det jeg vil. Usikker på hva som er mest upraktisk av å lagre store filer eller å ha tonnevis med mapper.
    def save_data(self, index):
        self.logger.add_log(f"Saving data for {index}")
        data = {}
        data["initial"] = self.initial_object
        data["reconstructed"] = self.reconstructed_object
        data["deviation"] = self.deviation
        data["deviation_abs"] = self.deviation_abs
        data["error"] = self.error
        data["error_abs"] = self.error_abs
        data["error_rel"] = self.error_rel
        data["error_rel_abs"] = self.error_rel_abs
        np.save(os.path.join(self.save_path, f"{self.filename_prefix}_{index}.npy"), data)
        self.logger.add_log(f"Data saved for {index}")

        if not os.path.isdir(os.path.join(self.save_path, "reconstructed_objects_abs")):
            os.mkdir(os.path.join(save_path, "reconstructed_objects_abs"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "reconstructed_objects_norm")):
            os.mkdir(os.path.join(save_path, "reconstructed_objects_norm"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "histograms")):
            os.mkdir(os.path.join(save_path, "histograms"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "histograms_abs")):
            os.mkdir(os.path.join(save_path, "histograms_abs"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "histograms_initial")):
            os.mkdir(os.path.join(save_path, "histograms_initial"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "histograms_reconstructed")):
            os.mkdir(os.path.join(save_path, "histograms_reconstructed"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "histograms_reconstructed_norm")):
            os.mkdir(os.path.join(save_path, "histograms_reconstructed_norm"))
            time.sleep(1)   
        
            os.mkdir(os.path.join(save_path, "gifs_deviation_abs"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "gifs_deviation_abs")):
            os.mkdir(os.path.join(save_path, "gifs_deviation_abs"))
            time.sleep(1)
        if not os.path.isdir(os.path.join(save_path, "gifs_deviation_norm")):

"""