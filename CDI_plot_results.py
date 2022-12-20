import os
import time
import numpy as np
import matplotlib.pyplot as plt
from CDI_logger import Logger
from CDI_description import Description
from scipy.optimize import curve_fit

#TODO: Save the histogram data to a file in CDI_error_analysis.py
#Might be an idea to save it to a folder given by the filename_prefix_prefix
#Then each of the different intensity reductions to an individual folder within this folder.
#Might also store a description file in stead of storing all the information in the filename, e.g. the grid_size and the size of the k-space and everything.
#This file might be accessed from different parts of the code to be made additions to
#TODO: Implement method to find intensity reduction percentage.

class PlotResults():
    def __init__(self, group, logger: Logger, description: Description):
        self.group = group
        self.logger = logger
        self.description = description
        self.data_path = os.path.join(os.getcwd(), "error_analysis", self.group)
        self.save_path = os.path.join(self.data_path, "plots")
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
            time.sleep(1)
        self.define_plot_parameters()
        self.plot_histograms()
        # self.plot_reconstructed_values_together()
        # self.plot_reconstructed_values()

    #TODO: Implement function to plot all values for each intensity reduction percentage
    #TODO: Implement function to plot all mean values for the different intensity reductions together.

    def define_plot_parameters(self):
        #Example of what can be set.
        #might have to have some differences between the different plots, but font size, type etc should always be the same. Preferably also figsize (if saving it as an svg does not keep the font size when the rest of the figure is scaled).
        # self.figsize = (10, 10)
        # self.fontsize = 15
        # self.linewidth = 2
        # self.markersize = 10
        # self.marker = "o"
        # self.grid = True
        # self.title_fontsize = 20
        # self.legend_fontsize = 15
        # self.xlabel_fontsize = 15
        # self.ylabel_fontsize = 15
        # self.title = "Reconstructed values"
        # self.xlabel = "Reconstructed value"
        # self.ylabel = "Frequency"
        plt.rcParams["figure.facecolor"] = (1,1,1,0) #Makes the background transparent
        plt.rcParams["font.family"] ="DeJavu serif"
        plt.rcParams["font.serif"] = "Palatino"
        #Burde variere alt ettersom det er ment å dekke hele siden eller ikke. Bør vel strengt talt sette den for vært plot.
        #Evt bare kjøre en jalla versjon der denne parameteren endres for hvert plot.
        plt.rcParams["font.size"] = 15
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["xtick.major.size"] = 5
        plt.rcParams["ytick.major.size"] = 5
        plt.rcParams["xtick.minor.size"] = 3
        plt.rcParams["ytick.minor.size"] = 3
        plt.rcParams["xtick.minor.width"] = 1.5
        plt.rcParams["ytick.minor.width"] = 1.5
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["grid.alpha"] = 0.5
        plt.rcParams["legend.fontsize"] = 15
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.loc"] = "best"
        plt.rcParams["legend.labelspacing"] = 0.5
        plt.rcParams["legend.handlelength"] = 1.5
        plt.rcParams["legend.handletextpad"] = 0.5
        plt.rcParams["legend.borderpad"] = 0.5
        plt.rcParams["legend.borderaxespad"] = 0.5
        plt.rcParams["legend.columnspacing"] = 1
        plt.rcParams["legend.markerscale"] = 1
        plt.rcParams["legend.numpoints"] = 1
        plt.rcParams["legend.scatterpoints"] = 1
        plt.rcParams["legend.framealpha"] = 1
        plt.rcParams["legend.facecolor"] = "inherit"
        plt.rcParams["legend.edgecolor"] = "inherit"
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.shadow"] = False
        plt.rcParams["legend.title_fontsize"] = 15
        # plt.rcParams["legend.title_fontweight"] = "normal"
        # plt.rcParams["legend.title_fontstyle"] = "normal"
        # plt.rcParams["legend.title_fancybox"] = False
        # plt.rcParams["legend.title_framealpha"] = 1
        # plt.rcParams["legend.title_facecolor"] = "inherit"
        # plt.rcParams["legend.title_edgecolor"] = "inherit"
        # plt.rcParams["legend.title_shadow"] = False
        # plt.rcParams["legend.title_borderpad"] = 0.5
        # plt.rcParams["legend.title_columnspacing"] = 1
        # plt.rcParams["legend.title_labelspacing"] = 0.5
        # plt.rcParams["legend.title_loc"] = "best"
        # plt.rcParams["legend.title_bbox_to_anchor"] = (1, 1)
        # plt.rcParams["legend.title_bbox_transform"] = "axes fraction"
        # plt.rcParams["legend.title_borderaxespad"] = 0.5
        # plt.rcParams["legend.title_borderpad"] = 0.5
        # plt.rcParams["legend.title_frameon"] = False
        
        self.filetype = "svg"

    def plot_histograms(self):
        #TODO: Call functions that plots histograms. Bins etc for all histogram are set here.
        try:
            self.logger.add_log("Staring to plot histograms of reconstructed objects.")
            self.plot_reconstructed_hists()
            self.logger.add_log("Finished plotting histograms of reconstructed objects.")
        except Exception as e:
            self.logger.add_log(f"Failed to plot histograms of reconstructed objects. Error: {e}")
        try:
            self.logger.add_log("Staring to plot histograms of initial object.")
            self.plot_initial_histogram()
            self.logger.add_log("Finished plotting histograms of initial object.")
        except Exception as e:
            self.logger.add_log(f"Failed to plot histograms of initial object. Error: {e}")
        

    def plot_reconstructed_hists(self):
        # stop = 1.6
        # self.bins_rec_hist = [0.01*i for i in range(0, int(stop*100) + 1)]
        # self.xticks_rec_hist = np.arange(0, stop, step=0.2)
        self.num_bins_hist = 500
        self.xticks_rec_hist, self.bins_rec_hist = self.find_xticks(folder = "reconstructed_objects")
        self.xticks_rec_hist_labels = [f"{i:.1e}" for i in self.xticks_rec_hist]

        self.figsize_hist = (7, 7)

        self.remove_yticks = False
        if self.remove_yticks:
            self.yticks_rec_hist = []
        else:
            self.yticks_rec_hist = self.find_yticks(folder = "reconstructed_objects")
        
        self.plot_legend = True
        try:
            self.logger.add_log("Plotting histograms of reconstructed objects")
            self.plot_reconstructed_values()
        except Exception as e:
            self.logger.add_log(f"Error when plotting histograms of reconstructed objects: {e}")
        try:
            self.logger.add_log("Plotting histograms of reconstructed objects together")
            self.plot_reconstructed_values_together()
        except Exception as e:
            self.logger.add_log(f"Error when plotting histograms of reconstructed objects together: {e}")
        try:
            self.logger.add_log("Plotting mean histogram of reconstructed objects")
            self.plot_reconstructed_mean_values()
        except Exception as e:
            self.logger.add_log(f"Error when plotting mean histogram of reconstructed objects: {e}")
        

    def plot_initial_histogram(self):
        data = np.load(os.path.join(self.data_path, "initial_object", "initial.npy"))
        hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.num_bins_hist)
        plt.stairs(hist, bin_edges)
        plt.xlabel("Density")
        plt.ylabel("Number of voxels")
        plt.title("Density distribution of initial object")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "initial_histogram." + self.filetype), dpi = 300)
        plt.close()

    def plot_reconstructed_values(self):
        for ind, dir in enumerate(os.listdir(os.path.join(self.data_path,"reconstructed_objects"))):
            plt.figure(figsize=self.figsize_hist)
            for file in os.listdir(os.path.join(self.data_path,"reconstructed_objects", dir)):
                if file.startswith("mean"):
                    continue
                data = np.load(os.path.join(self.data_path,"reconstructed_objects",dir,file))
                hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.bins_rec_hist)
                plt.stairs(hist, bin_edges, alpha = 0.5, label = file, fill = True)
                
            if dir != "none":
                plt.title(f"Density distribution for {dir}% intensity reduction", pad = 15)
            else:
                plt.title(f"Density distribution for no intensity reduction", pad = 15)
            #TODO:#?#?#?Kan være forvirrende at dette oppgis som intensitet ettersom man også snakker om intensitetreduksjoner.
            plt.xlabel("Density")
            plt.ylabel("Number of voxels")
            plt.xticks(ticks = self.xticks_rec_hist, labels = self.xticks_rec_hist_labels)
            # plt.yticks(ticks = self.yticks_rec_hist, labels = [])
            plt.yticks(ticks = self.yticks_rec_hist)

            #TODO: Ask Basab and DW if it's even nescassary to have a legend. Just different reconstruction numbers. Otherwise they are the same.
            if self.plot_legend: plt.legend(title="Reconstruction number")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"{dir}_rec_hist.{self.filetype}"))
            plt.close()     

    def plot_reconstructed_mean_values(self):
        plt.figure(figsize=self.figsize_hist)
        for dir in os.listdir(os.path.join(self.data_path,"reconstructed_objects")):
            data = np.load(os.path.join(self.data_path,"reconstructed_objects",dir,"mean.npy"))
            hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.bins_rec_hist)
            hist = hist / (len(os.listdir(os.path.join(self.data_path,"reconstructed_objects",dir))) - 1)
            plt.stairs(hist, bin_edges, alpha = 0.5, label = dir, fill = True)
        plt.title("Mean density distribution for all intensity reductions")
        plt.xlabel("Density")
        plt.ylabel("Number of voxels")
        plt.xticks(ticks = self.xticks_rec_hist, labels = self.xticks_rec_hist_labels)
        plt.yticks(ticks = self.yticks_rec_hist)
        plt.legend(title="Intensity reduction[%]")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f"mean_rec_hist.{self.filetype}"))
        plt.close()
        

    def plot_reconstructed_values_together(self):
        #Have to subtract the mean file.
        num_figs = len(os.listdir(os.path.join(self.data_path,"reconstructed_objects")))
        self.logger.add_log(f"Number of figures: {num_figs}")
        fig, axs = plt.subplots(nrows = num_figs//2 + num_figs%2, ncols = 2, sharey=True, sharex=True, figsize=(12,9))
        #TODO: Try differrent settings with sharex and sharey, "row, "col", "all".
        stop = 1.6
        bins = [0.01*i for i in range(0, int(stop*100) + 1)]
        xticks = np.arange(0, stop, step=0.2)
        alpha = 0.5
        plt.title("Density distribution for different intensity reductions", pad = 15)
        try: 
            directories = os.listdir(os.path.join(self.data_path,"reconstructed_objects"))
            directories.remove("none")
            directories.sort()
            directories.insert(0, "none")
            for ind, dir in enumerate(directories):
                if dir != "none":
                    if dir.startswith("0"):
                        axs[ind//2, ind%2].set_title(f"{dir[1:]}% intensity reduction")    
                    else:
                        axs[ind//2, ind%2].set_title(f"{dir}% intensity reduction")
                else:
                    axs[ind//2, ind%2].set_title(f"No intensity reduction")
                for file in os.listdir(os.path.join(self.data_path,"reconstructed_objects", dir)):
                    if file.startswith("mean"):
                        continue
                    data = np.load(os.path.join(self.data_path,"reconstructed_objects",dir,file))
                    hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.bins_rec_hist)
                    #Calculate mean and standard deviation here. Add it to description file in some reasonable way. 
                    #Have to save in the description file how many parts of the object there are, i.e. different densities. This will help the program known how many gaussians to fit.
                    try:
                        params, params_std = self.gaussian_fit(bin_edges, hist)
                        self.logger.add_log(f"Curve fitted values for intensity reduction {dir} and reconstruction number: {file}")
                        for param, value, value_std in zip(["a1", "x01", "sigma1", "a2", "x02", "sigma2"], params, params_std):
                            self.logger.add_log(f"{param}: {value}")
                            self.logger.add_log(f"{param}_std: {value_std}")
                    except Exception as e:
                        self.logger.add_log(f"Could not fit gaussian curve for {dir} and {file}. Error: {e}")
                    axs[ind//2, ind%2].stairs(hist, bin_edges, fill = True, label = file, alpha = alpha)
                # axs[ind//2, ind%2].legend()
                # axs[ind//2,ind%2].set_xticks(xticks)
            plt.setp(axs, xticks=self.xticks_rec_hist)#, xticklabels=self.xticks_reconstructed_values)
            axs[-1,0].set_xlabel("Relative density")
            axs[-1,1].set_xlabel("Relative density")
            plt.setp(axs, ylabel="Counts")
            hists_labels = [ax.get_legend_handles_labels() for ax in axs.ravel()]
            #Credit: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots
            hists, labels = [sum(lol, []) for lol in zip(*hists_labels)]
            self.logger.add_log(f"labels: {labels}")
            self.logger.add_log(f"hists: {hists}")
            labels = set(labels)
            labels = list(labels)
            labels.sort()
            hists = hists[:len(labels)]
            self.logger.add_log(f"labels: {labels}")
            self.logger.add_log(f"hists: {hists}")
            if self.plot_legend: fig.legend(hists, labels, loc='upper right')
            # axs[0,0].legend(loc="center left", bbox_to_anchor=(3, 0.5))
            fig.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"reconstructed_values.{self.filetype}"))
            plt.close()
            # plt.show()
        except Exception as e:
            print(f"Error: {e}")
            self.logger.add_log(f"Error when plotting the density distributions together: {e}")
            #In case there is only one or two files in the directory
            # for ind, dir in enumerate(os.listdir(os.path.join(self.data_path,"reconstructed_objects"))):
            #     print(dir)
            #     axs[ind].set_title(dir)
            #     for file in os.listdir(os.path.join(self.data_path,"reconstructed_objects", dir)):
            #         if file.startswith("mean"):
            #             continue
            #         data = np.load(os.path.join(self.data_path,"reconstructed_objects",dir,file))
            #         hist, bin_edges = np.histogram(data[data.nonzero()], bins = bins)
            #         axs[ind].stairs(hist, bin_edges, fill = True, label = file, alpha = alpha)
            #     axs[ind].legend()
            #     # axs[ind].xticks = [0.1*i for i in range(0, 16)]
            # plt.setp(axs, xticks=xticks)
            # plt.savefig(os.path.join(self.save_path, "reconstructed_values.png"))

    def find_yticks(self, folder):
        largest = 0
        for dir in os.listdir(os.path.join(self.data_path, folder)):
            for file in os.listdir(os.path.join(self.data_path, folder, dir)):
                if file.startswith("mean"):
                    continue
                data = np.load(os.path.join(self.data_path, folder, dir, file))
                hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.bins_rec_hist)
                if hist.max() > largest:
                    largest = hist.max()
        return np.arange(0,np.ceil(largest/10**(len(str(largest))-1))*10**(len(str(largest))-1), step = 10**(len(str(largest))-1))

    def find_xticks(self, folder):
        largest = 0
        bins = None
        for dir in os.listdir(os.path.join(self.data_path, folder)):
            for file in os.listdir(os.path.join(self.data_path, folder, dir)):
                if file.startswith("mean"):
                    continue
                data = np.load(os.path.join(self.data_path, folder, dir, file))
                hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.num_bins_hist)
                if bin_edges.max() > largest:
                    largest = bin_edges.max()
                    bins = bin_edges
        #Round up largest to the nearest of the second digit.
        largest = np.ceil(largest/10**(np.ceil(np.log10(largest)) - 2))*10**(np.ceil(np.log10(largest)) - 2)
        #To make sure that the printed ticks don't get to large, it should be nicely dividible by the number of ticks.
        #TODO: Implement this in a more general way. Should not be set here.
        nticks = 5
        while (largest/nticks)%10**(np.ceil(np.log10(largest/nticks)) - 2) != 0:
            largest += 10**(np.ceil(np.log10(largest)) - 2)
        return np.arange(0,largest,step = largest//nticks), bins
        # return np.arange(0,np.ceil(largest/10**(len(str(largest))-1))*10**(len(str(largest))-1), step = 0.2*10**(len(str(largest))-1)), bins

    def find_normalization_values(self, folder):
        value = 0
        for file in os.listdir(os.path.join(self.data_path, folder, "none")):
            if file.startswith("mean"):
                continue
            data = np.load(os.path.join(self.data_path, folder, "none", file))
            hist, bin_edges = np.histogram(data[data.nonzero()], bins = self.num_bins_hist)
            value += bin_edges[np.argmax(hist)]
        return value/len(os.listdir(os.path.join(self.data_path, folder, "none")))

    def gaussian_fit(self, bin_edges, counts):
        def gaussian(x, a, x0, sigma):
            return a*np.exp(-0.5*((x-x0)/sigma)**2)
        #Could in theory write this file when one knows how many objects there are, so that one can add gaussian together, but don't see any quick fixes.
        #Or potentially let it loop over a range of gaussians and find out what gives the best results, but might end up with some weird overfitting.
        #From https://stackoverflow.com/questions/35990467/fit-mixture-of-two-gaussian-normal-distributions-to-a-histogram-from-one-set-of/70907042#70907042
        def bimodal(x, a1, x01, sigma1, a2, x02, sigma2):
            return gaussian(x, a1, x01, sigma1) + gaussian(x, a2, x02, sigma2)
        mean_guess = bin_edges[np.argmax(counts)]
        #Should find magic value of 0.8 from description file.
        bin_edges = (bin_edges[1:] + bin_edges[:-1])/2
        #Magic 0.8 from density distribution.
        popt, pcov = curve_fit(bimodal, bin_edges, counts, p0=[counts.max(), mean_guess, mean_guess/self.num_bins_hist, counts[int(0.8*np.argmax(counts))], mean_guess*0.8, mean_guess/self.num_bins_hist])
        #Might even plot it, but just save for now.
        return popt, np.sqrt(np.diag(pcov))