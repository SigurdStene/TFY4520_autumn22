#Create file containing a dictionary that can be accessed from different parts of the code to be made additions to
import os
import time

class Description:
    
        def __init__(self, filename_prefix):
            self.filename_prefix = filename_prefix  
            current_path = os.getcwd()
            self.save_path = os.path.join(current_path, "description_files")
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)
                time.sleep(1)
            
            # self.grid_size = grid_size
            # self.size = size
            # self.number_of_reconstructions = number_of_reconstructions
            # self.error_source_dict = error_source_dict
            self.description = {}

        def save_description(self):
            filename = os.path.join(self.save_path,f"{self.filename_prefix}_description.txt")
            if os.path.isfile(filename):
                print(f"Description file {filename} already exists. Overwriting.")
            with open(filename, "w") as f:
                f.write(f"Description file for {self.filename_prefix}\n")
                for key, value in self.description.items():
                    f.write(f"{key}: {value}\n")

        def add_description(self, parameter, parameter_value):
            self.description[parameter] = parameter_value
        
        def get_description(self, parameter):
            return self.description[parameter]