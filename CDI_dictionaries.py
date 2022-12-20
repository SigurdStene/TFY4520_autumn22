import numpy as np
grid_size = 256
number_of_reconstructions = 10
error_source_dict_12 = {'index': 'none' , 'beamstop': False, 'cylindrical_reciprocal_space': False}
error_source_dict_05 = {'index':  '05' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 5, "intensity_reduction_angle": 45}
error_source_dict_10 = {'index': '10' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 10, "intensity_reduction_angle": 45}
error_source_dict_15 = {'index': '15' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 15, "intensity_reduction_angle": 45}
error_source_dict_20 = {'index': '20' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 20, "intensity_reduction_angle": 45}
error_source_dict_30 = {'index': '30' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 30, "intensity_reduction_angle": 45}
error_source_dict_40 = {'index': '40' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 40, "intensity_reduction_angle": 45}
error_source_dict_50 = {'index': '50' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 50, "intensity_reduction_angle": 45}
error_source_dict_60 = {'index': '60' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 60, "intensity_reduction_angle": 45}
error_source_dict_01 = {'index': '1' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 1, "intensity_reduction_angle": 45}
error_source_dict_03 = {'index': '3' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": 3, "intensity_reduction_angle": 45}
lite = 0.1/100
error_source_dict_lite = {'index': 'lite' , 'beamstop': False, 'cylindrical_reciprocal_space': False, "intensity_reduction": True, "intensity_reduction_percentage": lite, "intensity_reduction_angle": 45}

error_source_dict_list = [error_source_dict_12,error_source_dict_05,error_source_dict_10,error_source_dict_15,error_source_dict_20,error_source_dict_30,error_source_dict_40,error_source_dict_50,error_source_dict_60]#,error_source_dict_01,error_source_dict_03,error_source_dict_lite]
# object_dict_0000000 = {'cube': {'center': (0.5, 0.5, 0.5), 'side_length': grid_size * 0.5, 'density': 1}, 'index': '0000000'}
#Obs, cannot have multiple with the same key. Should different objects be stored together as a list of densities etc. for each object?
object_dict_1000000 = {'ellipsoid_0': {'center': (0.57, 0.51, 0.61), 'half_axes': (grid_size * 0.25,grid_size * 0.32,grid_size * 0.37), 'density': 1}, 'index': '1000000',
                       'ellipsoid_1': {'center': (0.41, 0.43, 0.47), 'half_axes': (grid_size * 0.22,grid_size * 0.27,grid_size * 0.4), 'density': 0.8}, 'index': '1000000'}
object_dict_list = [object_dict_1000000] #object_dict_0000000,
