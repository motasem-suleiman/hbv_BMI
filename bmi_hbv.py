'''
BMI code for the HBV (Hydrologiska byrÂns vattenavdelning) model.

This BMI code has been developed by Motasem S Abualqumboz, Utah State University (2022).
The code was developed using the BMI code developed by Jonathan Frame for the 
CFE (Conceptual Fuctional Eqvalent) model (https://github.com/jmframe/si_2022_train).

'''

#---------- Python libraries ----------#
import time
import numpy as np
import pandas as pd
import sys
import math
import json
import matplotlib.pyplot as plt
import hbv
import datetime

#---------- BMI class for the HBV model ----------#

class BMI_HBV():
    def __init__(self):
        """Create a Bmi HBV model that is ready for initialization."""
        super(BMI_HBV, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        
        #----------------------------------------------
        # Required, static attributes of the model
        #----------------------------------------------
        self._att_map = {
            'model_name':         'Hydrologiska Byråns Vattenbalansavdelning (HBV)',
            'version':            '1.0',
            'author_name':        'Motasem Suleiman Abualqumboz, Utah State University',
            'grid_type':          'scalar',
            'time_step_size':      1, 
            'time_units':         'd' }
    
        #---------------------------------------------
        # Input variable names (CSDMS standard names) (https://csdms.colorado.edu/wiki/CSN_Searchable_List-Names)
        #---------------------------------------------

        self._input_var_names = ['atmosphere_water__precipitation_mass_flux',
                                'atmosphere_air__temperature',
                                'earth_day']
        
        
    
        #---------------------------------------------
        # Output variable names (CSDMS standard names) (https://csdms.colorado.edu/wiki/CSN_Searchable_List-Names)
        #---------------------------------------------
        self._output_var_names = ['land_surface_water__potential_evaporation_volume_flux',
                                 'snow_fall',
                                 'catchment_water_input',
                                 'Actual_evaporation',
                                 'Soil_Storage',
                                 'shallow_flow',
                                 'flow_from_upper_storage',
                                 'flow_from_lower_storage',
                                 'storage_from_upper_GW_reservoir',
                                 'storage_from_lower_GW_reservoir',
                                 'unrouted_streamflow_through_channel_network',
                                 'total_storage',
                                 'Snowpack__liquid_equivalent_state_Variable',
                                 'Upper_zone_storage_state_Variable',
                                 'Lower_zone_storage_state_Variable',
                                 'Simulated_snowpack_state_Variable',
                                 'SP_liquied_water_state_Variable',
                                 'Soil_storage_state_Variable']
        
        #------------------------------------------------------
        # Create a Python dictionary that maps CSDMS Standard
        # Names to the model's internal variable names.
        # This is going to get long, 
        #     since the input variable names could come from any forcing...
        #------------------------------------------------------
        self._var_name_units_map = {
                                'atmosphere_water__precipitation_mass_flux':['timestep_rainfall_input_m','mm day-1'],
                                'atmosphere_air__temperature':['temperature', 'c'],
                                'earth_day':['DayOfYear','d'],
                                'land_surface_water__potential_evaporation_volume_flux':['average_watershed_potential_et','mm day-1'],
                                'snow_fall':['simulated_snowfall_SF', 'mm'],
                                'catchment_water_input':['catchment_input_inc', 'mm day-1'],
                                'Actual_evaporation':['average_watershed_actual_aet', 'mm day-1'],
                                'Soil_Storage':['soil', 'mm'],
                                'shallow_flow':['shallow_flow_Qstz', 'mm day-1'],
                                'flow_from_upper_storage':['flow_from_upper_storage_Qsuz', 'mm day-1'],
                                'flow_from_lower_storage':['flow_from_lower_storage_Qslz', 'mm day-1'],
                                'storage_from_upper_GW_reservoir':['storage_from_upper_GW_reservoir_S1', 'mm'],
                                'storage_from_lower_GW_reservoir':['storage_from_lower_GW_reservoir_S2', 'mm'],
                                'unrouted_streamflow_through_channel_network':['unrouted_streamflow_through_channel_network_Qgen', 'mm day-1'],
                                'total_storage':['total_storage_Storage', 'mm'],
                                'Snowpack__liquid_equivalent_state_Variable':['snow_water_equivalent_SWE', 'mm'],
                                'Upper_zone_storage_state_Variable':['upper_zone_storage_SUZ', 'mm'],
                                'Lower_zone_storage_state_Variable':['lower_zone_storage_SLZ', 'mm'],
                                'Simulated_snowpack_state_Variable':['simulated_snowpack_SP', 'mm'],
                                'SP_liquied_water_state_Variable':['liquid_water_in_snowpack_WC', 'mm'],
                                'Soil_storage_state_Variable':['soil_storage_content_SM', 'mm']
        }     


    #__________________________________________________________________
    #__________________________________________________________________
    # BMI: Model Control Function
    def initialize(self, cfg_file=None):

        #------------------------------------------------------------
        # this is the bmi configuration file
        self.cfg_file = cfg_file

        self.current_time_step = self._start_time

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for long_name in self._var_name_units_map.keys()}
        
        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        for long_var_name in list(self._var_name_units_map.keys()):
            # ---------- All the variables are single values ------------------#
            # ---------- so just set to zero for now.        ------------------#
            self._values[long_var_name] = 0
            setattr( self, self.get_var_name(long_var_name), 0 )  # The setattr function sets the value of the attribute of an object.

        ############################################################
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()                                    #
        
        self.load_forcing_file()
        
        
         # ________________________________________________
        # Time control
        self.time_step_size = 1
        self.timestep_h = self.time_step_size * 24.0
        self.timestep_s = self.timestep_h * 24.0 * 60.0 * 60.0
        self.current_time_step = 0
        self.current_time = pd.Timestamp(year=1981, month=1, day=1, hour=0)

        
        # ________________________________________________
        # Inputs
        self.temperature = 0.0
        self.DayOfYear   = 0.0
        self.timestep_rainfall_input_m = 0.0
        # ________________________________________________
        # output
        # Evaporation
        self.average_watershed_potential_et = 0.0

        
        # Snow routine
        self.simulated_snowfall_SF = 0.0
        self.catchment_input_inc = 0.0
        
        # soil routine
        self.average_watershed_actual_aet = 0.0
        self.soil=0.0
        
        # Reservoir
        self.shallow_flow_Qstz = 0.0
        self.flow_from_upper_storage_Qsuz = 0.0
        self.flow_from_lower_storage_Qslz = 0.0
        self.storage_from_upper_GW_reservoir_S1 = 0.0
        self.storage_from_lower_GW_reservoir_S2 = 0.0
        self.unrouted_streamflow_through_channel_network_Qgen = 0.0
        self.total_storage_Storage = 0.0


        
        # ________________________________________________
        # Evapotranspiration
        self.angular_velocityR = 0.0
        self.latitudeR    = 0.0
        
        #Snow Routine

        self.snowpack_melting_rate_melt  = 0.0
        self.refreeze = 0.0
        
        #soil routine
        self.old_soil_storage_content_oldSM = 0.0
        self.y = 0.0
        self.m = 0.0
        self.partitioning_function_dQdP = 0.0
        self.mean_storage_content_meanSM = 0.0
        self.recharge = 0.0
        
        # Model state variables
        self.snow_water_equivalent_SWE = 0.0                      # Initial snow water equivalent
        self.upper_zone_storage_SUZ = 0.0                         # Initial upper zone storage
        self.lower_zone_storage_SLZ = 0.0                         # Initial lower zone storage
        self.simulated_snowpack_SP = 152.4                        # Initial value for simulated snowpack
        self.liquid_water_in_snowpack_WC = 0.0                    # Initial liquid water in snowpack
        self.soil_storage_content_SM = self.field_capacity_FC     # Initial soil storage content


 
        #############################################################################
        # _________________________________________________________________________ #
        # _________________________________________________________________________ #
        # CREATE AN INSTANCE OF THE Hydrologiska Byråns Vattenbalansavdelning (HBV) #
        self.hbv_model = hbv.HBV()
        # _________________________________________________________________________ #
        # _________________________________________________________________________ #
        #############################################################################
        
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update(self):
        
        self.set_value('atmosphere_air__temperature', self.temp_input[self.current_time_step]) # set the temperature
        self.set_value('earth_day', self.dayOfYear_input[self.current_time_step]) # set the day of the year (1-365) for PET calculation
        self.set_value('atmosphere_water__precipitation_mass_flux', self.precip_input[self.current_time_step]) # set the precipitation value
        
        self.hbv_model.run_hbv(self)
        self.scale_output()

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update_until(self, until):
        for i in range(self.current_time_step, until):
            self.hbv_model.run_hbv(self)
            self.scale_output()

        
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def finalize(self):


        """Finalize model."""
        self.hbv_model = None
        self.hbv_state = None
    
   
    #________________________________________________________
    def config_from_json(self):
        with open(self.cfg_file) as data_file:
            data_loaded = json.load(data_file)

        # ___________________________________________________
        # MANDATORY CONFIGURATIONS
        self.forcing_file               = data_loaded['forcing_file']
        self.angular_velocity           = data_loaded['angular_velocity']
        self.latitude                   = data_loaded['latitude']
        
        # Snow
        self.threshold_temperature_TT           =  data_loaded['snow_parameters']['threshold_temperature_TT']
        self.snowfall_correction_factor_SFCF    =  data_loaded['snow_parameters']['snowfall_correction_factor_SFCF']
        self.snow_melt_degreeDay_factor_CFMAX   =  data_loaded['snow_parameters']['snow_melt_degreeDay_factor_CFMAX']
        self.Water_holding_capacity_CWH         =  data_loaded['snow_parameters']['Water_holding_capacity_CWH']
        self.refreezing_coefficient_CFR         =  data_loaded['snow_parameters']['refreezing_coefficient_CFR']
        
        # Soil moisture
        self.field_capacity_FC                  =  data_loaded['soil_parameters']['field_capacity_FC']
        self.evaporation_reduction_threshold_LP =  data_loaded['soil_parameters']['evaporation_reduction_threshold_LP']
        self.shape_coefficient_beta             =  data_loaded['soil_parameters']['shape_coefficient_beta']
        
        # Reservoir
        self.recession_constant_near_surface_K0  =  data_loaded['reservoir_parameters']['recession_constant_near_surface_K0']
        self.recession_constant_upper_storage_K1 =  data_loaded['reservoir_parameters']['recession_constant_upper_storage_K1']
        self.recession_constant_lower_storage_K2 =  data_loaded['reservoir_parameters']['recession_constant_lower_storage_K2']
        self.threshold_for_shallow_storage_UZL   =  data_loaded['reservoir_parameters']['threshold_for_shallow_storage_UZL']
        self.lower_to_upper_maxflow_Percolation  =  data_loaded['reservoir_parameters']['lower_to_upper_maxflow_Percolation']
        self.MAXBAS_routing                      =  data_loaded['routing_parameters']['MAXBAS']


        # ___________________________________________________
        # OPTIONAL CONFIGURATIONS

        if 'forcing_file' in data_loaded.keys():
            self.reads_own_forcing              = True
            self.forcing_file                   = data_loaded['forcing_file']
            
        return

       
    #________________________________________________________ 
    def load_forcing_file(self):
        # self.forcing_data = pd.read_csv(self.forcing_file)
        
        with open(self.forcing_file, 'r') as forcing_file:
            df_forcing = pd.read_csv(forcing_file)
            
            self.precip_input = df_forcing['P_mmday']
            self.temp_input = df_forcing['T_C']
            self.PET_input = df_forcing['PET_mmday']
            self.dayOfYear_input = df_forcing['Day']



        
    #------------------------------------------------------------ 
    def scale_output(self):
            
        self._values['land_surface_water__potential_evaporation_volume_flux'] = self.average_watershed_potential_et
        
        #snow routine
        
        self._values['snow_fall'] = self.simulated_snowfall_SF
        self._values['catchment_water_input'] = self.catchment_input_inc
        
        self._values['Actual_evaporation'] = self.average_watershed_actual_aet
        self._values['Soil_Storage'] = self.soil
        
        # Reservoir
        self._values['shallow_flow'] = self.shallow_flow_Qstz
        self._values['flow_from_upper_storage'] = self.flow_from_upper_storage_Qsuz
        self._values['flow_from_lower_storage'] = self.flow_from_lower_storage_Qslz
        self._values['storage_from_upper_GW_reservoir'] = self.storage_from_upper_GW_reservoir_S1
        self._values['storage_from_lower_GW_reservoir'] = self.storage_from_lower_GW_reservoir_S2
        self._values['unrouted_streamflow_through_channel_network'] = self.unrouted_streamflow_through_channel_network_Qgen
        self._values['total_storage'] = self.total_storage_Storage
        
        # Model State Variables
        self._values['Snowpack__liquid_equivalent_state_Variable'] = self.snow_water_equivalent_SWE
        self._values['Upper_zone_storage_state_Variable'] = self.upper_zone_storage_SUZ
        self._values['Lower_zone_storage_state_Variable'] = self.lower_zone_storage_SLZ
        self._values['Simulated_snowpack_state_Variable'] = self.simulated_snowpack_SP
        self._values['SP_liquied_water_state_Variable'] = self.liquid_water_in_snowpack_WC
        self._values['Soil_storage_state_Variable'] = self.soil_storage_content_SM       

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute( 'model_name' ) #JG Edit

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name)

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        # JG Edit
        return self.get_value_ptr(long_var_name)  #.dtype
    
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # JG Edit
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id  

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
#        return np.dtype(self.get_var_type(name)).itemsize
        return np.array(self.get_value(name)).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        # JG Edit
        # all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    #-------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return self._start_time #JG Edit

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return self._end_time #JG Edit


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.current_time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.get_attribute( 'time_step_size' ) #JG: Edit

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        setattr( self, self.get_var_name(var_name), value )
        self._values[var_name] = value

    #------------------------------------------------------------ 
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        # JG Note: TODO confirm this is correct. Get/set values ~=
#        val = self.get_value_ptr(name)
#        val.flat[inds] = src

        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(name)).flatten().shape[0] == 1:
            self.set_value(name, src)
        else:
            # JMFrame: Need to set the value with the updated array with new index value
            val = self.get_value_ptr(name)
            for i in inds.shape:
                val.flatten()[inds[i]] = src[i]
            self.set_value(name, val)

    #------------------------------------------------------------ 
    def get_var_nbytes(self, long_var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        # JMFrame NOTE: Had to import sys for this function
        return sys.getsizeof(self.get_value_ptr(long_var_name))

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(var_name)).flatten().shape[0] == 1:
            return self.get_value(var_name)
        else:
            val_array = self.get_value(var_name).flatten()
            return np.array([val_array[i] for i in indices])

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------ 
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
 
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0: 
            return 1

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
       
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id=0):

        # JG Edit
        # 0 is the only id we have        
        if grid_id == 0:
            return 'scalar'

    #------------------------------------------------------------ 
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    #------------------------------------------------------------ 
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    #------------------------------------------------------------ 
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 

