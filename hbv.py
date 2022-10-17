'''
Object-oriented Python code for the HBV (Hydrologiska byrÂns vattenavdelning) model.

This code has been developed by Motasem S Abualqumboz, Utah State University (2022).

The original R code of the HBV model was written by by J.P. Gannon. 
(https://github.com/VT-Hydroinformatics/16-Intro-Modeling-HBV)

Thanks to J. Seibert for sharing parts of HBV Light code.

This HBV code does not include elevation/vegetation zones.
'''

import time
import numpy as np
import pandas as pd
import math

class HBV():
    def __init__(self):
        super(HBV, self).__init__()
        
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_hbv(self, hbv_state):
        
        # ________________________________________________
        # calculate watershed-averaged potential evaporation
        self.average_watershed_et_mmDay(hbv_state)
        
        # ________________________________________________
        # Snow routine calculations
        self.snow_water_equivalent(hbv_state)
        
        # ________________________________________________
        # Soil routine calculations
        self.soil_moisture_routine(hbv_state)
        
        # ________________________________________________
        # Groundwater routine calculations
        self.ground_water_storage(hbv_state)

                                               
        #________________________________________________
        # time step
        hbv_state.current_time_step += 1
        hbv_state.current_time      += pd.Timedelta(value = hbv_state.time_step_size, unit='d')

        return
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # This fucntion calculate the daily-averaged Potnetial Evaporation at the watershed scale. 

    def average_watershed_et_mmDay(self, hbv_state):
        
        hbv_state.angular_velocityR = hbv_state.angular_velocity * math.pi / 180
        
        hbv_state.latitudeR = hbv_state.latitude * math.pi / 180
        
        DayAngle = (hbv_state.DayOfYear - 1) * 2 * math.pi / 365
        
        declinationR = (0.006918 - 0.399912 * math.cos(DayAngle) + 0.070257 * math.sin(DayAngle) -
                    0.006758 * math.cos(2 * DayAngle) + 0.000907 * math.sin(2 * DayAngle) - 0.002697 *
                    math.cos(3 * DayAngle) + 0.00148 * math.sin(3 * DayAngle))

        declinationD = declinationR * 180 / math.pi

        if abs(hbv_state.latitude - declinationD) < 90 and abs(hbv_state.latitude + declinationD) < 90:

            Tr = -1 * math.acos(-1 * math.tan(declinationR) * math.tan(hbv_state.latitudeR)) / hbv_state.angular_velocityR 
            Ts = math.acos(-1 * math.tan(declinationR) * math.tan(hbv_state.latitudeR)) / hbv_state.angular_velocityR 

        elif abs(hbv_state.latitude - declinationD) >= 90:
            Ts= 0
            Tr = Ts
        else:
            Ts = 12

        DayLenght = abs(Ts) + abs(Tr)
        SatVapour = 0.611 * math.exp(17.3 * 
                                     hbv_state.temperature /
                                     (hbv_state.temperature + 237.3))                        #6.2 (Pysical Hydrology,3rd edition), KPa
        PET = 29.8 * DayLenght * SatVapour / (hbv_state.temperature + 273.2)                 #6.68 (Pysical Hydrology,3rd edition), mm/day
        
        hbv_state.average_watershed_potential_et = PET
        
        return 

    
    
    # __________________________________________________________________________________________________________
    # Snow Routine

    def snow_water_equivalent(self, hbv_state):
        
    ## SNOW
        hbv_state.catchment_input_inc = 0 

        if hbv_state.simulated_snowpack_SP > 0:
            
            if hbv_state.timestep_rainfall_input_m > 0:
                if hbv_state.temperature > hbv_state.threshold_temperature_TT:
                    hbv_state.liquid_water_in_snowpack_WC += hbv_state.timestep_rainfall_input_m
                else:
                    hbv_state.simulated_snowpack_SP += (hbv_state.timestep_rainfall_input_m * hbv_state.snowfall_correction_factor_SFCF)
            if hbv_state.temperature > hbv_state.threshold_temperature_TT:
                hbv_state.snowpack_melting_rate_melt = hbv_state.snow_melt_degreeDay_factor_CFMAX *(hbv_state.temperature - hbv_state.threshold_temperature_TT)

                if hbv_state.snowpack_melting_rate_melt > hbv_state.simulated_snowpack_SP:
                    hbv_state.catchment_input_inc = hbv_state.simulated_snowpack_SP + hbv_state.liquid_water_in_snowpack_WC 
                    hbv_state.liquid_water_in_snowpack_WC  = 0 
                    hbv_state.simulated_snowpack_SP = 0 
                else:
                    hbv_state.simulated_snowpack_SP -= hbv_state.snowpack_melting_rate_melt        
                    hbv_state.liquid_water_in_snowpack_WC  += hbv_state.snowpack_melting_rate_melt   

                    if hbv_state.liquid_water_in_snowpack_WC  >= (hbv_state.Water_holding_capacity_CWH * hbv_state.simulated_snowpack_SP):   
                        hbv_state.catchment_input_inc = hbv_state.liquid_water_in_snowpack_WC - hbv_state.Water_holding_capacity_CWH * hbv_state.simulated_snowpack_SP  
                        hbv_state.liquid_water_in_snowpack_WC  = hbv_state.Water_holding_capacity_CWH * hbv_state.simulated_snowpack_SP 

            else:
                hbv_state.refreeze = hbv_state.refreezing_coefficient_CFR * hbv_state.snow_melt_degreeDay_factor_CFMAX * (hbv_state.threshold_temperature_TT - hbv_state.temperature)   

                if hbv_state.refreeze > hbv_state.liquid_water_in_snowpack_WC :
                    hbv_state.refreeze = hbv_state.liquid_water_in_snowpack_WC   

                hbv_state.simulated_snowpack_SP += hbv_state.refreeze   
                hbv_state.liquid_water_in_snowpack_WC  -= hbv_state.refreeze   
                hbv_state.simulated_snowfall_SF = hbv_state.timestep_rainfall_input_m * hbv_state.snowfall_correction_factor_SFCF  #Snowfall

        else:
            if hbv_state.temperature > hbv_state.threshold_temperature_TT:
                hbv_state.catchment_input_inc = hbv_state.timestep_rainfall_input_m   #If too warm, input is rain
            else:
                hbv_state.simulated_snowpack_SP = hbv_state.timestep_rainfall_input_m * hbv_state.snowfall_correction_factor_SFCF

        hbv_state.snow_water_equivalent_SWE = hbv_state.simulated_snowpack_SP + hbv_state.liquid_water_in_snowpack_WC       
        
      
        return
    
    # __________________________________________________________________________________________________________
    # Soil Routine
    def soil_moisture_routine(self, hbv_state):

        ## SOIL MOISTURE 
        hbv_state.recharge = 0 
        hbv_state.old_soil_storage_content_oldSM = hbv_state.soil_storage_content_SM

        if hbv_state.catchment_input_inc > 0:
            if hbv_state.catchment_input_inc < 1:
                hbv_state.y = hbv_state.catchment_input_inc
            else:
                hbv_state.m = math.floor(hbv_state.catchment_input_inc)  #loop through 1 mm increments
                hbv_state.y = hbv_state.catchment_input_inc- hbv_state.m

                for i in range(0,hbv_state.m):   #Loop for adding input to soil 1 mm at a time to avoid instability
                    hbv_state.partitioning_function_dQdP = (hbv_state.soil_storage_content_SM / hbv_state.field_capacity_FC) ** hbv_state.shape_coefficient_beta

                    if hbv_state.partitioning_function_dQdP > 1:
                        hbv_state.partitioning_function_dQdP = 1

                    hbv_state.soil_storage_content_SM += (1 - hbv_state.partitioning_function_dQdP) 
                    hbv_state.recharge += hbv_state.partitioning_function_dQdP 

            hbv_state.partitioning_function_dQdP = (hbv_state.soil_storage_content_SM / hbv_state.field_capacity_FC) ** hbv_state.shape_coefficient_beta

            if hbv_state.partitioning_function_dQdP > 1:
                hbv_state.partitioning_function_dQdP = 1

            hbv_state.soil_storage_content_SM += ((1 - hbv_state.partitioning_function_dQdP) * hbv_state.y)     
            hbv_state.recharge += (hbv_state.partitioning_function_dQdP * hbv_state.y)           


        hbv_state.mean_storage_content_meanSM = (hbv_state.soil_storage_content_SM + hbv_state.old_soil_storage_content_oldSM) / 2
        if hbv_state.mean_storage_content_meanSM < (hbv_state.evaporation_reduction_threshold_LP * hbv_state.field_capacity_FC):
            hbv_state.average_watershed_actual_aet = hbv_state.average_watershed_potential_et * hbv_state.mean_storage_content_meanSM / (hbv_state.evaporation_reduction_threshold_LP*hbv_state.field_capacity_FC)

        else:
            hbv_state.average_watershed_actual_aet = hbv_state.average_watershed_potential_et

        if hbv_state.simulated_snowpack_SP + hbv_state.liquid_water_in_snowpack_WC > 0:
            hbv_state.average_watershed_actual_aet = 0       #No evap if snow present (SFCF accounts for this and catch error)

        hbv_state.soil_storage_content_SM -= hbv_state.average_watershed_actual_aet      #Update soil moisture with AET flux

        if hbv_state.soil_storage_content_SM < 0:
            hbv_state.soil_storage_content_SM = 0
            hbv_state.soil = 0
            
        hbv_state.soil = hbv_state.soil_storage_content_SM
            
        return
    
    # __________________________________________________________________________________________________________
    # groundwater Routine

    def ground_water_storage(self, hbv_state):

        hbv_state.upper_zone_storage_SUZ += hbv_state.recharge     
        
        if (hbv_state.upper_zone_storage_SUZ - hbv_state.lower_to_upper_maxflow_Percolation) < 0:
            hbv_state.lower_zone_storage_SLZ += hbv_state.upper_zone_storage_SUZ 
            hbv_state.upper_zone_storage_SUZ = 0 
        else:
            hbv_state.lower_zone_storage_SLZ += hbv_state.lower_to_upper_maxflow_Percolation
            hbv_state.upper_zone_storage_SUZ -= hbv_state.lower_to_upper_maxflow_Percolation

        if hbv_state.upper_zone_storage_SUZ < hbv_state.threshold_for_shallow_storage_UZL:
            hbv_state.shallow_flow_Qstz = 0
        else:
            hbv_state.shallow_flow_Qstz = (hbv_state.upper_zone_storage_SUZ - hbv_state.threshold_for_shallow_storage_UZL) * hbv_state.recession_constant_near_surface_K0

        hbv_state.flow_from_upper_storage_Qsuz = hbv_state.upper_zone_storage_SUZ * hbv_state.recession_constant_upper_storage_K1 
        hbv_state.flow_from_lower_storage_Qslz = hbv_state.lower_zone_storage_SLZ * hbv_state.recession_constant_lower_storage_K2
        hbv_state.upper_zone_storage_SUZ =hbv_state.upper_zone_storage_SUZ - hbv_state.flow_from_upper_storage_Qsuz - hbv_state.shallow_flow_Qstz
        hbv_state.lower_zone_storage_SLZ -= hbv_state.flow_from_lower_storage_Qslz
        hbv_state.storage_from_upper_GW_reservoir_S1    = hbv_state.upper_zone_storage_SUZ 
        hbv_state.storage_from_lower_GW_reservoir_S2    = hbv_state.lower_zone_storage_SLZ 

        ##DISCHARGE COMPONENTS
        hbv_state.unrouted_streamflow_through_channel_network_Qgen = hbv_state.shallow_flow_Qstz + hbv_state.flow_from_upper_storage_Qsuz + hbv_state.flow_from_lower_storage_Qslz

        ## OUTPUT STORAGE
        hbv_state.total_storage_Storage = hbv_state.storage_from_upper_GW_reservoir_S1 + hbv_state.storage_from_lower_GW_reservoir_S2 + hbv_state.soil
        
        
        return
    
    
            ##--------- END TIME LOOP ------------







