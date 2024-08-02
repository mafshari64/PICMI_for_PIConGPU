'''
PICMI user script reproducing the PIConGPU LWFA example

This Python script is example PICMI user script reproducing the LaserWakefield example setup(https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/share/picongpu/examples/LaserWakefield) 

Intended to be used with PIConGPU (https://github.com/brian/picongpu/tree/topic-addIonizerSupport)

authors: Masoud Afshari, Brian Marre.

# changes (31.07.2024):
 - uses new PICMI configuration of ionization and moving_window
'''

# Recreation of of the PIConGPU LWFA simulation based on .param files of
# https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/share/picongpu/examples/LaserWakefield/include/picongpu/param
# and  8.cfg file:
# https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/share/picongpu/examples/LaserWakefield/etc/picongpu/8.cfg

from picongpu import picmi
from picongpu import pypicongpu
import numpy as np

from picmi.interaction.ionization.fieldionization import ADK, ADKVariant
from picmi.interaction.ionization.fieldIonization import BSIEffectiveZ, BSIExtension
import scipy.constants as constants


##### Input parameters

# Enable or disable IONIZATION based on speciesInitialization.param
ENABLE_IONS = True
ENABLE_IONIZATION = True

numberCells = np.array([192 , 2048, 192])
cellSize = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter)

# Define the simulation grid based on grid.param
grid = picmi.Cartesian3DGrid(
    picongpu_n_gpus=[2,4,1],
    number_of_cells=numberCells.tolist(),  
    lower_bound=[0, 0, 0],
    upper_bound=(numberCells * cellSize).tolist(),
    lower_boundary_conditions =["open", "open", "open"], 
    upper_boundary_conditions =["open", "open", "open"])

gaussianProfile = picmi.distribution.GaussianDistribution(
    density=  1.e25
    center_front=8.0e-5,
    sigma_front= 8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor= -1.0,
    power=4.0,
    vacuum_cells_front=50
)

# for particle type see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_SpeciesType.md
electrons = picmi.Species(
    particle_type='electron',
    name='electron',
    initial_distribution=gaussianProfile)

hydrogen_ionization = picmi.Species(
    particle_type='H',
    name='hydrogen',
    charge_state=0,
    initial_distribution=gaussianProfile)

hydrogen_fully_ionized = picmi.Species(
    particle_type='H',
    name='hydrogen',
    picongpu_fixed_charge = True,
    initial_distribution=gaussianProfile)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method='Yee',
)

laser = picmi.GaussianLaser(
    wavelength= 0.8e-6,
    waist= 5.0e-6 / 1.17741,
    duration= 5.e-15,
    propagation_direction= [0., 1., 0.],
    polarization_direction= [1., 0., 0.],
    focal_position= [float(numberCells[0]*cellSize[0]/2.), 4.62e-5, float(numberCells[2]*cellSize[2]/2.)],
    centroid_position=[float(numberCells[0]*cellSize[0]/2.), 0., float(numberCells[2]*cellSize[2]/2.)],
    picongpu_polarization_type = pypicongpu.laser.GaussianLaser.PolarizationType.CIRCULAR,
    a0=8.0,
    picongpu_phase = 0.0
)

randomLayout=picmi.PseudoRandomLayout(n_macroparticles_per_cell = 2)

# Initialize particles  based on speciesInitialization.param
# simulation schema : https://github.com/BrianMarre/picongpu/blob/2ddcdab4c1aca70e1fc0ba02dbda8bd5e29d98eb/share/picongpu/pypicongpu/schema/simulation.Simulation.json

if not ENABLE_IONIZATION:
    hydrogen = hydrogen_fully_ionized
    interaction = None
else:
    hydrogen = hydrogen_ionization
    adk_ionization_model = ADK(
        ADK_variant = ADKVariant.CircularPolarization,
        ion_species = hydrogen_ionization,
        ionization_electron_species=electrons,
        ionization_current = None)

    bsi_effectiveZ_ionization_model = BSI(
        BSI_extension = [BSIExtension.EffectiveZ],
        ion_species = hydrogen_ionization,
        ionization_electron_species=electrons,
        ionization_current = None)

    interaction = Interaction(ground_state_ionizaion_model_list=[adk_ionization_model, bsi_effectiveZ_ionization_model])

sim = picmi.Simulation(
    solver=solver,
    max_steps= 4000,
    time_step_size=1.39e-16,
    picongpu_moving_window_move_point=0.9,
    picongpu_interaction=interaction)

sim.add_species(electrons, layout=randomLayout)

if ENABLE_IONS:
    sim.add_species(hydrogen, layout=randomLayout)

sim.add_laser(laser, None)

# adding additional non standardized input
# for generating setup with custom input see standard implementation,
#  see https://picongpu.readthedocs.io/en/latest/usage/picmi/custom_template.html

min_weight_input = pypicongpu.customuserinput.CustomUserInput()
min_weight_input.addToCustomInput({"minimum_weight": 10.0}, "minimum_weight")
sim.picongpu_add_custom_user_input(min_weight_input)

output_configuration = pypicongpu.customuserinput.CustomUserInput()
output_configuration.addToCustomInput(
    {
        "png_plugin_data_list": "['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Jx', 'Jy', 'Jz']",
        "png_plugin_SCALE_IMAGE": 1.0,
        "png_plugin_SCALE_TO_CELLSIZE": True,
        "png_plugin_WHITE_BOX_PER_GPU": False,
        "png_plugin_EM_FIELD_SCALE_CHANNEL1": 7,
        "png_plugin_EM_FIELD_SCALE_CHANNEL2": -1,
        "png_plugin_EM_FIELD_SCALE_CHANNEL3": -1,
        "png_plugin_CUSTOM_NORMALIZATION_SI": [5.0e12 / constants.c, 5.0e12, 15.0],
        "png_plugin_PRE_PARTICLE_DENS_OPACITY": 0.25,
        "png_plugin_PRE_CHANNEL1_OPACITY": 1.0,
        "png_plugin_PRE_CHANNEL2_OPACITY": 1.0,
        "png_plugin_PRE_CHANNEL3_OPACITY": 1.0,
        "png_plugin_preParticleDensCol": "colorScales::grayInv",
        "png_plugin_preChannel1Col": "colorScales::green",
        "png_plugin_preChannel2Col": "colorScales::none",
        "png_plugin_preChannel3Col": "colorScales::none",
        "png_plugin_preChannel1": "field_E.x() * field_E.x();",
        "png_plugin_preChannel2": "field_E.y()",
        "png_plugin_preChannel3": "-1.0_X * field_E.y()",
        "png_plugin_period":100,
        "png_plugin_axis": "yx",
        "png_plugin_slicePoint": 0.5,
        "png_plugin_species_name": "electron",
        "png_plugin_folder_name": "pngElectronsYX"
    },
    "png plugin configuration")
output_configuration.addToCustomInput(
    {
        "energy_histogram_species_name": "electron",
        "energy_histogram_period": 100,
        "energy_histogram_bin_count": 1024,
        "energy_histogram_min_energy": 0.,
        "energy_histogram_maxEnergy": 1000.,
        "energy_histogram_filter": "all"
    }
    "energy histogram")

output_configuration.addToCustomInput(
    {
        "phase_space_species_name": "electron",
        "phase_space_period": 100,
        "phase_space_space": "y",
        "phase_space_momentum": "py",
        "phase_space_min": -1.,
        "phase_space_max": 1.,
        "phase_space_filter": "all"
    }
    "phase space")

output_configuration.addToCustomInput(
    {
        "opnePMD_period": 100,
        "opnePMD_file": "simData",
        "opnePMD_extension": "bp"
    }
    "openPMD")

output_configuration.addToCustomInput(
    {
        "checkpoint_period": 100,
        "checkpoint_backend": "openPMD",
        "checkpoint_restart_backend": "openPMD"
    }
    "checkpoint")

output_configuration.addToCustomInput(
    {
        "macro_particle_count_period": 100,
        "macro_particle_count_species_name": "electron"
    }
    "macro particle count")
sim.picongpu_add_custom_user_input(output_configuration)

sim.write_input_file("masoud_lwfa_ionization")
