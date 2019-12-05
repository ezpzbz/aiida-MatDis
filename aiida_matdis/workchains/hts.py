# -*- coding: utf-8 -*-
"""HTSWorkChain."""
from __future__ import absolute_import
from __future__ import print_function
import os
from six.moves import range

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Dict, List, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_
import aiida_lsmo.calcfunctions.ff_builder_module as FFBuilder

from aiida_matdis.aide_de_camp import (get_molecules_input_dict,
                                       get_ff_parameters,
                                       get_geometric_output,
                                       get_pressure_list,
                                       get_output_parameters,
                                       get_replciation_factors,
                                       update_workchain_params)

GACMWorkChain = WorkflowFactory('matdis.gacm')  #pylint: disable=invalid-name
WidomGCMCWorkChain = WorkflowFactory('matdis.widom_gcmc')  #pylint: disable=invalid-name

# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name

# Default parameters
WC_PARAMS_DEFAULT = Dict(
    dict={  #TODO: create IsothermParameters instead of Dict # pylint: disable=fixme
        "ff_framework": "UFF",  # str, Forcefield of the structure (used also as a definition of ff.rad for zeopp)
        "ff_shifted": False,  # bool, Shift or truncate at cutoff
        "ff_tail_corrections": True,  # bool, Apply tail corrections
        "ff_mixing_rule": 'Lorentz-Berthelot',  # str, Mixing rule for the forcefield
        "ff_separate_interactions": False,  # bool, if true use only ff_framework for framework-molecule interactions
        "ff_cutoff": 12.0,  # float, CutOff truncation for the VdW interactions (Angstrom)
        "temperature": 300,  # float, Temperature of the simulation
        "zeopp_volpo_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_sa_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_block_samples": int(100),  # int, Number of samples for BLOCK calculation (per A^3)
        "zeopp_accuracy": 'DEF', #Zeopp default when it is -ha
        "raspa_verbosity": 10,  # int, Print stats every: number of cycles / raspa_verbosity
        "raspa_widom_cycles": int(1e5),  # int, Number of widom cycles
        "raspa_gcmc_init_cycles": int(1e3),  # int, Number of GCMC initialization cycles
        "raspa_gcmc_prod_cycles": int(1e4),  # int, Number of GCMC production cycles
        "lcd_max": 15.0,  # Maximum allowed LCD.
        "pld_scale": 1.0,  # Scaling factor for minimum allowed PLD.
        "pressure_list": None,  # list, Pressure list for the isotherm (bar): if given it will skip  guess
        "temperature_list": None,  # list, Pressure list for the isotherm (bar): if given it will skip  guess
        "ideal_selectivity_threshold": 1.0,  #mandatory if protocol is relative.
        "run_gcmc_protocol": 'always',  # always, loose, and tight!
        "run_zeopp": True, #We will set it to false when it is called from MultiTemp
    })

class HTSWorkChain(WorkChain):
    """
    HTSWorkChain computes pore diameter, surface area, pore volume,
    and block pockets based on provided mixture composition and based on these
    results decides to run Henry coefficient calculations and
    possibly multi-molecules GCMC calculations using RASPA.
    """

    @classmethod
    def define(cls, spec):
        super(HTSWorkChain, cls).define(spec)

        spec.expose_inputs(GACMWorkChain)
        spec.expose_inputs(WidomGCMCWorkChain)

        # spec.input('structure', valid_type=CifData, help='Adsorbent framework CIF.')
        # #
        # spec.input("molecules",
        #            valid_type=Dict,
        #            help='A dictionary of molecules with their corresponding mol fractions in the mixture.')
        #
        # spec.input("parameters",
        #            valid_type=Dict,
        #            help='It provides the parameters which control the decision making behavior of workchain.')

        spec.outline(
            cls.setup,
            cls.run_gacm_workchain,
            cls.run_widom_raspa_workchain,
            cls.return_results,
        )

        spec.expose_outputs(GACMWorkChain)
        spec.expose_outputs(WidomGCMCWorkChain)


    def setup(self):
        """Initialize parameters"""
        self.ctx.parameters = update_workchain_params(WC_PARAMS_DEFAULT, self.inputs.parameters)
        self.ctx.molecules = self.exposed_inputs(GACMWorkChain)['molecules']
        # self.ctx.molecules = get_molecules_dict(self.inputs.molecules, self.ctx.parameters)
        # self.ctx.ff_params = get_ff_parameters(self.ctx.parameters, molecule=None, components=self.ctx.molecules)
        # self.ctx.temperatures = self.ctx.parameters['temperature_list']

    def run_gacm_workchain(self):
        """ """
        # inps['structure'] = self.inputs.structure
        # inps['molecules'] = self.inputs.molecules
        # inps['parameters'] = self.ctx.parameters
        # inps['zeopp']['zeopp.code'] = self.inputs

        inps = self.exposed_inputs(GACMWorkChain)
        # print(inps)
        # inps.update({
        #     'structure':self.inputs.structure,
        #     'molecules': self.ctx.molecules,
        #     'parameters': self.ctx.parameters,
        # })
        running = self.submit(GACMWorkChain, **inps)
        return ToContext(gamc=running)

    def run_widom_raspa_workchain(self):
        """ """
        inps = self.exposed_inputs(WidomGCMCWorkChain)
        # inps['structure'] = self.inputs.structure
        # inps['molecules'] = self.ctx.molecules
        # inps['parameters'] = self.ctx.parameters
        inps['geometric_data'] = self.ctx.gamc.outputs['geometric_output']
        comps = list(self.ctx.gamc.outputs['geometric_output'].get_dict().keys())
        # for comp in comps:
        #     if self.ctx.gamc.outputs['geometric_output'].get_dict()[comp]['Number_of_blocking_spheres'] > 0:
        #         inps['block_files'][comp + '_block_files'] = self.ctx.gamc.outputs['block_files__' + comp + '_block_file']

        running = self.submit(WidomGCMCWorkChain, **inps)
        return ToContext(raspa=running)

    def return_results(self):
        """Merge all the parameters into output_parameters, depending on is_porous and is_kh_ehough."""
        # What are we going to output here?
        # HTS outdic for each performed temperature? NO!
        self.out_many(self.exposed_outputs(self.ctx.gamc, GACMWorkChain))
        self.out_many(self.exposed_outputs(self.ctx.raspa, WidomGCMCWorkChain))
        self.report("WorkChain has been completed successfully!")

# EOF
