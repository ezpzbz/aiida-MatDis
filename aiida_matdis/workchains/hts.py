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

from aiida_matdis.aide_de_camp import (get_molecules_input_dict,
                                       get_ff_parameters,
                                       get_geometric_output,
                                       get_pressure_list,
                                       get_output_parameters,
                                       get_replciation_factors,
                                       update_workchain_params,
                                       dict_merge)

RaspaBaseWorkChain = WorkflowFactory('raspa.base')  #pylint: disable=invalid-name

# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name
ZeoppParameters = DataFactory("zeopp.parameters")  #pylint: disable=invalid-name

ZeoppCalculation = CalculationFactory("zeopp.network")  #pylint: disable=invalid-name
FFBuilder = CalculationFactory('matdis.ff_builder')
# Default parameters
HTSPARAMETERS_DEFAULT = Dict(
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
    possibly multi-components GCMC calculations using RASPA.
    """

    @classmethod
    def define(cls, spec):
        super(HTSWorkChain, cls).define(spec)

        spec.expose_inputs(ZeoppCalculation, namespace='zeopp', include=['atomic_radii','code', 'metadata'])
        spec.expose_inputs(RaspaBaseWorkChain, namespace='raspa_base', exclude=['raspa.structure', 'raspa.parameters'])
        spec.input('structure', valid_type=CifData, help='Adsorbent framework CIF.')
        spec.input("mixture",
                   valid_type=Dict,
                   help='A dictionary of components with their corresponding mol fractions in the mixture.')
        spec.input("parameters",
                   valid_type=Dict,
                   help='It provides the parameters which control the decision making behavior of workchain.')

        spec.input("geometric",
                   valid_type=Dict,
                   required=False,
                   help='[Only used by IsothermMultiTempWorkChain] Already computed geometric properties')

        spec.outline(
            cls.setup,
            cls.run_zeopp,
            cls.inspect_zeopp_calc,
            if_(cls.should_run_widom)(
                cls.run_raspa_widom,
                cls.inspect_widom_calc,
                if_(cls.should_run_gcmc)(
                    cls.run_raspa_gcmc
                ),
            ),
            cls.return_output_parameters,
        )

        spec.output('output_parameters',
                    valid_type=Dict,
                    required=True,
                    help='Results of the HTSWorkChain')

        spec.output_namespace('block_files',
                              valid_type=SinglefileData,
                              required=False,
                              dynamic=True,
                              help='Generated block pocket files for each probe if there are blocking spheres.')

    def setup(self):
        """Initialize parameters"""
        self.ctx.parameters = update_workchain_params(HTSPARAMETERS_DEFAULT, self.inputs.parameters)
        self.ctx.components = get_molecules_input_dict(self.inputs.mixture, self.ctx.parameters)

        self.ctx.temperature = int(round(self.ctx.parameters['temperature']))

        # Single-Temp - Standalone!
        if ("geometric" not in self.inputs) and (self.ctx.parameters['temperature_list'] is None):
            self.ctx.mode = 'standalone'
        # Geomonly for MultiTemp - So run zeo and skip the rest!
        elif ("geometric" not in self.inputs) and (self.ctx.parameters['temperature_list'] is not None):
            self.ctx.mode = 'geom_only'
        # Submitting it for several temp! sO SKIP zeo and run the rest.
        elif ("geometric" in self.inputs) and (self.ctx.parameters['temperature_list'] is None):
            self.ctx.mode = 'multi_temp'

        if self.ctx.mode == 'standalone' or self.ctx.mode =='multi_temp':
            self.ctx.ff_params = get_ff_parameters(self.ctx.parameters, molecule=None, components=self.ctx.components)

    def run_zeopp(self):
        """It performs the full zeopp calculation for all components."""
        if self.ctx.mode == 'standalone' or self.ctx.mode =='geom_only':
            zeopp_inputs = self.exposed_inputs(ZeoppCalculation, 'zeopp')
            dict_merge(
                zeopp_inputs, {
                    'metadata': {
                        'label': "ZeoppResSaVolpoBlock",
                    },
                    'structure': self.inputs.structure,
                    'atomic_radii': self.inputs.zeopp.atomic_radii,
                }
            )

            for key, value in self.ctx.components.get_dict().items():
                comp = value['name']
                dict_merge(
                    zeopp_inputs, {
                        'parameters': ZeoppParameters(dict=self.ctx.components[key]['zeopp']),
                        'metadata':{
                            'call_link_label': 'run_zeopp_' + comp
                        }
                    }
                )

                running = self.submit(ZeoppCalculation, **zeopp_inputs)
                zeopp_label = "zeopp_{}".format(comp)
                self.report("Running zeo++ res, sa, volpo, and block calculation<{}>".format(running.id))
                self.to_context(**{zeopp_label: running})


    def inspect_zeopp_calc(self):
        """Asserts whether all widom calculations are finished ok."""
        if self.ctx.mode == ('standalone' or 'geom_only'):
            for value in self.ctx.components.get_dict().values():
                assert self.ctx["zeopp_{}".format(value['name'])].is_finished_ok

    def should_run_widom(self):
        """Decided whether to run Henry coefficient calculation or not!"""
        self.ctx.should_run_widom = []
        if self.ctx.mode == 'standalone' or self.ctx.mode =='geom_only':
            self.ctx.geom = {}
            for value in self.ctx.components.get_dict().values():
                comp = value['name']
                zeopp_label = "zeopp_{}".format(comp)
                pld_lim = value["proberad"] * self.ctx.parameters["pld_scale"]
                self.ctx.geom[comp] = get_geometric_output(self.ctx[zeopp_label].outputs.output_parameters)
                if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                    self.out("block_files.{}_block_file".format(comp), self.ctx[zeopp_label].outputs.block)
            if self.ctx.mode == 'standalone':
                pld_component = self.ctx.geom[comp]["Largest_free_sphere"]
                lcd_component = self.ctx.geom[comp]["Largest_included_sphere"]
                poav_component = self.ctx.geom[comp]["POAV_A^3"]
                lcd_lim = self.ctx.parameters["lcd_max"]
                if (lcd_component <= lcd_lim) and (pld_component >= pld_lim) and (poav_component > 0.0):
                    self.ctx.should_run_widom.append(True)
                else:
                    self.ctx.should_run_widom.append(False)
            else:
                self.ctx.should_run_widom.append(False)
        elif self.ctx.mode == 'multi_temp':
            self.ctx.geom = self.inputs['geometric']['geometric_output']
            self.ctx.should_run_widom.append(True)

        return all(self.ctx.should_run_widom)


    def _get_widom_param(self):
        """Write Raspa input parameters from scratch, for a Widom calculation"""
        param = {
            "GeneralSettings": {
                "SimulationType": "MonteCarlo",
                "NumberOfInitializationCycles": 0,
                "NumberOfCycles": self.ctx.parameters['raspa_widom_cycles'],
                "PrintPropertiesEvery": self.ctx.parameters['raspa_widom_cycles'] / self.ctx.parameters['raspa_verbosity'],
                "PrintEvery": int(1e10),
                "RemoveAtomNumberCodeFromLabel": True,
                "Forcefield": "Local",
                "CutOff": self.ctx.parameters['ff_cutoff'],
            },
            "System": {
                self.inputs.structure.label: {
                    "type": "Framework",
                    "ExternalTemperature": self.ctx.parameters['temperature'],
                }
            },
            "Component": {},
        }
        repf = get_replciation_factors(self.inputs.structure, 2 * self.ctx.parameters['ff_cutoff'])
        param["System"][self.inputs.structure.label]["UnitCells"] = "{} {} {}".format(repf[0], repf[1], repf[2])
        return param

    def run_raspa_widom(self):
        """Run parallel Widom calculation in RASPA."""
        self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        self.ctx.raspa_inputs['metadata']['label'] = "RaspaWidom"
        self.ctx.raspa_inputs['raspa']['framework'] = {self.inputs.structure.label: self.inputs.structure}
        self.ctx.raspa_inputs['raspa']['file'] = FFBuilder(self.ctx.ff_params)
        self.ctx.blocks = self.ctx.raspa_inputs["raspa"]["block_pocket"]

        for value in self.ctx.components.get_dict().values():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_widom_{}".format(comp)
            self.ctx.raspa_params= self._get_widom_param()
            self.ctx.raspa_params["Component"][comp] = {}
            self.ctx.raspa_params["Component"][comp]["MoleculeDefinition"] = value['forcefield']
            self.ctx.raspa_params["Component"][comp]["WidomProbability"] = 1.0
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                self.ctx.raspa_params["Component"][comp]["BlockPocketsFileName"] = comp + "_block_file"
                if self.ctx.mode == 'standalone':
                    self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
                    self.ctx.raspa_inputs["raspa"]["block_pocket"] = {
                        comp + "_block_file": self.ctx[zeopp_label].outputs.block
                    }
                if self.ctx.mode == 'multi_temp':
                    self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
                    self.ctx.raspa_inputs["raspa"]["block_pocket"][comp + "_block_file"] = self.ctx.blocks[comp + "_block_file"]

            if value['charged']:
                self.ctx.raspa_params["GeneralSettings"].update({
                    "UseChargesFromCIFFile": "yes",
                    "ChargeMethod": "Ewald",
                    "EwaldPrecision": 1e-6
                })
            self.ctx.raspa_inputs['raspa']['parameters'] = Dict(dict=self.ctx.raspa_params)
            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            widom_label = "widom_{}".format(comp)
            self.report("Running Raspa Widom @ {}K for the Henry coefficient of <{}>".format(self.ctx.temperature, comp))
            self.to_context(**{widom_label: running})

    def inspect_widom_calc(self):
        """Asserts whether all widom calculations are finished ok."""
        for value in self.ctx.components.get_dict().values():
            assert self.ctx["widom_{}".format(value['name'])].is_finished_ok

    def should_run_gcmc(self):
        """
        Based on user-defined protocol, decides to run the GCMC calculation or not
        always: it skips the check and runs GCMC anyway!
        loose: it only checkes comp1 against comp2.
        tight: it checkes comp1 against all other componenets.
        """

        self.ctx.should_run_gcmc = []
        if self.ctx.parameters['run_gcmc_protocol'] == 'always':
            self.ctx.should_run_gcmc.append(True)

        if self.ctx.parameters['run_gcmc_protocol'] == 'loose':
            widom_label_comp1 = "widom_{}".format(self.ctx.components.get_dict()['comp1']['name'])
            widom_label_comp2 = "widom_{}".format(self.ctx.components.get_dict()['comp2']['name'])
            output1 = self.ctx[widom_label_comp1].outputs.output_parameters.get_dict()
            output2 = self.ctx[widom_label_comp2].outputs.output_parameters.get_dict()
            self.ctx.kh_comp1 = output1[self.inputs.structure.label]["components"][self.ctx.components.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            self.ctx.kh_comp2 = output2[self.inputs.structure.label]["components"][self.ctx.components.get_dict()['comp2']
                                                                     ['name']]["henry_coefficient_average"]
            self.ctx.ideal_selectivity = self.ctx.kh_comp1 / self.ctx.kh_comp2
            self.ctx.ideal_selectivity_threshold = self.ctx.parameters["ideal_selectivity_threshold"]
            if self.ctx.ideal_selectivity >= self.ctx.ideal_selectivity_threshold:
                self.report("Ideal selectivity is greater than threshold: compute the GCMC")
                self.ctx.should_run_gcmc.append(True)
            else:
                self.report("Ideal selectivity is less than threshold: DO NOT compute the GCMC")
                self.ctx.should_run_gcmc.append(False)

        if self.ctx.parameters['run_gcmc_protocol'] == 'tight':
            widom_label_comp1 = "widom_{}".format(self.ctx.components.get_dict()['comp1']['name'])
            output1 = self.ctx[widom_label_comp1].outputs.output_parameters.get_dict()
            self.ctx.kh_comp1 = output1[self.inputs.structure.label]["components"][self.ctx.components.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            for value in self.ctx.components.get_dict().values():
                comp = value['name']
                widom_label = "widom_{}".format(comp)
                output = self.ctx[widom_label].outputs.output_parameters.get_dict()
                self.ctx.kh_comp = output[self.inputs.structure.label]["components"][comp]["henry_coefficient_average"]
                self.ctx.ideal_selectivity = self.ctx.kh_comp1 / self.ctx.kh_comp
                if self.ctx.ideal_selectivity >= self.ctx.ideal_selectivity_threshold:
                    self.ctx.should_run_gcmc.append(True)
                else:
                    self.ctx.should_run_gcmc.append(False)

        return all(self.ctx.should_run_gcmc)

    def _update_param_input_for_gcmc(self):
        """Update Raspa inputs and parameters to run GCMC"""
        params = self.ctx.raspa_params
        inputs = self.ctx.raspa_inputs
        inputs["raspa"]["block_pocket"] = {}
        inputs["raspa"]["block_pocket"] = self.ctx.blocks
        params["GeneralSettings"].update({
            "NumberOfInitializationCycles": self.ctx.parameters['raspa_gcmc_init_cycles'],
            "NumberOfCycles": self.ctx.parameters['raspa_gcmc_prod_cycles'],
            "PrintPropertiesEvery": int(1e6),
            "PrintEvery": self.ctx.parameters['raspa_gcmc_prod_cycles'] / self.ctx.parameters['raspa_verbosity']
        })
        params["Component"] = {}
        params["Component"] = {item: {} for index, item in enumerate(list(self.ctx.components.get_dict()))}
        if self.ctx.mode == 'standalone':
            inputs["raspa"]["block_pocket"] = {}
        for key, value in self.ctx.components.get_dict().items():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            params["Component"][comp] = params["Component"].pop(key)
            params["Component"][comp].update({
                "MolFraction": value['molfraction'],
                "TranslationProbability": 1.0,
                "ReinsertionProbability": 1.0,
                "SwapProbability": 2.0,
                "IdentityChangeProbability": 2.0,
                "NumberOfIdentityChanges": len(list(self.ctx.components.get_dict())),
                "IdentityChangesList": [i for i in range(len(list(self.ctx.components.get_dict())))]
            })
            if not value['singlebead']:
                params["Component"][comp].update({"RotationProbability": 1.0})
            if value['charged']:
                params["GeneralSettings"].update({
                    "UseChargesFromCIFFile": "yes",
                    "ChargeMethod": "Ewald",
                    "EwaldPrecision": 1e-6
                })

            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                params["Component"][comp]["BlockPocketsFileName"] = {}
                params["Component"][comp]["BlockPocketsFileName"][self.inputs.structure.label] = comp + "_block_file"
                if self.ctx.mode == 'standalone':
                    inputs["raspa"]["block_pocket"][comp + "_block_file"] = self.ctx[zeopp_label].outputs.block

        return params, inputs

    def run_raspa_gcmc(self):
        """
        It submits Raspa calculation to RaspaBaseWorkchain.
        """

        self.ctx.current_p_index = 0
        self.ctx.pressures = get_pressure_list(self.ctx.parameters)
        self.report("<{}> pressure points are chosen for GCMC calculations".format(len(self.ctx.pressures)))

        self.ctx.raspa_params, self.ctx.raspa_inputs = self._update_param_input_for_gcmc()
        self.ctx.raspa_inputs['metadata']['description'] = 'Called by HTSWorkChain'

        for index, pressure in enumerate(self.ctx.pressures):
            self.ctx.raspa_inputs['metadata']['label'] ="RaspaGCMC_{}".format(index + 1)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gcmc_{}".format(index + 1)
            self.ctx.raspa_params["System"][self.inputs.structure.label]["ExternalPressure"] = self.ctx.pressures[index] * 1e5
            self.ctx.raspa_inputs['raspa']['parameters'] = Dict(dict=self.ctx.raspa_params)

            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            self.report("Running Raspa GCMC @ {}K/{:.3f}bar (pressure {} of {})".format(self.ctx.temperature, self.ctx.pressures[index], index + 1,len(self.ctx.pressures)))
            self.to_context(raspa_gcmc=append_(running))

    def inspect_raspa_gcmc_calc(self):
        """ """
        for workchain in self.ctx.raspa_gcmc:
            assert workchain.is_finished_ok

    def return_output_parameters(self):
        """Merge all the parameters into output_parameters, depending on is_porous and is_kh_ehough."""

        all_out_dict = {}

        if self.ctx.mode == 'geom_only':
            for value in self.ctx.components.get_dict().values():
                zeopp_label = "zeopp_{}".format(value['name'])
                all_out_dict[zeopp_label] = self.ctx.geom[value['name']]
                self.ctx.pressures = None
                self.ctx.components = None

        else:
            if all(self.ctx.should_run_widom):
                if self.ctx.mode == 'multi_temp':
                    all_out_dict['geometric'] = self.inputs['geometric']

                for value in self.ctx.components.get_dict().values():
                    widom_label = "widom_{}".format(value['name'])
                    zeopp_label = "zeopp_{}".format(value['name'])
                    all_out_dict[widom_label] = self.ctx[widom_label].outputs.output_parameters
                    if self.ctx.mode == 'standalone':
                        all_out_dict[zeopp_label] = self.ctx.geom[value['name']]
                if all(self.ctx.should_run_gcmc):
                    for workchain in self.ctx.raspa_gcmc:
                        all_out_dict[workchain.label] = workchain.outputs.output_parameters
                else:
                    self.ctx.pressures = None
            else:
                self.ctx.pressures = None
                self.ctx.components = None
        self.out(
            "output_parameters",
            get_output_parameters(wc_params=self.ctx.parameters,
                                  pressures=self.ctx.pressures,
                                  components=self.ctx.components,
                                  **all_out_dict))
        if self.ctx.mode == 'geom_only':
            self.report("HTSWorkChain completed in geom_only mode: ouput Dict<{}>".format(self.outputs['output_parameters'].pk))
        else:
            self.report("HTSWorkChain @ {}K completed: ouput Dict<{}>".format(self.ctx.temperature, self.outputs['output_parameters'].pk))

# EOF
