# -*- coding: utf-8 -*-
"""WidomGCMCWorkChain."""
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

RaspaBaseWorkChain = WorkflowFactory('raspa.base')  #pylint: disable=invalid-name
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
        "raspa_verbosity": 10,  # int, Print stats every: number of cycles / raspa_verbosity
        "raspa_widom_cycles": int(1e5),  # int, Number of widom cycles
        "raspa_gcmc_init_cycles": int(1e3),  # int, Number of GCMC initialization cycles
        "raspa_gcmc_prod_cycles": int(1e4),  # int, Number of GCMC production cycles
        "lcd_max": 15.0,  # Maximum allowed LCD.
        "pld_scale": 1.0,  # Scaling factor for minimum allowed PLD.
        "pressure_list": None,  # list, Pressure list for the isotherm (bar): if given it will skip  guess
        "ideal_selectivity_threshold": 1.0,  #mandatory if protocol is relative.
        "run_gcmc_protocol": 'always',  # always, loose, and tight!
    })

class WidomGCMCWorkChain(WorkChain):
    """
    WidomGCMCWorkChain computes pore diameter, surface area, pore volume,
    and block pockets based on provided mixture composition and based on these
    results decides to run Henry coefficient calculations and
    possibly multi-molecules GCMC calculations using RASPA.
    """

    @classmethod
    def define(cls, spec):
        super(WidomGCMCWorkChain, cls).define(spec)

        spec.expose_inputs(RaspaBaseWorkChain, namespace='raspa_base', exclude=['raspa.structure', 'raspa.parameters'])
        spec.input('structure', valid_type=CifData, help='Adsorbent framework CIF.')
        spec.input("molecules",
                   valid_type=Dict,
                   required=True,
                   help='A dictionary of molecules with their corresponding mol fractions in the mixture.')
        spec.input("parameters",
                   valid_type=Dict,
                   help='It provides the parameters which control the decision making behavior of workchain.')

        spec.input("geometric_data",
                   valid_type=Dict,
                   required=False,
                   help='[Only used by IsothermMultiTempWorkChain] Already computed geometric properties')

        spec.input_namespace("block_files",
                              valid_type=SinglefileData,
                              required=False,
                              dynamic=True,
                              help='Calculated block pocket files from zeopp workchain!')
        spec.outline(
            cls.setup,
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
                    help='Results of the WidomGCMCWorkChain')

    def setup(self):
        """Initialize parameters"""
        self.ctx.parameters = update_workchain_params(WC_PARAMS_DEFAULT, self.inputs.parameters)
        self.ctx.molecules = get_molecules_input_dict(self.inputs.molecules, self.ctx.parameters)
        self.ctx.ff_params = get_ff_parameters(self.ctx.parameters, molecule=None, components=self.ctx.molecules)
        self.ctx.temperature = int(round(self.ctx.parameters['temperature']))
        self.ctx.geom = self.inputs.geometric_data

    def should_run_widom(self):
        """Decided whether to run Henry coefficient calculation or not!"""
        self.ctx.should_run_widom = []
        # self.ctx.geom = {}
        lcd_lim = self.ctx.parameters["lcd_max"]
        for value in self.ctx.molecules.get_dict().values():
            comp = value['name']
            # zeopp_label = "zeopp_{}".format(comp)
            pld_lim = value["proberad"] * self.ctx.parameters["pld_scale"]
            # self.ctx.geom[comp] = get_geometric_output(self.ctx[zeopp_label].outputs.output_parameters)
            pld_component = self.ctx.geom[comp]["Largest_free_sphere"]
            lcd_component = self.ctx.geom[comp]["Largest_included_sphere"]
            poav_component = self.ctx.geom[comp]["POAV_A^3"]
            if (lcd_component <= lcd_lim) and (pld_component >= pld_lim) and (poav_component > 0.0):
                # self.report("Found {} blocking spheres".format(self.ctx.geom[comp]['Number_of_blocking_spheres']))
                # if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                    # self.out("block_files.{}_block_file".format(comp), self.ctx[zeopp_label].outputs.block)
                self.ctx.should_run_widom.append(True)
            else:
                self.ctx.should_run_widom.append(False)
        if all(self.ctx.should_run_widom):
            self.report("ALL pre-selection conditions are satisfied: Calculate Henry coefficients")
        else:
            self.report("All/Some of pre-selection criteria are NOT met: terminate!")
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
        self.ctx.raspa_inputs['raspa']['file'] = FFBuilder.ff_builder(self.ctx.ff_params)


        for value in self.ctx.molecules.get_dict().values():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_widom_{}".format(comp)
            self.ctx.raspa_params= self._get_widom_param()
            self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
            self.ctx.raspa_params["Component"][comp] = {}
            self.ctx.raspa_params["Component"][comp]["MoleculeDefinition"] = value['forcefield']
            self.ctx.raspa_params["Component"][comp]["WidomProbability"] = 1.0
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                if not "block_pocket" in self.ctx.raspa_inputs["raspa"]:
                    self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
                self.ctx.raspa_inputs["raspa"]["block_pocket"] = {
                    # comp + "_block_file": self.inputs.block_files['block_files__' + comp + '_block_file']
                    comp + "_block_file": self.inputs.block_files[comp + '_block_file']
                }
                self.ctx.raspa_params["Component"][comp]["BlockPocketsFileName"] = comp + "_block_file"
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
        for value in self.ctx.molecules.get_dict().values():
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
            widom_label_comp1 = "widom_{}".format(self.ctx.molecules.get_dict()['comp1']['name'])
            widom_label_comp2 = "widom_{}".format(self.ctx.molecules.get_dict()['comp2']['name'])
            output1 = self.ctx[widom_label_comp1].outputs.output_parameters.get_dict()
            output2 = self.ctx[widom_label_comp2].outputs.output_parameters.get_dict()
            self.ctx.kh_comp1 = output1[self.inputs.structure.label]["molecules"][self.ctx.molecules.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            self.ctx.kh_comp2 = output2[self.inputs.structure.label]["molecules"][self.ctx.molecules.get_dict()['comp2']
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
            widom_label_comp1 = "widom_{}".format(self.ctx.molecules.get_dict()['comp1']['name'])
            output1 = self.ctx[widom_label_comp1].outputs.output_parameters.get_dict()
            self.ctx.kh_comp1 = output1[self.inputs.structure.label]["molecules"][self.ctx.molecules.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            for value in self.ctx.molecules.get_dict().values():
                comp = value['name']
                widom_label = "widom_{}".format(comp)
                output = self.ctx[widom_label].outputs.output_parameters.get_dict()
                self.ctx.kh_comp = output[self.inputs.structure.label]["molecules"][comp]["henry_coefficient_average"]
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
        params["GeneralSettings"].update({
            "NumberOfInitializationCycles": self.ctx.parameters['raspa_gcmc_init_cycles'],
            "NumberOfCycles": self.ctx.parameters['raspa_gcmc_prod_cycles'],
            "PrintPropertiesEvery": int(1e6),
            "PrintEvery": self.ctx.parameters['raspa_gcmc_prod_cycles'] / self.ctx.parameters['raspa_verbosity']
        })
        params["Component"] = {}
        params["Component"] = {item: {} for index, item in enumerate(list(self.ctx.molecules.get_dict()))}
        inputs["raspa"]["block_pocket"] = {}
        for key, value in self.ctx.molecules.get_dict().items():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            params["Component"][comp] = params["Component"].pop(key)
            params["Component"][comp].update({
                "MolFraction": value['molfraction'],
                "TranslationProbability": 1.0,
                "ReinsertionProbability": 1.0,
                "SwapProbability": 2.0,
                "IdentityChangeProbability": 2.0,
                "NumberOfIdentityChanges": len(list(self.ctx.molecules.get_dict())),
                "IdentityChangesList": [i for i in range(len(list(self.ctx.molecules.get_dict())))]
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
                inputs["raspa"]["block_pocket"][comp + "_block_file"] = self.inputs.block_files[comp + '_block_file']
                # inputs["raspa"]["block_pocket"][comp + "_block_file"] = self.inputs.block_files['block_files__' + comp + '_block_file']

        return params, inputs

    def run_raspa_gcmc(self):
        """
        It submits Raspa calculation to RaspaBaseWorkchain.
        """

        # self.ctx.current_p_index = 0
        self.ctx.pressures = get_pressure_list(self.ctx.parameters)
        self.report("<{}> pressure points are chosen for GCMC calculations".format(len(self.ctx.pressures)))

        self.ctx.raspa_params, self.ctx.raspa_inputs = self._update_param_input_for_gcmc()
        self.ctx.raspa_inputs['metadata']['description'] = 'Called by WidomGCMCWorkChain'

        # TODO: Improve this section by adding temperature!
        # for pressure in self.ctx.pressures:
        for index, pressure in enumerate(self.ctx.pressures):
            # gcmc_label = "RaspaGCMC_{}".format(index + 1)
            # self.ctx.raspa_inputs['metadata']['label'] = "RaspaGCMC_{}".format(self.ctx.current_p_index + 1)
            self.ctx.raspa_inputs['metadata']['label'] ="RaspaGCMC_{}".format(index + 1)
            # self.ctx.raspa_inputs['metadata']['label'] =gcmc_label

            # self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gcmc_{}".format(self.ctx.current_p_index + 1)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gcmc_{}".format(index + 1)

            # self.ctx.raspa_params["System"][self.inputs.structure.label]["ExternalPressure"] = self.ctx.pressures[self.ctx.current_p_index] * 1e5
            self.ctx.raspa_params["System"][self.inputs.structure.label]["ExternalPressure"] = self.ctx.pressures[index] * 1e5

            self.ctx.raspa_inputs['raspa']['parameters'] = Dict(dict=self.ctx.raspa_params)

            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            # self.report("Running Raspa GCMC @ {}K/{:.3f}bar (pressure {} of {})".format(self.ctx.temperature, self.ctx.pressures[self.ctx.current_p_index], self.ctx.current_p_index + 1,len(self.ctx.pressures)))
            self.report("Running Raspa GCMC @ {}K/{:.3f}bar (pressure {} of {})".format(self.ctx.temperature, self.ctx.pressures[index], index + 1,len(self.ctx.pressures)))
            # self.ctx.current_p_index += 1
            # self.to_context(**{gcmc_label: running})
            self.to_context(raspa_gcmc=append_(running))

    def inspect_raspa_gcmc_calc(self):
        """ """
        for workchain in self.ctx.raspa_gcmc:
            assert workchain.is_finished_ok

    def return_output_parameters(self):
        """Merge all the parameters into output_parameters, depending on is_porous and is_kh_ehough."""

        all_out_dict = {}

        if all(self.ctx.should_run_widom):
            for value in self.ctx.molecules.get_dict().values():
                widom_label = "widom_{}".format(value['name'])
                # zeopp_label = "zeopp_{}".format(value['name'])
                all_out_dict[widom_label] = self.ctx[widom_label].outputs.output_parameters
                # all_out_dict[zeopp_label] = self.ctx.geom[value['name']]
            if all(self.ctx.should_run_gcmc):
                for workchain in self.ctx.raspa_gcmc:
                    all_out_dict[workchain.label] = workchain.outputs.output_parameters
            else:
                self.ctx.pressures = None
        else:
            self.ctx.pressures = None
            self.ctx.molecules = None
        self.out(
            "output_parameters",
            get_output_parameters(wc_params=self.ctx.parameters,
                                  pressures=self.ctx.pressures,
                                  components=self.ctx.molecules,
                                  **all_out_dict))

        self.report("Isotherm @ {}K computed: ouput Dict<{}>".format(self.ctx.temperature,
                                                                     self.outputs['output_parameters'].pk))


# EOF
