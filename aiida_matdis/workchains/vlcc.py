# -*- coding: utf-8 -*-
"""VLCCWorkChain"""
from __future__ import absolute_import
import os

# from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.orm import Dict, Int, List, Str
from aiida.engine import calcfunction, ToContext, WorkChain, while_, append_
# TODO: Should we?
import aiida_lsmo.calcfunctions.ff_builder_module as FFBuilder

from aiida_matdis.aide_de_camp import (get_molecule_dict,
                                       get_ff_parameters,
                                       get_temperature_points,
                                       get_vlcc_output,
                                       update_workchain_params)

RaspaBaseWorkChain = WorkflowFactory('raspa.base')  #pylint: disable=invalid-name


VLCPARAMETERS_DEFAULT = Dict(
    dict={  #TODO: create IsothermParameters instead of Dict # pylint: disable=fixme
        "ff_framework": None,  # str, Forcefield of the structure (used also as a definition of ff.rad for zeopp)
        "ff_shifted": False,  # bool, Shift or truncate at cutoff
        "ff_tail_corrections": True,  # bool, Apply tail corrections
        "ff_mixing_rule": 'Lorentz-Berthelot',  # str, Mixing rule for the forcefield
        "ff_separate_interactions": False,  # bool, if true use only ff_framework for framework-molecule interactions
        "ff_cutoff": 12.0,  # float, CutOff truncation for the VdW interactions (Angstrom)
        "temperature": 300,  # float, Temperature of the simulation
        "raspa_verbosity": 10,  # int, Print stats every: number of cycles / raspa_verbosity
        "raspa_init_cycles": int(1e3),  # int, Number of GCMC initialization cycles
        "raspa_prod_cycles": int(1e4),  # int, Number of GCMC production cycles
        "temperature_list": None,
    })


class VLCCWorkChain(WorkChain):
    """This Worchain is designed to construct Vapor-Liquid Coexistence Curve through GEMC simulation"""
    @classmethod
    def define(cls, spec):
        super(VLCCWorkChain, cls).define(spec)

        spec.expose_inputs(RaspaBaseWorkChain, namespace='raspa_base', exclude=['raspa.structure', 'raspa.parameters'])
        spec.input("parameters", valid_type=Dict, required=False, help='parameters to control the workchain')
        spec.input("molecule", valid_type=Str, required=True)

        # Workflow
        spec.outline(
            cls.setup,
            cls.run_raspa_gemc,
            cls.inspect_raspa_gemc,
            cls.return_results
        )
        spec.outputs.dynamic = True

    def setup(self):
        """Initialize variables and setup screening protocol!"""
        if isinstance(self.inputs.molecule, Str):
            self.ctx.molecule = get_molecule_dict(self.inputs.molecule)
        # else:
            # raise ValueError('Molecule is not provided properly!')

        self.ctx.parameters = update_workchain_params(VLCPARAMETERS_DEFAULT, self.inputs.parameters)

        self.ctx.temperatures = get_temperature_points(self.ctx.parameters)
        # self.ctx.ff_params = get_ff_parameters(self.ctx.components, self.ctx.parameters)
        self.ctx.ff_params = get_ff_parameters(self.ctx.parameters, molecule=self.ctx.molecule, components=None)

        # self.ctx.raspa_parameters = get_raspa_param(self.ctx.parameters, self.ctx.molecule, self.ctx.temperatures)
        # self.ctx.current_T_index = 0
        self.report("<{}> number of temperature points are chosen for GEMC".format(len(self.ctx.temperatures)))
        # self.report("Starting from <{}>K, then toward maximum <{}>, and finally minimum <{}>".format(self.ctx.temperatures[1],self.ctx.temperatures[0][-1],self.ctx.temperatures[0][0]))

    def _get_raspa_params(self):
        """Write RASPA input parameters from scratch, for a GEMC calculation"""
        b1l = self.ctx.parameters['box_one_length']
        b2l = self.ctx.parameters['box_two_length']
        params = {
            "GeneralSettings": {
                "SimulationType": "MonteCarlo",
                "NumberOfCycles": self.ctx.parameters['raspa_prod_cycles'],
                "NumberOfInitializationCycles": self.ctx.parameters['raspa_init_cycles'],
                "PrintEvery": self.ctx.parameters['raspa_prod_cycles'] / self.ctx.parameters['raspa_verbosity'],
                "Forcefield": "Local",
                "CutOff": self.ctx.parameters['ff_cutoff'],
                "GibbsVolumeChangeProbability": 0.1,
            },
            "System": {
                "box_one": {
                    "type": "Box",
                    "BoxLengths": "{} {} {}".format(b1l, b1l, b1l),
                    "BoxAngles": "90 90 90",
                },
                "box_two": {
                    "type": "Box",
                    "BoxLengths": "{} {} {}".format(b2l, b2l, b2l),
                    "BoxAngles": "90 90 90",
                }
            },
            "Component": {
                self.ctx.molecule['name']: {
                    "MoleculeDefinition": "Local",
                    "TranslationProbability": 1.0,
                    "ReinsertionProbability": 1.0,
                    "GibbsSwapProbability": 1.0,
                    "CreateNumberOfMolecules": {
                        "box_one": self.ctx.parameters['box_one_nmols'],
                        "box_two": self.ctx.parameters['box_two_nmols'],
                    },
                },
            },
        }

        if self.ctx.molecule['charged']:
            params["GeneralSettings"].update({"ChargeMethod": "Ewald", "EwaldPrecision": 1e-6})
        return params

    def run_raspa_gemc(self):
        """It runs a GEMC calculation in RASPA."""
        self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        self.ctx.raspa_params = self._get_raspa_params()
        self.ctx.raspa_inputs['raspa']['file'] = FFBuilder.ff_builder(self.ctx.ff_params)

        for index, temp in enumerate(self.ctx.temperatures):
            # label = "RaspaGEMC_{}".format(index + 1)
            self.ctx.raspa_inputs['metadata']['label'] = "RaspaGEMC_{}".format(index + 1)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gemc_{}".format(index + 1)
            self.ctx.raspa_params["System"]["box_one"]["ExternalTemperature"] = temp
            self.ctx.raspa_params["System"]["box_two"]["ExternalTemperature"] = temp
            self.ctx.raspa_inputs['raspa']["parameters"] = Dict(dict=self.ctx.raspa_params)
            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            self.report("pk: <{}> | Running Raspa GEMC calculation".format(running.pk))
            # self.to_context(**{label: running})
            self.to_context(raspa_gemc=append_(running))

    def inspect_raspa_gemc(self):
        """Assering the submitted calculations """
        for workchain in self.ctx.raspa_gemc:
            assert workchain.is_finished_ok
        # for index, temp in enumerate(self.ctx.temperatures):
        #     label = "RaspaGEMC_{}".format(index + 1)
        #     assert self.ctx[label].is_finished_ok

    def return_results(self):
        """Extracting and wrapping up the results."""
        gemc_out_dict = {}
        for workchain in self.ctx.raspa_gemc:
            gemc_out_dict[workchain.label] = workchain.outputs.output_parameters
        # for index, temp in enumerate(self.ctx.temperatures):
        #     label = "RaspaGEMC_{}".format(index + 1)
        #     gemc_out_dict[label] = self.ctx[label].outputs.output_parameters

        self.out("vlcc_output", get_vlcc_output(self.ctx.temperatures, **gemc_out_dict))
        self.report("VLCCWorkChain has been completed successfully: Results Dict<{}>".format(self.outputs["vlcc_output"].pk))

    # EOF
