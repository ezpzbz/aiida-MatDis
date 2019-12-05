# -*- coding: utf-8 -*-
"""
GACMWorkChain: An AiiDA workchain for Geomtery-based Analysis of Crystalline Materials using Zeo++
Indeed, it is the zeopp parts of IsothermWC, IsothermMultiCompWC, and etc.
Here, I separated as an independent workchain for the following reasons:
1- It makes it easier to implement Multi-Temperature screening workchains.
2- It makes it easier to only run these for different purposes.
3- It makes it possible to construct a database of Geometirc-based properties of MOFs.
"""
from __future__ import absolute_import
import os
from six.moves import range

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Dict, List, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida_matdis.aide_de_camp import (get_molecules_input_dict,
                                       extract_merge_outputs,
                                       update_workchain_params)
# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name
ZeoppParameters = DataFactory("zeopp.parameters")  #pylint: disable=invalid-name
ZeoppCalculation = CalculationFactory("zeopp.network")  #pylint: disable=invalid-name

# Deafault parameters
WC_PARAMS_DEFAULT = Dict(
    dict={
        "zeopp_accuracy": 'DEF',
        "zeopp_volpo_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_sa_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_block_samples": int(100),  # int, Number of samples for BLOCK calculation (per A^3)
    })

class GACMWorkChain(WorkChain):
    """
    GACMWorkChain is designed to perform a series of calculation and
    and constrcut the database for later usage in other workchains.
    """

    @classmethod
    def define(cls, spec):
        super(GACMWorkChain, cls).define(spec)

        # Exposing the Zeopp and RASPA inputs!
        spec.expose_inputs(ZeoppCalculation, namespace='zeopp', include=['atomic_radii', 'code', 'metadata'])
        # GACMWorkChain inputs!
        spec.input("structure",
                    valid_type=CifData,
                    required=True,
                    help="Input structure in cif format")

        spec.input("parameters",
                   valid_type=Dict,
                   default=WC_PARAMS_DEFAULT,
                   required=True,
                   help='It provides the parameters which control the decision making behavior of workchain.')

        spec.input("molecules",
                   valid_type=Dict,
                   help='A dictionary of components with their corresponding mol fractions in the mixture.')


        # Workflow.
        spec.outline(
            cls.setup,
            cls.run_zeopp,
            cls.inspect_zeopp_calc,
            cls.return_results,
        )
        spec.expose_outputs(ZeoppCalculation, include=['block'])  #only if porous

        spec.output(
            'geometric_output',
            valid_type=Dict,
            required=False,
            help='Results of the Zeo++ calculation (density, pore volume, etc.) plus some extra results (Qsat)')

        spec.output_namespace('block_files',
                              valid_type=SinglefileData,
                              required=False,
                              dynamic=True,
                              help='Generated block pocket files for each probe if there are blocking spheres.')

    def setup(self):
        """Setting up initial calculation parameters."""
        self.ctx.parameters = update_workchain_params(WC_PARAMS_DEFAULT, self.inputs.parameters)
        self.ctx.molecules = get_molecules_input_dict(self.inputs.molecules, self.ctx.parameters)

    def run_zeopp(self):
        """
        Preparing and submitting the calculations in parallel
        It performs the full zeopp calculation for all components
        """
        # Required inputs
        zeopp_inputs = self.exposed_inputs(ZeoppCalculation, 'zeopp')
        zeopp_inputs.update({
            'metadata': {
                'label': "ZeoppResSaVolpoBlock",
            },
            'structure': self.inputs.structure,
            'atomic_radii': self.inputs.zeopp.atomic_radii, # needs to be changed.
        })
        for key, value in self.ctx.molecules.get_dict().items():
            comp = value['name']
            zeopp_inputs.update({
                'parameters': ZeoppParameters(dict=self.ctx.molecules[key]['zeopp']),
                'metadata':{
                    'call_link_label': 'run_zeopp_' + comp
                }
            })
            running = self.submit(ZeoppCalculation, **zeopp_inputs)
            zeopp_label = "zeopp_{}".format(comp)
            self.report("Running zeo++ res, sa, volpo, and block calculation<{}>".format(running.id))
            self.to_context(**{zeopp_label: running})

    def inspect_zeopp_calc(self):
        """Asserts whether all widom calculations are finished ok."""
        for key, value in self.ctx.molecules.get_dict().items():
            assert self.ctx["zeopp_{}".format(value['name'])].is_finished_ok

    def return_results(self):
        """ """
        all_out_dict = {}
        geom_out = {}
        for key, value in self.ctx.molecules.get_dict().items():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            all_out_dict[zeopp_label] = self.ctx[zeopp_label].outputs.output_parameters
            if self.ctx[zeopp_label].outputs.output_parameters['Number_of_blocking_spheres'] > 0:
                self.out("block_files.{}_block_file".format(comp), self.ctx[zeopp_label].outputs.block)

        geom_out = extract_merge_outputs(self.ctx.molecules,**all_out_dict)

        self.out("geometric_output", geom_out)
        self.report("WorkChain is completed successfully! Output Dict is <{}>".format(self.outputs['geometric_output'].pk))
#EOF
