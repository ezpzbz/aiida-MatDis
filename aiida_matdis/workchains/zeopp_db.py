# -*- coding: utf-8 -*-
"""ZeoppDBWorkChain"""
from __future__ import absolute_import
import os
from six.moves import range

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Bool, Dict, List, Str, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida_matdis.utils import aiida_dict_merge, check_resize_unit_cell

# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name
ZeoppParameters = DataFactory("zeopp.parameters")  #pylint: disable=invalid-name
ZeoppCalculation = CalculationFactory("zeopp.network")  #pylint: disable=invalid-name

# Calcfuntions
@calcfunction
def get_zeopp_parameters(molecule, wcparams):
    """Get the ZeoppParameters from the components Dict!"""
    import ruamel.yaml as yaml
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    proberad = yaml_dict['proberad']
    params = {
        'ha': 'DEF',
        'res': False,
        'sa': [proberad, proberad, wcparams['zeopp_sa_samples']],
        'volpo': [proberad, proberad, wcparams['zeopp_volpo_samples']],
        'block': [proberad, wcparams['zeopp_block_samples']],
    }
    return ZeoppParameters(dict=params)


@calcfunction
def get_geometric_output(zeopp_out):
    """Return the geometric_output Dict from Zeopp results, including Qsat and is_porous"""
    geometric_output = zeopp_out.get_dict()
    geometric_output.update({'is_porous': geometric_output["POAV_A^3"] > 0.000})
    return Dict(dict=geometric_output)


# Deafault parameters
ZEOPPPARAMETERS_DEFAULT = Dict(
    dict={  #TODO: create IsothermParameters instead of Dict # pylint: disable=fixme
        "zeopp_volpo_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_sa_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_block_samples": int(100),  # int, Number of samples for BLOCK calculation (per A^3)
    })

class ZeoppDBWorkChain(WorkChain):
    """ZeoppDBWorkChain is designed to perform a series of calculation and
    and constrcut the database for later usage in other workchains.
    """

    @classmethod
    def define(cls, spec):
        super(ZeoppDBWorkChain, cls).define(spec)

        # ZeoppDBWorkChain inputs!
        spec.input("structure", valid_type=CifData, required=True, help="Input structure in cif format")
        spec.input("parameters",
                   valid_type=Dict,
                   default=ZEOPPPARAMETERS_DEFAULT,
                   required=True,
                   help='It provides the parameters which control the decision making behavior of workchain.')
        spec.input("molecule",
                   valid_type=Str,
                   required=True,
                   help='A molecules for them we want to run zeopp calcs')

        # Exposing the Zeopp and RASPA inputs!
        spec.expose_inputs(ZeoppCalculation, namespace='zeopp', include=['atomic_radii', 'code', 'metadata'])

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
            required=False,  # only if not skip_zeopp
            help='Results of the Zeo++ calculation (density, pore volume, etc.) plus some extra results (Qsat)')

    def setup(self):
        """Setting up initial calculation parameters."""
        self.ctx.parameters = aiida_dict_merge(ZEOPPPARAMETERS_DEFAULT, self.inputs.parameters)

    def run_zeopp(self):
        """Preparing and submitting the calculations in parallel
        It performs the full zeopp calculation for all components
        """
        # Required inputs
        zeopp_inputs = self.exposed_inputs(ZeoppCalculation, 'zeopp')
        zeopp_inputs.update({
            'metadata': {
                'label': "ZeoppResSaVolpoBlock",
                'call_link_label': 'run_zeopp',
                'description': 'Called by ZeoppDBWorkChain',
            },
            'structure': self.inputs.structure,
            'atomic_radii': self.inputs.zeopp.atomic_radii,
            'parameters': get_zeopp_parameters(self.inputs.molecule, self.ctx.parameter)
        })

        running = self.submit(ZeoppCalculation, **zeopp_inputs)
        self.report("Running zeo++ block and volpo Calculation<{}>".format(running.id))
        return ToContext(zeopp=running)

    def return_results(self):
        """Decided whether to run Henry coefficient calculation or not!"""
        geom_out = get_geometric_output(self.ctx.zeopp.outputs.output_parameters)
        if geom_out['Number_of_blocking_spheres'] > 0:
            self.out_many(self.exposed_outputs(self.ctx.zeopp, ZeoppCalculation))

        self.out("geometric_output", self.ctx.geom)
        self.report("WorkChain is completed successfully!")
