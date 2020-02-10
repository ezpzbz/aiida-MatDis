# -*- coding: utf-8 -*-
"""HTSMultiTempWorkChain"""
from __future__ import absolute_import

from six.moves import range

from aiida.plugins import WorkflowFactory
from aiida.orm import Dict
from aiida.engine import calcfunction
from aiida.engine import WorkChain, ToContext, if_, while_
from aiida_matdis.aide_de_camp import dict_merge

# import sub-workchains
HTSWorkChain = WorkflowFactory('matdis.hts')  # pylint: disable=invalid-name


def get_parameters_singletemp(i, parameters):
    parameters_singletemp = parameters.get_dict()
    parameters_singletemp['temperature'] = parameters_singletemp['temperature_list'][i]
    parameters_singletemp['temperature_list'] = None
    return Dict(dict=parameters_singletemp)

class HTSMultiTempWorkChain(WorkChain):
    """
    It uses similar logic to https://github.com/lsmo-epfl/aiida-lsmo/blob/develop/aiida_lsmo/workchains/isotherm_multi_temp.py
    It is able to perform several instances of HTSWorkChain to compute multi-component isotherms at different temperatures.
    """

    @classmethod
    def define(cls, spec):
        super(HTSMultiTempWorkChain, cls).define(spec)

        spec.expose_inputs(HTSWorkChain)

        spec.outline(
            cls.run_geometric,
            if_(cls.should_run_widom)(
                while_(cls.should_continue)(
                    cls.run_isotherms,
                )
            ),
            cls.collect_isotherms
            )

        spec.expose_outputs(HTSWorkChain, include=['block'])

        spec.output_namespace('output_parameters',
                              valid_type=Dict,
                              required=False,
                              dynamic=True,
                              help='Output parameters of each temperature!')


    def run_geometric(self):
        """Perform Zeo++ block and VOLPO calculation with IsothermWC."""
        # create inputs: exposed are code and metadata
        inputs = self.exposed_inputs(HTSWorkChain)

        dict_merge(
            inputs, {
                'metadata': {
                    'label': "IsothermGeometric",
                    'call_link_label': 'run_geometric',
                },
            }
        )

        running = self.submit(HTSWorkChain, **inputs)
        self.report("Computing common gemetric properties")
        return ToContext(geom_only=running)

    def should_run_widom(self):
        """Decided whether to run Henry coefficient calculation or not!"""
        self.ctx.current_T_index = 0

        self.ctx.should_run_widom = []
        self.ctx.geom = self.ctx.geom_only.outputs.output_parameters
        self.ctx.comps = list(self.ctx.geom.get_dict()['geometric_output'].keys())


        lcd_lim = self.inputs.parameters["lcd_max"]
        pld_lim = self.inputs.parameters["pld_min"]
        pld_current = self.ctx.geom['geometric_output'][self.ctx.comps[0]]["Largest_free_sphere"]
        lcd_current = self.ctx.geom['geometric_output'][self.ctx.comps[0]]["Largest_included_sphere"]

        for comp in self.ctx.comps:
            poav_component = self.ctx.geom['geometric_output'][comp]["POAV_A^3"]
            if (lcd_current <= lcd_lim) and (pld_current >= pld_lim) and (poav_component > 0.0):
                self.ctx.should_run_widom.append(True)
            else:
                self.ctx.should_run_widom.append(False)

        if all(self.ctx.should_run_widom):
            self.report("ALL pre-selection conditions are satisfied: Calculate Henry coefficients")
        else:
            self.report("All/Some of pre-selection criteria are NOT met: terminate!")

        return all(self.ctx.should_run_widom)

    def should_continue(self):
        return self.ctx.current_T_index < len(self.inputs.parameters['temperature_list'])

    def run_isotherms(self):
        """Compute isotherms at different temperatures."""
        # create inputs: exposed are code and metadata
        inputs = self.exposed_inputs(HTSWorkChain)
        inputs['raspa_base']['raspa']['block_pocket'] = {}
        inputs['geometric'] = self.ctx.geom
        for comp in self.ctx.comps:
            if self.ctx.geom['geometric_output'][comp]['Number_of_blocking_spheres'] > 0:
                inputs['raspa_base']["raspa"]["block_pocket"][comp + "_block_file"] = self.ctx.geom_only.outputs['block_files__' + comp + '_block_file']

        # Update the parameters with only one temperature and submit
        for index, temp in enumerate(self.inputs.parameters['temperature_list']):
            self.ctx.parameters_singletemp = get_parameters_singletemp(index, self.inputs.parameters)
            dict_merge(
                inputs, {
                    'metadata': {
                        'label': "Isotherm_{}".format(index),
                        'call_link_label': 'run_isotherm_{}'.format(index),
                    },
                    'parameters': self.ctx.parameters_singletemp
                }
            )

            running = self.submit(HTSWorkChain, **inputs)
            self.ctx.current_T_index += 1
            self.to_context(**{'isotherm_{}'.format(index): running})

    def collect_isotherms(self):
        """ Collect all the results in one Dict """
        # pk_list = []
        if all(self.ctx.should_run_widom):
            for index, temp in enumerate(self.inputs.parameters['temperature_list']):
                self.out('output_parameters.T_{}_K'.format(temp), self.ctx['isotherm_{}'.format(index)].outputs['output_parameters'])
                # pk_list.append(self.outputs['output_parameters']['T_{}_K'.format(temp)].pk)

        self.report("HTSMultiTempWorkChain is completed for {} temperatures!".format(len(self.inputs.parameters['temperature_list'])))
#EOF
