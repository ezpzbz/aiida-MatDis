
# -*- coding: utf-8 -*-
"""General Purpose (Calc)Functions"""
from __future__ import absolute_import
from __future__ import print_function
import os

from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Dict, List, SinglefileData
from aiida.engine import calcfunction

@calcfunction
def get_components_dict(mixture, wc_params):
    """Construct components dict"""
    import ruamel.yaml as yaml
    components_dict = {}
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    for key, value in mixture.get_dict().items():
        components_dict[key] = yaml_dict[value['name']]
        components_dict[key]['molfraction'] = value['molfraction']
        probe_rad = components_dict[key]['proberad']
        components_dict[key]['zeopp'] = {
            'ha': 'DEF',
            'res': True,
            'sa': [probe_rad, probe_rad, wc_params['zeopp_sa_samples']],
            'volpo': [probe_rad, probe_rad, wc_params['zeopp_volpo_samples']],
            'block': [probe_rad, wc_params['zeopp_block_samples']],
        }
    return Dict(dict=components_dict)


@calcfunction
def get_ff_parameters(components, wc_params):
    """Get the parameters for ff_builder."""
    ff_params = {
        'ff_framework': wc_params['forcefield'],
        'ff_molecules': {},
        'shifted': wc_params['ff_shifted'],
        'tail_corrections': wc_params['ff_tail_corrections'],
        'mixing_rule': wc_params['ff_mixing_rule'],
        'separate_interactions': wc_params['ff_separate_interactions']
    }
    for value in components.get_dict().values():
        ff = value['forcefield']  #pylint: disable=invalid-name
        ff_params['ff_molecules'][value['name']] = ff
    return Dict(dict=ff_params)


@calcfunction
def get_atomic_radii(wc_params):
    """Get {ff_framework}.rad as SinglefileData form workchain/isotherm_data. If not existing use DEFAULT.rad."""
    thisdir = os.path.dirname(os.path.abspath(__file__))
    filename = wc_params['ff_framework'] + ".rad"
    filepath = os.path.join(thisdir, "isotherm_data", filename)
    if not os.path.isfile(filepath):
        filepath = os.path.join(thisdir, "isotherm_data", "DEFAULT.rad")
    return SinglefileData(file=filepath)


# TODO: Make it multi-component compatible for experimenting the protocol for choosing pressure. #pylint: disable=fixme
@calcfunction
def get_geometric_output(zeopp_out):
    """Return the geometric_output Dict from Zeopp results, including Qsat and is_porous"""
    geometric_output = zeopp_out.get_dict()
    geometric_output.update({'is_porous': geometric_output["POAV_A^3"] > 0.000})
    return Dict(dict=geometric_output)


@calcfunction
def get_pressure_list(wc_params):
    """Gets the pressure list as the AiiDA List"""
    if wc_params["pressure_list"]:
        pressure_points = wc_params["pressure_list"]
        return  List(list=pressure_points)
    else:
        raise ValueError("pressure list is not provided properly!")

@calcfunction
def choose_pressure_points(wc_params):
    """If 'presure_list' is not provide, model the isotherm as single-site langmuir and return the most important
    pressure points to evaluate for an isotherm, in a List.
    """
    if wc_params["pressure_list"]:
        pressure_points = wc_params["pressure_list"]
    else:
        # Simply create a linear range of pressure points.
        # TODO: Make it possible to guess but needs benchmarking. #pylint: disable=fixme
        pressure_points = [wc_params['pressure_min']]
        delta_p = wc_params['pressure_precision']
        while True:
            pold = pressure_points[-1]
            pnew = pold + delta_p
            if pnew <= wc_params['pressure_max']:
                pressure_points.append(pnew)
            else:
                pressure_points.append(wc_params['pressure_max'])
                break
    return List(list=pressure_points)


#pylint: disable = too-many-branches
# TODO: Improve it by adding separation descriptors.
@calcfunction
def get_output_parameters(wc_params, pressures=None, components=None, **all_out_dict):
    """ Extract Widom and GCMC results to isotherm Dict """
    out_dict = {}
    out_dict['geometric_output'] = {}

    for key in all_out_dict:
        if key.startswith('zeopp'):
            comp = key.split('_')[1]
            out_dict['geometric_output'][comp] = all_out_dict[key].get_dict()

    if components is not None:  #At least we have the widom!
        strc_label = list(all_out_dict["widom_{}".format(components['comp1']['name'])].get_dict().keys())[0]
        out_dict.update({
            'temperature': wc_params['temperature'],
            'temperature_unit': 'K',
            'henry_coefficient_unit': 'mol/kg/Pa',
            'adsorption_energy_widom_unit': 'kJ/mol',
        })

        widom_labels = [
            'henry_coefficient_average',
            'henry_coefficient_dev',
            'adsorption_energy_widom_average',
            'adsorption_energy_widom_dev',
        ]

        for label in widom_labels:
            out_dict[label] = {}

        for value in components.get_dict().values():
            comp = value['name']
            widom_label = "widom_{}".format(comp)
            output_widom = all_out_dict[widom_label].get_dict()
            for label in widom_labels:
                out_dict[label][comp] = output_widom[strc_label]['components'][comp][label]

    if pressures is not None:  #we also have the GCMC!
        isotherm = {}
        multi_comp_isotherm_labels = [
            'loading_absolute_average',
            'loading_absolute_dev',
            'enthalpy_of_adsorption_average',
            'enthalpy_of_adsorption_dev',
        ]
        general_labels = [
            'mol_fraction', "conversion_factor_molec_uc_to_cm3stp_cm3", "conversion_factor_molec_uc_to_gr_gr",
            "conversion_factor_molec_uc_to_mol_kg"
        ]
        out_dict.update({
            'pressure': pressures,
            'pressure_unit': 'bar',
            'loading_absolute_unit': 'mol/kg',
            'enthalpy_of_adsorption_unit': 'kJ/mol'
        })
        for label in multi_comp_isotherm_labels:
            isotherm[label] = {}
        for label in general_labels:
            out_dict[label] = {}

        conv_ener = 1.0 / 120.273  # K to kJ/mol
        for i in range(len(pressures)):
            gcmc_out = all_out_dict['RaspaGCMC_{}'.format(i + 1)][strc_label]
            for value in components.get_dict().values():
                comp = value['name']
                conv_load = gcmc_out['components'][comp]["conversion_factor_molec_uc_to_mol_kg"]
                for label in ['loading_absolute_average', 'loading_absolute_dev']:
                    if i == 0:
                        isotherm[label][comp] = []
                    isotherm[label][comp].append(conv_load * gcmc_out['components'][comp][label])

                for label in ['enthalpy_of_adsorption_average', 'enthalpy_of_adsorption_dev']:
                    if i == 0:
                        isotherm[label][comp] = []
                    isotherm[label][comp].append(conv_ener * gcmc_out['components'][comp][label])

                for label in general_labels:
                    out_dict[label][comp] = gcmc_out['components'][comp][label]

        out_dict.update({
            "isotherm": isotherm,
        })

    return Dict(dict=out_dict)
