
# -*- coding: utf-8 -*-
"""General Purpose (Calc)Functions"""
from __future__ import absolute_import
from __future__ import print_function
import os

from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Dict, List, SinglefileData
from aiida.engine import calcfunction

@calcfunction
def get_molecule_dict(molecule_name):
    """Get a Dict from the isotherm_molecules.yaml"""
    import ruamel.yaml as yaml
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    molecule_dict = yaml_dict[molecule_name.value]
    return Dict(dict=molecule_dict)

@calcfunction
def get_molecules_input_dict(molecules, wc_params):
    """Construct molecules dict"""
    import ruamel.yaml as yaml
    molecules_dict = {}
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    for key, value in molecules.get_dict().items():
        molecules_dict[key] = yaml_dict[value['name']]
        molecules_dict[key]['molfraction'] = value['molfraction']
        probe_rad = molecules_dict[key]['proberad']
        if 'zeopp_accuracy' in wc_params.keys():
            molecules_dict[key]['zeopp'] = {
                'ha': wc_params['zeopp_accuracy'],
                'res': True,
                'sa': [probe_rad, probe_rad, wc_params['zeopp_sa_samples']],
                'volpo': [probe_rad, probe_rad, wc_params['zeopp_volpo_samples']],
                'block': [probe_rad, wc_params['zeopp_block_samples']],
            }

    return Dict(dict=molecules_dict)

# @calcfunction
# def get_molecules_input_dict(molecules, wc_params):
#     """Construct components dict"""
#     import ruamel.yaml as yaml
#     molecules_dict = {}
#     thisdir = os.path.dirname(os.path.abspath(__file__))
#     yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
#     with open(yamlfile, 'r') as stream:
#         yaml_dict = yaml.safe_load(stream)
#     if isinstance(molecules, Dict):
#         for key, value in molecules.get_dict().items():
#             molecules_dict[key] = yaml_dict[value['name']]
#             molecules_dict[key]['molfraction'] = value['molfraction']
#             probe_rad = molecules_dict[key]['proberad']
#             molecules_dict[key]['zeopp'] = {
#                 'ha': wc_params['zeopp_accuracy'],
#                 'res': True,
#                 'sa': [probe_rad, probe_rad, wc_params['zeopp_sa_samples']],
#                 'volpo': [probe_rad, probe_rad, wc_params['zeopp_volpo_samples']],
#                 'block': [probe_rad, wc_params['zeopp_block_samples']],
#             }
#     elif isinstance(molecules, List):
#         for mol in molecules:
#             probe_rad = yaml_dict[mol]['proberad']
#             molecules_dict[mol] = {}
#             molecules_dict[mol]['zeopp'] = {
#                 'ha': wc_params['zeopp_accuracy'],
#                 'res': True,
#                 'sa': [probe_rad, probe_rad, wc_params['zeopp_sa_samples']],
#                 'volpo': [probe_rad, probe_rad, wc_params['zeopp_volpo_samples']],
#                 'block': [probe_rad, wc_params['zeopp_block_samples']],
#             }
#
#     return Dict(dict=molecules_dict)

# @calcfunction
# def get_zeopp_input_dict(molecules, wc_params):
#     """Construct components dict"""
#     import ruamel.yaml as yaml
#     zeopp_dict = {}
#     thisdir = os.path.dirname(os.path.abspath(__file__))
#     yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
#     with open(yamlfile, 'r') as stream:
#         yaml_dict = yaml.safe_load(stream)
#     # for key, value in molecules.get_dict().items():
#     for mol in molecules:
#         # zeopp_dict[key] = yaml_dict[value['name']]
#         probe_rad = yaml_dict[mol]['proberad']
#         zeopp_dict[mol]['zeopp'] = {
#             'ha': wc_params['zeopp_accuracy'],
#             'res': True,
#             'sa': [probe_rad, probe_rad, wc_params['zeopp_sa_samples']],
#             'volpo': [probe_rad, probe_rad, wc_params['zeopp_volpo_samples']],
#             'block': [probe_rad, wc_params['zeopp_block_samples']],
#         }
#     return Dict(dict=zeopp_dict)


# Calcfuntions
# @calcfunction
# def get_zeopp_parameters(molecule, wcparams):
#     """Get the ZeoppParameters from the components Dict!"""
#     import ruamel.yaml as yaml
#     thisdir = os.path.dirname(os.path.abspath(__file__))
#     yamlfile = os.path.join(thisdir, "..", "data", "molecules.yaml")
#     with open(yamlfile, 'r') as stream:
#         yaml_dict = yaml.safe_load(stream)
#     proberad = yaml_dict['proberad']
#     params = {
#         'ha': 'DEF',
#         'res': False,
#         'sa': [proberad, proberad, wcparams['zeopp_sa_samples']],
#         'volpo': [proberad, proberad, wcparams['zeopp_volpo_samples']],
#         'block': [proberad, wcparams['zeopp_block_samples']],
#     }
#     return ZeoppParameters(dict=params)

@calcfunction
def get_ff_parameters(wc_params, molecule=None, components=None):
    """Get the parameters for ff_builder."""
    ff_params = {
        'ff_framework': wc_params['ff_framework'],
        'ff_molecules': {},
        'shifted': wc_params['ff_shifted'],
        'tail_corrections': wc_params['ff_tail_corrections'],
        'mixing_rule': wc_params['ff_mixing_rule'],
        'separate_interactions': wc_params['ff_separate_interactions']
    }
    if molecule is not None:
        ff_params['ff_molecules'] = {molecule['name']: molecule['forcefield']}
    if components is not None:
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

@calcfunction
def extract_merge_outputs(molecules, **all_out_dict):
    out = {}
    for key, value in molecules.get_dict().items():
        comp = value['name']
        zeopp_label = "zeopp_{}".format(comp)
        out[comp] = all_out_dict[zeopp_label].get_dict()
        out[comp]['is_porous'] = out[comp]["POAV_A^3"] > 0.000
    return Dict(dict=out)



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
@calcfunction
def get_output_parameters(wc_params, pressures=None, components=None, **all_out_dict):
    """ Extract Widom and GCMC results to isotherm Dict """
    out_dict = {}
    # out_dict['geometric_output'] = {}

    # for key in all_out_dict:
    #     if key.startswith('zeopp'):
    #         comp = key.split('_')[1]
    #         out_dict['geometric_output'][comp] = all_out_dict[key].get_dict()

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
            'mol_fraction', "conversion_factor_molec_uc_to_cm3stp_cm3", "conversion_factor_molec_uc_to_mg_g",
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

@calcfunction
def get_temperature_points(vlcparams):
    """Chooses the pressure points for VLCCWorkChain
    Current version: Only gets inital and final T with spacing.
    TODO: also read the reference data and get the info from there.
    """
    if vlcparams["temperature_list"]:
        T = vlcparams["temperature_list"]
    else:
        import numpy as np

        T_min = vlcparams['T_min']
        T_max = vlcparams['T_max']
        dT = vlcparams['dT']
        T = list(np.arange(T_min, T_max + 1, dT))

    return List(list=T)

@calcfunction
def get_vlcc_output(temperatures, **gemc_out_dict):
    component = list(gemc_out_dict['RaspaGEMC_1']["box_one"]["components"].keys())[0]

    vlcc_output = {
        'temperatures': temperatures,
        'temperature_unit': 'K',
        'loading_absolute_average': {'vapor':[],'liquid':[]},
        'loading_absolute_dev': {'vapor':[],'liquid':[]},
        'loading_absolute_unit': 'mol/UC',
        'adsorbate_density_average': {'vapor':[],'liquid':[]},
        'adsorbate_density_dev': {'vapor':[],'liquid':[]},
        'adsorbate_density_unit': 'kg/m^3',
        "ads_ads_total_energy_average":{'vapor':[],'liquid':[]},
        "ads_ads_total_energy_dev":{'vapor':[],'liquid':[]},
        "ads_ads_vdw_energy_average":{'vapor':[],'liquid':[]},
        "ads_ads_vdw_energy_dev":{'vapor':[],'liquid':[]},
        "ads_ads_coulomb_energy_average":{'vapor':[],'liquid':[]},
        "ads_ads_coulomb_energy_dev":{'vapor':[],'liquid':[]},
        "energy_unit": "kJ/mol",
        "box_ax_average":{'vapor':[],'liquid':[]},
        "box_ax_dev":{'vapor':[],'liquid':[]},
        "box_by_average":{'vapor':[],'liquid':[]},
        "box_by_dev":{'vapor':[],'liquid':[]},
        "box_cz_average":{'vapor':[],'liquid':[]},
        "box_cz_dev":{'vapor':[],'liquid':[]},
        "box_length_unit": "A",
        "cell_volume_average":{'vapor':[],'liquid':[]},
        "cell_volume_dev":{'vapor':[],'liquid':[]},
        "cell_volume_unit": "A^3",
    }
    labels_comp = [
        'loading_absolute_average',
        'loading_absolute_dev',
        'adsorbate_density_average',
        'adsorbate_density_dev'
    ]

    labels_general = [
        "ads_ads_total_energy_average",
        "ads_ads_total_energy_dev",
        "ads_ads_vdw_energy_average",
        "ads_ads_vdw_energy_dev",
        "ads_ads_coulomb_energy_average",
        "ads_ads_coulomb_energy_dev",
        "box_ax_average",
        "box_ax_dev",
        "box_by_average",
        "box_by_dev",
        "box_cz_average",
        "box_cz_dev",
        "cell_volume_average",
        "cell_volume_dev"
    ]

    for i in range(len(temperatures)):
        gemc_out = gemc_out_dict['RaspaGEMC_{}'.format(i + 1)]
        box_one_density_ave = gemc_out["box_one"]["components"][component]['adsorbate_density_average']
        box_two_density_ave = gemc_out["box_two"]["components"][component]['adsorbate_density_average']

        if box_one_density_ave < box_two_density_ave:
            gemc_out_vap = gemc_out["box_one"]
            gemc_out_liq = gemc_out["box_two"]

        if box_one_density_ave > box_two_density_ave:
            gemc_out_vap = gemc_out["box_two"]
            gemc_out_liq = gemc_out["box_one"]

        for label in labels_comp:
            vlcc_output[label]['vapor'].append(gemc_out_vap["components"][component][label])
            vlcc_output[label]['liquid'].append(gemc_out_liq["components"][component][label])

        for label in labels_general:
            vlcc_output[label]['vapor'].append(gemc_out_vap["general"][label])
            vlcc_output[label]['liquid'].append(gemc_out_liq["general"][label])

    return Dict(dict=vlcc_output)
