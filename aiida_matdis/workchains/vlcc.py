# -*- coding: utf-8 -*-
"""VLCCWorkChain"""
from __future__ import absolute_import
import os

# from aiida.common import AttributeDict
from aiida.orm import Dict, Int, List, Str
from aiida.engine import calcfunction, ToContext, WorkChain, while_, append_

from aiida_matdis.utils import aiida_dict_merge
from aiida_raspa.workchains import RaspaBaseWorkChain

@calcfunction
def get_molecule_dict(molecule_name):
    """Get a Dict from the isotherm_molecules.yaml"""
    import ruamel.yaml as yaml
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "data", "molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    molecule_dict = yaml_dict[molecule_name.value]
    return Dict(dict=molecule_dict)

@calcfunction
def choose_temp_points(vlcparams):
    """Chooses the pressure points for VLCCWorkChain
    Current version: Only gets inital and final T with spacing.
    TODO: also read the reference data and get the info from there.
    """
    import numpy as np

    T_min = vlcparams['T_min']
    T_max = vlcparams['T_max']
    dT = vlcparams['dT']

    T_mean = np.mean([T_min,T_max])
    T = set()
    T = [T_min]
    while True:
        Told = T[-1]
        Tnew = Told + dT
        if Tnew < T_max:
            T.append(Tnew)
        else:
            T.append(T_max)
            if T_mean not in T:
                T.append(T_mean)
            T.sort()
            return List(list=T)

@calcfunction
def get_raspa_param(vlcparams, molecule):
    """Write RASPA input parameters from scratch, for a Widom calculation"""
    b1l = vlcparams['box_one_length']
    b2l = vlcparams['box_two_length']
    param={
        # "MOLECULE": molecule.get_dict(),
        # "TEMPERATURE" : temperatures,
        "GeneralSettings": {
            "SimulationType": "MonteCarlo",
            "NumberOfCycles": vlcparams['raspa_prod_cycles'],
            "NumberOfInitializationCycles": vlcparams['raspa_init_cycles'],
            "PrintEvery": vlcparams['raspa_prod_cycles'] / vlcparams['raspa_verbosity'],
            "Forcefield": "GenericMOFs",
            "CutOff": vlcparams['ff_cutoff'],
            "GibbsVolumeChangeProbability": 0.1,
        },
        "System": {
            "box_one": {
                "type": "Box",
                "BoxLengths": "{} {} {}".format(b1l, b1l, b1l),
                "BoxAngles": "90 90 90",
                # "ExternalTemperature": temperatures[0],
            },
            "box_two": {
                "type": "Box",
                "BoxLengths": "{} {} {}".format(b2l, b2l, b2l),
                "BoxAngles": "90 90 90",
                # "ExternalTemperature": temperatures[0],
            }
        },
        "Component": {
            molecule['name']: {
                "MoleculeDefinition": molecule["forcefield"],
                "TranslationProbability": 1.0,
                "ReinsertionProbability": 1.0,
                "GibbsSwapProbability": 1.0,
                "CreateNumberOfMolecules": {
                    "box_one": vlcparams['box_one_nmols'],
                    "box_two": vlcparams['box_two_nmols'],
                },
            },
        },
    }

    if molecule['charged']:
        param["GeneralSettings"].update({"ChargeMethod": "Ewald", "EwaldPrecision": 1e-6})
    return Dict(dict=param)

@calcfunction
def update_raspa_param(raspa_params, temperatures, index):
    param = raspa_params.get_dict()
    comp = list(param['Component'].keys())[0]
    # T_index = temperatures.index(param["System"]["box_one"]["ExternalTemperature"])
    param["System"]["box_one"]["ExternalTemperature"] = temperatures[index.value]
    param["System"]["box_two"]["ExternalTemperature"] = temperatures[index.value]

    # param["Component"][comp]["CreateNumberOfMolecules"]['box_one'] = 0
    # param["Component"][comp]["CreateNumberOfMolecules"]['box_two'] = 0

    return Dict(dict=param)

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

VLCPARAMETERS_DEFAULT = Dict(
    dict={  #TODO: create IsothermParameters instead of Dict # pylint: disable=fixme
        "forcefield": "UFF",  # str, Forcefield of the structure
        "ff_tailcorr": True,  # bool, Apply tail corrections
        "ff_shift": False,  # bool, Shift or truncate at cutoff
        "ff_cutoff": 12.0,  # float, CutOff truncation for the VdW interactions (Angstrom)
        "temperature": 300,  # float, Temperature of the simulation
        "raspa_verbosity": 10,  # int, Print stats every: number of cycles / raspa_verbosity
        "raspa_init_cycles": int(1e3),  # int, Number of GCMC initialization cycles
        "raspa_prod_cycles": int(1e4),  # int, Number of GCMC production cycles
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

        self.ctx.parameters = aiida_dict_merge(VLCPARAMETERS_DEFAULT, self.inputs.parameters)

        self.ctx.temperatures = choose_temp_points(self.ctx.parameters)
        self.ctx.raspa_parameters = get_raspa_param(self.ctx.parameters, self.ctx.molecule, self.ctx.temperatures)
        self.ctx.current_T_index = 0
        self.report("<{}> number of temperature points are chosen for GEMC".format(len(self.ctx.temperatures)))
        # self.report("Starting from <{}>K, then toward maximum <{}>, and finally minimum <{}>".format(self.ctx.temperatures[1],self.ctx.temperatures[0][-1],self.ctx.temperatures[0][0]))


    def run_raspa_gemc(self):
        """It runs a GEMC calculation in RASPA."""
        self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        for index, temp in enumerate(self.ctx.temperatures):
            label = "RaspaGEMC_{}".format(index + 1)
            self.ctx.raspa_inputs['metadata']['label'] = label
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gemc_{}".format(index + 1)
            self.ctx.raspa_inputs['raspa']["parameters"] = update_raspa_param(self.ctx.raspa_parameters, self.ctx.temperatures, Int(index))
            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            self.report("pk: <{}> | Running Raspa GEMC calculation".format(running.pk))
            self.to_context(**{label: running})

    def inspect_raspa_gemc(self):
        """Assering the submitted calculations """
        for index, temp in enumerate(self.ctx.temperatures):
            label = "RaspaGEMC_{}".format(index + 1)
            assert self.ctx[label].is_finished_ok

    def return_results(self):
        """Extracting and wrapping up the results."""
        gemc_out_dict = {}
        for index, temp in enumerate(self.ctx.temperatures):
            label = "RaspaGEMC_{}".format(index + 1)
            gemc_out_dict[label] = self.ctx[label].outputs.output_parameters

        self.out("vlcc_output", get_vlcc_output(self.ctx.temperatures, **gemc_out_dict))
        self.report("VLCCWorkChain has been completed successfully: Results Dict<{}>".format(self.outputs["vlcc_output"].pk))

    # EOF
