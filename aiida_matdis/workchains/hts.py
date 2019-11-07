# -*- coding: utf-8 -*-
"""HTSWorkChain."""
from __future__ import absolute_import
import os
from six.moves import range

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Bool, Dict, List, Str, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida_matdis.utils import aiida_dict_merge, check_resize_unit_cell

RaspaBaseWorkChain = WorkflowFactory('raspa.base')  #pylint: disable=invalid-name

# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name
ZeoppParameters = DataFactory("zeopp.parameters")  #pylint: disable=invalid-name

ZeoppCalculation = CalculationFactory("zeopp.network")  #pylint: disable=invalid-name


@calcfunction
def get_components_dict(mixture, isotparam):
    """Construct components dict"""
    import ruamel.yaml as yaml
    mixture_dict = {}
    thisdir = os.path.dirname(os.path.abspath(__file__))
    yamlfile = os.path.join(thisdir, "isotherm_data", "isotherm_molecules.yaml")
    with open(yamlfile, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
    for key, value in mixture.get_dict().items():
        mixture_dict[key] = yaml_dict[value['name']]
        mixture_dict[key]['molfraction'] = value['molfraction']
        probe_rad = mixture_dict[key]['proberad']
        mixture_dict[key]['zeopp'] = {
            'ha': 'DEF',
            'res': True,
            'sa': [probe_rad, probe_rad, isotparam['zeopp_sa_samples']],
            'volpo': [probe_rad, probe_rad, isotparam['zeopp_volpo_samples']],
            'block': [probe_rad, isotparam['zeopp_block_samples']],
        }
    return Dict(dict=mixture_dict)


@calcfunction
def get_zeopp_parameters(components, comp):
    """Get the ZeoppParameters from the components Dict!"""
    return ZeoppParameters(dict=components[comp.value]['zeopp'])


@calcfunction
def get_atomic_radii(ff_param):
    """Get {forcefield}.rad as SinglefileData form workchain/isotherm_data"""
    forcefield = ff_param['ff_framework']
    thisdir = os.path.dirname(os.path.abspath(__file__))
    fullfilename = forcefield + ".rad"
    return SinglefileData(file=os.path.join(thisdir, "isotherm_data", fullfilename))


@calcfunction
def get_geometric_output(zeopp_out):
    """Return the geometric_output Dict from Zeopp results, including Qsat and is_porous"""
    geometric_output = zeopp_out.get_dict()
    geometric_output.update({'is_porous': geometric_output["POAV_A^3"] > 0.000})
    return Dict(dict=geometric_output)


@calcfunction
def get_ff_params(isotparam, components):
    """Get RASPA ff_parameters"""
    ff_param = {
        'ff_framework': isotparam['forcefield'],
        'ff_molecules': {},
        'shifted': isotparam['ff_shift'],
        'tailcorrections': isotparam['ff_tailcorr'],
        'mixing_rule': 'Lorentz-Berthelot'
    }
    for value in components.get_dict().values():
        ff = value['forcefield']  #pylint: disable=invalid-name
        ff_param['ff_molecules'][value['name']] = ff
    return Dict(dict=ff_param)


#pylint: disable = too-many-branches
@calcfunction
def get_isotherm_output(should_run_gcmc, parameters, components, pressures, **all_raspa_out_dict):
    """ Extract Widom and GCMC results to isotherm Dict """
    isotherm_output = {
        'temperature': parameters['temperature'],
        'temperature_unit': 'K',
        'henry_coefficient_unit': 'mol/kg/Pa',
        'adsorption_energy_widom_unit': 'kJ/mol',
    }

    widom_labels = [
        'henry_coefficient_average',
        'henry_coefficient_dev',
        'adsorption_energy_widom_average',
        'adsorption_energy_widom_dev',
    ]

    for label in widom_labels:
        isotherm_output[label] = {}

    for value in components.get_dict().values():
        comp = value['name']
        widom_label = "widom_{}".format(comp)
        output_widom = all_raspa_out_dict[widom_label].get_dict()
        for label in widom_labels:
            isotherm_output[label][comp] = output_widom['framework_1']['components'][comp][label]

    if should_run_gcmc:
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
        isotherm_output.update({
            'pressure': pressures,
            'pressure_unit': 'bar',
            'loading_absolute_unit': 'mol/kg',
            'enthalpy_of_adsorption_unit': 'kJ/mol'
        })
        for label in multi_comp_isotherm_labels:
            isotherm[label] = {}
        for label in general_labels:
            isotherm_output[label] = {}

        conv_ener = 1.0 / 120.273  # K to kJ/mol
        for i in range(len(pressures)):
            gcmc_out = all_raspa_out_dict['RaspaGCMC_{}'.format(i + 1)]["framework_1"]
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
                    isotherm_output[label][comp] = gcmc_out['components'][comp][label]

        isotherm_output.update({
            "isotherm": isotherm,
        })

    return Dict(dict=isotherm_output)


# Deafault parameters
ISOTHERMPARAMETERS_DEFAULT = Dict(
    dict={  #TODO: create IsothermParameters instead of Dict # pylint: disable=fixme
        "forcefield": "UFF",  # str, Forcefield of the structure
        "ff_tailcorr": True,  # bool, Apply tail corrections
        "ff_shift": False,  # bool, Shift or truncate at cutoff
        "ff_cutoff": 12.0,  # float, CutOff truncation for the VdW interactions (Angstrom)
        "temperature": 300,  # float, Temperature of the simulation
        "zeopp_volpo_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_sa_samples": int(1e5),  # int, Number of samples for VOLPO calculation (per UC volume)
        "zeopp_block_samples": int(100),  # int, Number of samples for BLOCK calculation (per A^3)
        "raspa_verbosity": 10,  # int, Print stats every: number of cycles / raspa_verbosity
        "raspa_widom_cycles": int(1e5),  # int, Number of widom cycles
        "raspa_gcmc_init_cycles": int(1e3),  # int, Number of GCMC initialization cycles
        "raspa_gcmc_prod_cycles": int(1e4),  # int, Number of GCMC production cycles
        "lcd_max": 15.0,  # Maximum allowed LCD.
        "pld_scale": 1.0,  # Scaling factor for minimum allowed PLD.
        "pressure_list": [0.1e5, 1.0e5],  # list, Pressure list for the isotherm (bar): if given it will skip  guess
        "ideal_selectivity_threshold": 1.0,  #mandatory if protocol is relative.
        "run_gcmc_protocol": 'always',  # always, loose, and tight!
    })


class HTSWorkChain(WorkChain):
    """HTSWorkChain computes pore diameter, surface area, pore volume,
    and block pockets for provided mixture composition and based on these
    results decides to run Henry coefficient and possibly multi-components
    GCMC calculations using RASPA.
    """

    @classmethod
    def define(cls, spec):
        super(HTSWorkChain, cls).define(spec)

        # HTSWorkChain inputs!
        spec.input("structure", valid_type=CifData, required=True, help="Input structure in cif format")
        spec.input("parameters",
                   valid_type=Dict,
                   default=ISOTHERMPARAMETERS_DEFAULT,
                   required=True,
                   help='It provides the parameters which control the decision making behavior of workchain.')
        spec.input("mixture",
                   valid_type=Dict,
                   required=True,
                   help='A dictionary of components with their corresponding mol fractions in the mixture.')

        # Exposing the Zeopp and RASPA inputs!
        spec.expose_inputs(ZeoppCalculation, namespace='zeopp', include=['code', 'metadata'])
        spec.expose_inputs(RaspaBaseWorkChain, namespace='raspa_base', exclude=['raspa.structure', 'raspa.parameters'])

        # Workflow.
        spec.outline(
            cls.setup,
            cls.run_zeopp,
            cls.inspect_zeopp_calc,
            if_(cls.should_run_widom)(
                cls.run_raspa_widom,
                cls.inspect_widom_calc,
                if_(cls.should_run_gcmc)(
                    cls.init_raspa_gcmc,
                    while_(cls.should_run_another_gcmc)(cls.run_raspa_gcmc,),
                    cls.return_isotherm,
                ),
            ),
        )

        # Dynamic output ports.
        spec.outputs.dynamic = True

    def setup(self):
        """Initialize parameters"""
        # Getting and updating the HTS parameters!
        self.ctx.parameters = aiida_dict_merge(ISOTHERMPARAMETERS_DEFAULT, self.inputs.parameters)
        # Getting the components dict.
        self.ctx.components = get_components_dict(self.inputs.mixture, self.ctx.parameters)
        self.ctx.ff_param = get_ff_params(self.ctx.parameters, self.ctx.components)
        # Get integer temperature in context for easy reports
        self.ctx.temperature = int(round(self.ctx.parameters['temperature']))

    def run_zeopp(self):
        """It performs the full zeopp calculation for all components."""
        # Required inputs
        zeopp_inputs = self.exposed_inputs(ZeoppCalculation, 'zeopp')
        zeopp_inputs.update({
            'metadata': {
                'label': "ZeoppResSaVolpoBlock",
                'call_link_label': 'run_zeopp',
                'description': 'Called by HTSWorkChain',
            },
            'structure': self.inputs.structure,
            'atomic_radii': get_atomic_radii(self.ctx.ff_param)
        })

        for key, value in self.ctx.components.get_dict().items():
            comp = value['name']
            zeopp_inputs.update({'parameters': get_zeopp_parameters(self.ctx.components, Str(key))})
            running = self.submit(ZeoppCalculation, **zeopp_inputs)
            zeopp_label = "zeopp_{}".format(comp)
            self.report("Running zeo++ block and volpo Calculation<{}>".format(running.id))
            self.to_context(**{zeopp_label: running})

    def inspect_zeopp_calc(self):
        """Asserts whether all widom calculations are finished ok."""
        for value in self.ctx.components.get_dict().values():
            assert self.ctx["zeopp_{}".format(value['name'])].is_finished_ok

    def should_run_widom(self):
        """Decided whether to run Henry coefficient calculation or not!"""
        self.ctx.should_run_widom = []
        self.ctx.geom = {}
        lcd_lim = self.ctx.parameters["lcd_max"]
        for value in self.ctx.components.get_dict().values():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            pld_lim = value["proberad"] * self.ctx.parameters["pld_scale"]
            self.ctx.geom[comp] = get_geometric_output(self.ctx[zeopp_label].outputs.output_parameters)
            pld_component = self.ctx.geom[comp]["Largest_free_sphere"]
            lcd_component = self.ctx.geom[comp]["Largest_included_sphere"]
            poav_component = self.ctx.geom[comp]["POAV_A^3"]
            if (lcd_component <= lcd_lim) and (pld_component >= pld_lim) and (poav_component > 0.0):
                self.report("Found {} blocking spheres".format(self.ctx.geom[comp]['Number_of_blocking_spheres']))
                if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                    self.ctx.geom[comp + "_block_file"] = self.ctx[zeopp_label].outputs.block
                self.ctx.should_run_widom.append(True)
            else:
                self.ctx.should_run_widom.append(False)
        if all(self.ctx.should_run_widom):
            self.report("ALL pre-selection conditions are satisfied: Calculate Henry coefficients")
        else:
            self.report("All/Some of pre-selection criteria are NOT met: terminate!")
        self.out("geometric_output", self.ctx.geom)
        return all(self.ctx.should_run_widom)

    def _get_widom_param(self):
        """Write Raspa input parameters from scratch, for a Widom calculation"""
        param = {
            "GeneralSettings": {
                "SimulationType":
                "MonteCarlo",
                "NumberOfInitializationCycles":
                0,
                "NumberOfCycles":
                self.ctx.parameters['raspa_widom_cycles'],
                "PrintPropertiesEvery":
                self.ctx.parameters['raspa_widom_cycles'] / self.ctx.parameters['raspa_verbosity'],
                "PrintEvery":
                int(1e10),
                "RemoveAtomNumberCodeFromLabel":
                True,  # be careful!
                "Forcefield":
                "{}_{}_{}".format(self.ctx.parameters['forcefield'], ["notc", "tc"][self.ctx.parameters['ff_tailcorr']],
                                  ["trunc", "shift"][self.ctx.parameters['ff_shift']]),
                "CutOff":
                self.ctx.parameters['ff_cutoff'],
            },
            "System": {
                "framework_1": {
                    "type": "Framework",
                    "ExternalTemperature": self.ctx.parameters['temperature'],
                }
            },
            "Component": {},
        }

        # Check particular conditions and settings
        mult = check_resize_unit_cell(self.inputs.structure, 2 * self.ctx.parameters['ff_cutoff'])
        param["System"]["framework_1"]["UnitCells"] = "{} {} {}".format(mult[0], mult[1], mult[2])

        return param

    def run_raspa_widom(self):
        """Run parallel Widom calculation in RASPA."""
        self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        self.ctx.raspa_inputs['metadata']['label'] = "RaspaWidom"
        self.ctx.raspa_inputs['metadata']['description'] = "Called by HTSWorkChain"
        self.ctx.raspa_inputs['raspa']['framework'] = {"framework_1": self.inputs.structure}
        self.ctx.raspa_inputs['raspa']['ff_parameters'] = self.ctx.ff_param

        for value in self.ctx.components.get_dict().values():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_widom_" + comp
            self.ctx.raspa_param = self._get_widom_param()
            self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
            self.ctx.raspa_param["Component"][comp] = {}
            self.ctx.raspa_param["Component"][comp]["MoleculeDefinition"] = value['forcefield']
            self.ctx.raspa_param["Component"][comp]["WidomProbability"] = 1.0
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                self.ctx.raspa_inputs["raspa"]["block_pocket"] = {
                    comp + "_block_file": self.ctx[zeopp_label].outputs.block
                }
                self.ctx.raspa_param["Component"][comp]["BlockPocketsFileName"] = comp + "_block_file"
            if value['charged']:
                self.ctx.raspa_param["GeneralSettings"].update({
                    "UseChargesFromCIFFile": "yes",
                    "ChargeMethod": "Ewald",
                    "EwaldPrecision": 1e-6
                })

            self.ctx.raspa_inputs['raspa']['parameters'] = Dict(dict=self.ctx.raspa_param)
            running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
            widom_label = "widom_{}".format(comp)
            self.report("Running Raspa Widom @ {}K for the Henry coefficient of <{}>".format(
                self.ctx.temperature, comp))
            self.to_context(**{widom_label: running})

    def inspect_widom_calc(self):
        """Asserts whether all widom calculations are finished ok."""
        for value in self.ctx.components.get_dict().values():
            assert self.ctx["widom_{}".format(value['name'])].is_finished_ok

    def should_run_gcmc(self):
        """Based on user-defined protocol, decides to run the GCMC calculation or not
        always: it skips the check!
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
            self.ctx.kh_comp1 = output1["framework_1"]["components"][self.ctx.components.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            self.ctx.kh_comp2 = output2["framework_1"]["components"][self.ctx.components.get_dict()['comp2']
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
            self.ctx.kh_comp1 = output1["framework_1"]["components"][self.ctx.components.get_dict()['comp1']
                                                                     ['name']]["henry_coefficient_average"]
            for value in self.ctx.components.get_dict().values():
                comp = value['name']
                widom_label = "widom_{}".format(comp)
                output = self.ctx[widom_label].outputs.output_parameters.get_dict()
                self.ctx.kh_comp = output["framework_1"]["components"][comp]["henry_coefficient_average"]
                self.ctx.ideal_selectivity = self.ctx.kh_comp1 / self.ctx.kh_comp
                if self.ctx.ideal_selectivity >= self.ctx.ideal_selectivity_threshold:
                    self.ctx.should_run_gcmc.append(True)
                else:
                    self.ctx.should_run_gcmc.append(False)

        return all(self.ctx.should_run_gcmc)

    def _update_param_input_for_gcmc(self):
        """Update Raspa input parameter, from Widom to GCMC"""

        param = self.ctx.raspa_param
        inp = self.ctx.raspa_inputs
        param["GeneralSettings"].update({
            "NumberOfInitializationCycles":
            self.ctx.parameters['raspa_gcmc_init_cycles'],
            "NumberOfCycles":
            self.ctx.parameters['raspa_gcmc_prod_cycles'],
            "PrintPropertiesEvery":
            int(1e6),
            "PrintEvery":
            self.ctx.parameters['raspa_gcmc_prod_cycles'] / self.ctx.parameters['raspa_verbosity']
        })
        param["Component"] = {}
        param["Component"] = {item: {} for index, item in enumerate(list(self.ctx.components.get_dict()))}
        inp["raspa"]["block_pocket"] = {}
        for key, value in self.ctx.components.get_dict().items():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            bp_label = comp + "_block_file"
            param["Component"][comp] = param["Component"].pop(key)
            param["Component"][comp].update({
                "MolFraction":
                value['molfraction'],
                "TranslationProbability":
                1.0,
                "ReinsertionProbability":
                1.0,
                "SwapProbability":
                2.0,
                "IdentityChangeProbability":
                2.0,
                "NumberOfIdentityChanges":
                len(list(self.ctx.components.get_dict())),
                "IdentityChangesList": [i for i in range(len(list(self.ctx.components.get_dict())))]
            })
            if not value['singlebead']:
                param["Component"][comp].update({"RotationProbability": 1.0})
            if value['charged']:
                param["GeneralSettings"].update({
                    "UseChargesFromCIFFile": "yes",
                    "ChargeMethod": "Ewald",
                    "EwaldPrecision": 1e-6
                })
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                param["Component"][comp]["BlockPocketsFileName"] = {}
                param["Component"][comp]["BlockPocketsFileName"]["framework_1"] = bp_label
                inp["raspa"]["block_pocket"][bp_label] = self.ctx[zeopp_label].outputs.block

        return param, inp

    def init_raspa_gcmc(self):
        """Initialize RASPA gcmc"""

        self.ctx.current_p_index = 0
        self.ctx.pressures = self.ctx.parameters["pressure_list"]
        self.report("Now evaluating the isotherm @ {}K for {} pressure points".format(
            self.ctx.temperature, len(self.ctx.pressures)))

        self.ctx.raspa_param, self.ctx.raspa_inputs = self._update_param_input_for_gcmc()

    def should_run_another_gcmc(self):
        """
        We run another raspa calculation only if the current iteration is
        smaller than the total number of pressures we want to compute.
        """
        return self.ctx.current_p_index < len(self.ctx.pressures)

    def run_raspa_gcmc(self):
        """
        It submits Raspa calculation to RaspaBaseWorkchain.
        """
        self.ctx.raspa_inputs['metadata']['label'] = "RaspaGCMC_{}".format(self.ctx.current_p_index + 1)
        self.ctx.raspa_inputs['metadata']['description'] = 'Called by HTSWorkChain'
        self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_gcmc_{}".format(self.ctx.current_p_index + 1)

        self.ctx.raspa_param["System"]["framework_1"]["ExternalPressure"] = self.ctx.pressures[
            self.ctx.current_p_index] * 1e5

        if self.ctx.current_p_index > 0:
            self.ctx.raspa_inputs["raspa"]['retrieved_parent_folder'] = self.ctx.raspa_gcmc[self.ctx.current_p_index -
                                                                                            1].outputs.retrieved
        self.ctx.raspa_inputs['raspa']['parameters'] = Dict(dict=self.ctx.raspa_param)

        # Create the calculation process and launch it
        running = self.submit(RaspaBaseWorkChain, **self.ctx.raspa_inputs)
        self.report("Running Raspa GCMC @ {}K/{:.3f}bar (pressure {} of {})".format(
            self.ctx.temperature, self.ctx.pressures[self.ctx.current_p_index], self.ctx.current_p_index + 1,
            len(self.ctx.pressures)))
        self.ctx.current_p_index += 1
        return ToContext(raspa_gcmc=append_(running))

    def return_isotherm(self):
        """If is_porous and is_kh_enough create the isotherm_output Dict and report the pks"""

        all_raspa_out_dict = {}

        for value in self.ctx.components.get_dict().values():
            widom_label = "widom_{}".format(value['name'])
            all_raspa_out_dict[widom_label] = self.ctx[widom_label].outputs.output_parameters

        for calc in self.ctx.raspa_gcmc:
            all_raspa_out_dict[calc.label] = calc.outputs.output_parameters

        self.out(
            "isotherm_output",
            get_isotherm_output(Bool(self.ctx.should_run_gcmc), self.ctx.parameters, self.ctx.components,
                                List(list=self.ctx.pressures), **all_raspa_out_dict))
        self.report("Isotherm @ {}K computed: isotherm Dict<{}>".format(self.ctx.temperature,
                                                                        self.outputs['isotherm_output'].pk))


# EOF
