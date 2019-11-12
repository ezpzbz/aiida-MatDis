# -*- coding: utf-8 -*-
"""HTSWorkChain."""
from __future__ import absolute_import
import os
from six.moves import range

# AiiDA modules
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.orm import Bool, Dict, Float, List, Str, SinglefileData
from aiida.engine import calcfunction
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida_lsmo.calcfunctions import ff_builder
from aiida_matdis.utils import aiida_dict_merge, get_replication_factors

RaspaBaseWorkChain = WorkflowFactory('raspa.base')  #pylint: disable=invalid-name

# Defining DataFactory and CalculationFactory
CifData = DataFactory("cif")  #pylint: disable=invalid-name
ZeoppParameters = DataFactory("zeopp.parameters")  #pylint: disable=invalid-name

ZeoppCalculation = CalculationFactory("zeopp.network")  #pylint: disable=invalid-name

# Order of appearance.
@calcfunction
def get_components_dict(mixture, htsparams):
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
            'sa': [probe_rad, probe_rad, htsparams['zeopp_sa_samples']],
            'volpo': [probe_rad, probe_rad, htsparams['zeopp_volpo_samples']],
            'block': [probe_rad, htsparams['zeopp_block_samples']],
        }
    return Dict(dict=components_dict)

@calcfunction
def get_ff_parameters(htsparams, components):
    """Get the parameters for ff_builder."""
    ff_params = {
        'ff_framework': htsparams['forcefield'],
        'ff_molecules': {},
        'shifted': htsparams['ff_shifted'],
        'tail_corrections': htsparams['ff_tail_corrections'],
        'mixing_rule': htsparams['ff_mixing_rule'],
        'separate_interactions': htsparams['ff_separate_interactions']
    }
    for value in components.get_dict().values():
        ff = value['forcefield']  #pylint: disable=invalid-name
        ff_params['ff_molecules'][value['name']] = ff
    return Dict(dict=ff_params)

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
    return SinglefileData(file=os.path.join(thisdir, "..", "data", fullfilename))


@calcfunction
def get_geometric_output(zeopp_out):
    """Return the geometric_output Dict from Zeopp results, including Qsat and is_porous"""
    geometric_output = zeopp_out.get_dict()
    geometric_output.update({'is_porous': geometric_output["POAV_A^3"] > 0.000})
    return Dict(dict=geometric_output)

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

@calcfunction
def get_raspa_params(htsparams, replication):
    """Write Raspa input parameters from scratch, for a Widom calculation"""
    param = {
        "GeneralSettings": {
            "SimulationType":
            "MonteCarlo",
            "NumberOfInitializationCycles": 0,
            "NumberOfCycles": htsparams['raspa_widom_cycles'],
            "PrintPropertiesEvery": htsparams['raspa_widom_cycles'] / htsparams['raspa_verbosity'],
            "PrintEvery": int(1e10),
            "RemoveAtomNumberCodeFromLabel": True,  # be careful!
            "Forcefield": "Local",
            "CutOff": htsparams['ff_cutoff'],
        },
        "System": {
            "framework_1": {
                "type": "Framework",
                "ExternalTemperature": htsparams['temperature'],
                "UnitCells": "{} {} {}".format(replication['nx'], replication['ny'], replication['nz'])
            }
        },
        "Component": {}
    }

    return Dict(dict=param)

@calcfunction
def update_widom_params(raspa_params, components, geom_out, comp):
    """Write Raspa input parameters from scratch, for a Widom calculation"""
    params = raspa_params.get_dict()
    for key, value in components.get_dict().items():
        comp_name = value['name']
        if comp_name == comp.value:
            charged = value['charged']
            cmp = value['name']
    params["Component"][cmp] = {}
    params["Component"][cmp]["MoleculeDefinition"] = "Local"
    params["Component"][cmp]["WidomProbability"] = 1.0

    if geom_out['Number_of_blocking_spheres'] > 0:
        params["Component"][comp.value]["BlockPocketsFileName"] = comp.value + "_block_file"

    if charged:
        params["GeneralSettings"].update({
            "UseChargesFromCIFFile": "yes",
            "ChargeMethod": "Ewald",
            "EwaldPrecision": 1e-6
        })

    return Dict(dict=params)


@calcfunction
def update_raspa_params(raspa_params, htsparams, components, **geom_out):
    """Updating the Raspa params to perform the first GCMC"""
    params = raspa_params.get_dict()
    params["GeneralSettings"].update({
        "NumberOfInitializationCycles": htsparams['raspa_gcmc_init_cycles'],
        "NumberOfCycles": htsparams['raspa_gcmc_prod_cycles'],
        "PrintPropertiesEvery": int(1e6),
        "PrintEvery": htsparams['raspa_gcmc_prod_cycles'] / htsparams['raspa_verbosity']
    })
    params["Component"] = {}
    params["Component"] = {item: {} for index, item in enumerate(list(components.get_dict()))}

    for key, value in components.get_dict().items():
        comp = value['name']
        bp_label = comp + "_block_file"
        params["Component"][comp] = params["Component"].pop(key)
        params["Component"][comp].update({
            "MolFraction": value['molfraction'],
            "TranslationProbability": 1.0,
            "ReinsertionProbability": 1.0,
            "SwapProbability": 2.0,
            "IdentityChangeProbability": 2.0,
            "NumberOfIdentityChanges": len(list(components.get_dict())),
            "IdentityChangesList": [i for i in range(len(list(components.get_dict())))]
        })

        if not value['singlebead']:
            params["Component"][comp].update({"RotationProbability": 1.0})
        # If any of components is charged, it does the job!
        if value['charged']:
            params["GeneralSettings"].update({
                "UseChargesFromCIFFile": "yes",
                "ChargeMethod": "Ewald",
                "EwaldPrecision": 1e-6
            })

        if geom_out[comp]['Number_of_blocking_spheres'] > 0:
            params["Component"][comp]["BlockPocketsFileName"] = {}
            params["Component"][comp]["BlockPocketsFileName"]["framework_1"] = bp_label

    return Dict(dict=params)

@calcfunction
def update_pressure(raspa_params, pressure):
    """ """
    params = raspa_params.get_dict()
    params["System"]["framework_1"]["ExternalPressure"] = pressure.value
    return Dict(dict=params)
# Deafault parameters
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
                   default=HTSPARAMETERS_DEFAULT,
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
                cls.init_raspa_widom,
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
        self.ctx.parameters = aiida_dict_merge(HTSPARAMETERS_DEFAULT, self.inputs.parameters)

        # Getting the components dict.
        self.ctx.components = get_components_dict(self.inputs.mixture, self.ctx.parameters)

        # Getting the FF parameters.
        self.ctx.ff_params = get_ff_parameters(self.ctx.parameters, self.ctx.components)

        # Get integer temperature in context for easy reports
        self.ctx.temperature = int(round(self.ctx.parameters['temperature']))

        # todo move somewhere that would be called only if we do the widom.
        # self.ctx.replication_factors = get_replication_factors(self.inputs.structure, self.ctx.parameters)

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
            'atomic_radii': get_atomic_radii(self.ctx.ff_params)
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

    def init_raspa_widom(self):
        """ """
        self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        self.ctx.raspa_inputs['metadata']['label'] = "RaspaWidom"
        self.ctx.raspa_inputs['metadata']['description'] = "Called by HTSWorkChain"
        self.ctx.raspa_inputs['raspa']['framework'] = {"framework_1": self.inputs.structure}
        self.ctx.raspa_inputs['raspa']['file'] = ff_builder(self.ctx.ff_params)
        self.ctx.replication_factors = get_replication_factors(self.inputs.structure, self.ctx.parameters)
        self.ctx.raspa_params = get_raspa_params(self.ctx.parameters, self.ctx.replication_factors)

    def run_raspa_widom(self):
        """Run parallel Widom calculation in RASPA."""
        # self.ctx.raspa_inputs = self.exposed_inputs(RaspaBaseWorkChain, 'raspa_base')
        # self.ctx.raspa_inputs['metadata']['label'] = "RaspaWidom"
        # self.ctx.raspa_inputs['metadata']['description'] = "Called by HTSWorkChain"
        # self.ctx.raspa_inputs['raspa']['framework'] = {"framework_1": self.inputs.structure}
        # self.ctx.raspa_inputs['raspa']['file'] = ff_builder(self.ctx.ff_params)

        for value in self.ctx.components.get_dict().values():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            self.ctx.raspa_inputs['metadata']['call_link_label'] = "run_raspa_widom_" + comp
            self.ctx.raspa_inputs["raspa"]["block_pocket"] = {}
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                self.ctx.raspa_inputs["raspa"]["block_pocket"] = {
                    comp + "_block_file": self.ctx[zeopp_label].outputs.block
                }
            # widom_params = update_widom_params(self.ctx.raspa_params, self.ctx.components, self.ctx.geom[comp], Str(comp))
            # self.ctx.raspa_param = get_widom_param(self.ctx.parameters, self.ctx.components, self.ctx.replication_factors, self.ctx.geom[comp], Str(comp))
            self.ctx.raspa_inputs['raspa']['parameters'] = update_widom_params(self.ctx.raspa_params, self.ctx.components, self.ctx.geom[comp], Str(comp))
            # self.ctx.raspa_inputs['raspa']['parameters'] = get_widom_param(self.ctx.parameters, self.ctx.components, self.ctx.replication_factors, self.ctx.geom[comp], Str(comp))
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

    def _update_input_for_gcmc(self):
        """Update Raspa input parameter, from Widom to GCMC"""
        inp = self.ctx.raspa_inputs
        inp["raspa"]["block_pocket"] = {}
        for key, value in self.ctx.components.get_dict().items():
            comp = value['name']
            zeopp_label = "zeopp_{}".format(comp)
            bp_label = comp + "_block_file"
            if self.ctx.geom[comp]['Number_of_blocking_spheres'] > 0:
                inp["raspa"]["block_pocket"][bp_label] = self.ctx[zeopp_label].outputs.block
        return inp

    def init_raspa_gcmc(self):
        """Initialize RASPA gcmc"""

        self.ctx.current_p_index = 0
        self.ctx.pressures = self.ctx.parameters["pressure_list"]
        self.report("Now evaluating the isotherm @ {}K for {} pressure points".format(
            self.ctx.temperature, len(self.ctx.pressures)))
        self.ctx.raspa_inputs = self._update_input_for_gcmc()
        self.ctx.raspa_params = update_raspa_params(self.ctx.raspa_params, self.ctx.parameters, self.ctx.components, **self.ctx.geom)

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

        pressure = Float(self.ctx.pressures[self.ctx.current_p_index] * 1e5)

        if self.ctx.current_p_index > 0:
            self.ctx.raspa_inputs["raspa"]['retrieved_parent_folder'] = self.ctx.raspa_gcmc[self.ctx.current_p_index -
                                                                                            1].outputs.retrieved
        self.ctx.raspa_inputs['raspa']['parameters'] = update_pressure(self.ctx.raspa_params, pressure)

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
