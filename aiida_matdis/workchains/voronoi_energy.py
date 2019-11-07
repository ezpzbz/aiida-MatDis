"""
Voronoi Energy Calculation WorkChain
"""
from __future__ import absolute_import
import os

from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Dict, List, Str
from aiida.engine import calcfunction, if_, ToContext, WorkChain

ZeoppCalculation = CalculationFactory("zeopp.network")  # pylint: disable=invalid-name
PorousMaterialsCalculation = CalculationFactory("porousmaterials")  # pylint: disable=invalid-name

CifData = DataFactory("cif")  # pylint: disable=invalid-name
NetworkParameters = DataFactory("zeopp.parameters")  # pylint: disable=invalid-name


@calcfunction
def update_components(inp_components, zeopp_res_output):
    """ Updating component Dictionary"""
    components = inp_components.get_dict()
    # probe_radius = (zeopp_res_output.get_dict()["Largest_free_sphere"] / 2) * 0.95
    probe_radius = (zeopp_res_output.get_dict()["Largest_free_sphere"] / 2)
    components['PLD'] = {'probe_radius': probe_radius}
    return Dict(dict=components)


@calcfunction
def modify_zeopp_parameters(param, **kwargs):
    """Modifying the NetworkParameters to keep the provenance."""
    for key in kwargs.keys():  # pylint: disable=consider-iterating-dictionary
        if key in ['zeopp_res', 'zeopp_visvoro', 'probe', 'components']:
            if key == 'zeopp_res':
                params = kwargs[key].get_dict()
                updated_params = {"res": True, "ha": param['pld_accuracy']}
                params.update(updated_params)
            if key == 'zeopp_visvoro':
                params = kwargs[key].get_dict()
                probe_radius = kwargs['components'].get_dict()[kwargs['probe'].value]['probe_radius']
                # del params["res"]
                updated_params = {"res": False,"visVoro": probe_radius, "ha": param['visvoro_accuracy']}
                params.update(updated_params)
        else:
            raise AttributeError("The modification protocol is not supported!")

    return NetworkParameters(dict=params)


@calcfunction
def modify_pm_parameters(pm_parameters, input_template):
    params = pm_parameters.get_dict()
    params['input_template'] = input_template.value
    return Dict(dict=params)


@calcfunction
def extract_wrap_results(**kwargs):
    """
    It gets all generated output_parameters from workchain,
    process them, and wrap them in a single Dict object!
    """
    # pylint: disable=too-many-locals
    results = {}
    # ZeoppCalculation Section
    results['zeopp'] = {}
    results['zeopp']['Largest_free_sphere'] = kwargs['zeopp_res'].get_dict()['Largest_free_sphere']
    results['zeopp']['Largest_included_sphere'] = kwargs['zeopp_res'].get_dict()['Largest_included_sphere']
    # PorousMaterials Secion!
    if 'pm_out' in kwargs.keys():
        import pandas as pd
        K_to_kJ_mol = 1.0 / 120.273  # pylint: disable=invalid-name
        R = 8.3144598  # pylint: disable=invalid-name
        results['porousmaterials'] = {}
        results["porousmaterials"] = kwargs['pm_out'].get_dict()
        ev_setting = kwargs['ev_setting']

    ev_output_list = []
    for key in kwargs.keys():  # pylint: disable=consider-iterating-dictionary
        if key.startswith('pm_ev'):
            ev_output_list.append(kwargs[key])
    for ev_out in ev_output_list:
        fname = ev_out.filename
        comp = fname[:-4].split("_")[-1]
        probe = fname[:-4].split("_")[-2] + "_probe"
        density = results["porousmaterials"][comp][probe]['framework_density']
        temperature = results["porousmaterials"][comp][probe]['temperature']
        output_abs_path = os.path.join(
            ev_out._repository._get_base_folder().abspath,  # pylint: disable=protected-access
            fname
        )
        df = pd.read_csv(output_abs_path, skiprows=5)  # pylint: disable=invalid-name
        n_nodes = df.shape[0]
        minimum = df.Ev_K.min()
        average = df.Ev_K.mean() * K_to_kJ_mol
        boltzmann_factor_sum = df.boltzmann_factor.sum()
        wtd_energy_sum = df.weighted_energy_K.sum()
        adsorption_energy = (wtd_energy_sum / boltzmann_factor_sum) * K_to_kJ_mol
        Kh = boltzmann_factor_sum / (R * temperature * n_nodes * density * 100000.0)  # pylint: disable=invalid-name
        results['porousmaterials'][comp][probe]['Ev_average'] = average
        results['porousmaterials'][comp][probe]["Eads_average"] = adsorption_energy
        results['porousmaterials'][comp][probe]["Kh"] = Kh

        for percentile in ev_setting:
            threshold = (percentile / 100) * minimum
            df_selected = df[df['Ev_K'] <= threshold]
            num_selected_nodes = df_selected.shape[0]
            percentile_average = df_selected.mean()['Ev_K'] * K_to_kJ_mol
            results['porousmaterials'][comp][probe]["Ev_p" + str(percentile)] = percentile_average
            results['porousmaterials'][comp][probe]["number_of_Voronoi_nodes_in_p" +
                                                    str(percentile)] = num_selected_nodes

    return Dict(dict=results)


class VoronoiEnergyWorkChain(WorkChain):
    """
    The VoronoiEnergyWorkChain is designed to perform zeo++ and
    PorousMaterials calculations to obtain and exctract Voronoi
    energy and other corresponding results.
    """

    @classmethod
    def define(cls, spec):
        """
        Exposing inputs, defining work chain parameters, and defining the
        workflow logics.
        """
        super(VoronoiEnergyWorkChain, cls).define(spec)
        # Zeopp
        spec.expose_inputs(ZeoppCalculation, namespace='zeopp', exclude=['parameters', 'structure'])
        # PorousMaterialsCalculation
        spec.expose_inputs(
            PorousMaterialsCalculation, namespace='porousmaterials', exclude=['structure', 'acc_voronoi_nodes']
        )
        # VoronoiEnergyWorkChain specific inputs!
        spec.input("structure", valid_type=CifData, required=True, help="Input structure in CIF format")
        spec.input("parameters", valid_type=Dict, required=True, help="Parameters to do the logic in workchain")
        spec.input(
            "components", valid_type=Dict, required=False, help="Dictionary of components along with their probe size."
        )
        # Workflow
        spec.outline(
            cls.setup,
            cls.run_zeopp_res,
            if_(cls.should_run_zeopp_visvoro)(
                cls.run_zeopp_visvoro,
                cls.inspect_zeopp_visvoro,
                if_(cls.should_run_pm)(cls.run_pm,),
            ),
            cls.return_results,
        )
        # Dyanmic output!
        spec.outputs.dynamic = True

    def setup(self):
        """
        Initialize variables and setup screening protocol!
        """
        self.ctx.parameters = self.inputs.parameters
        self.ctx.label = self.inputs.structure.filename[:-4]
        self.ctx.default_zeopp_params = NetworkParameters(dict={"res": True, "ha": "DEF"})
        self.ctx.zeopp_param = {}
        self.ctx.should_run_comp = [False]
        self.ctx.should_run_pld = [False]
        self.ctx.number_acc_voronoi_nodes = {}

    def run_zeopp_res(self):
        """
        It performs the zeopp pore diameter calculation.
        """
        zeopp_input = self.exposed_inputs(ZeoppCalculation, 'zeopp')
        zeopp_input['structure'] = self.inputs.structure
        zeopp_input['atomic_radii'] = self.inputs.zeopp.atomic_radii
        zeopp_input['metadata']['label'] = "Zeopp Pore Diameter Calculation"
        zeopp_input['metadata']['call_link_label'] = 'run_zeopp_res'
        self.ctx.zeopp_param['zeopp_res'] = self.ctx.default_zeopp_params
        zeopp_input['parameters'] = modify_zeopp_parameters(self.ctx.parameters, **self.ctx.zeopp_param)

        res = self.submit(ZeoppCalculation, **zeopp_input)
        self.report("pk: <{}> | Submitted Zeo++ Pore Diameter Calculation".format(res.pk))
        return ToContext(zeopp_res=res)

    def should_run_zeopp_visvoro(self):
        """
        It uses largest included sphere (Di or LCD) and largest free sphere
        (Df or PLD) as pre-screenig descriptors to pass or reject the
        structure.
        """
        self.ctx.should_run_visvoro = []
        lcd_lim = self.ctx.parameters['lcd_max']
        pld_lim = self.ctx.parameters['pld_min']
        lcd_current = self.ctx.zeopp_res.outputs.output_parameters.get_dict()["Largest_included_sphere"]
        pld_current = self.ctx.zeopp_res.outputs.output_parameters.get_dict()["Largest_free_sphere"]
        if (lcd_current <= lcd_lim) and (pld_current >= pld_lim):
            self.report("<{}> is a suitable structure for further investigation".format(self.ctx.label))
            self.ctx.should_run_visvoro.append(True)
        else:
            self.report("<{}> does not look like promising: stop".format(self.ctx.label))
            self.ctx.should_run_visvoro.append(False)

        return all(self.ctx.should_run_visvoro)

    def run_zeopp_visvoro(self):
        """
        It performs the visVoro calculation.
        """
        zeopp_input = self.exposed_inputs(ZeoppCalculation, 'zeopp')
        zeopp_input['structure'] = self.inputs.structure
        zeopp_input['metadata']['label'] = "Zeopp visVoro Calculation"

        self.ctx.components = update_components(self.inputs.components, self.ctx.zeopp_res.outputs.output_parameters)
        # All together submission!
        for key in self.ctx.components.keys():
            self.ctx.zeopp_param['zeopp_visvoro'] = self.ctx.default_zeopp_params
            self.ctx.zeopp_param['probe'] = Str(key)
            self.ctx.zeopp_param['components'] = self.ctx.components
            zeopp_input['parameters'] = modify_zeopp_parameters(self.ctx.parameters, **self.ctx.zeopp_param)
            zeopp_input['metadata']['call_link_label'] = 'run_zeopp_visvoro_' + key
            visvoro = self.submit(ZeoppCalculation, **zeopp_input)
            zeopp_label = "zeopp_{}".format(key)
            self.to_context(**{zeopp_label: visvoro})

    def inspect_zeopp_visvoro(self):
        """
        Checks if all zeopp_full calculations are finished ok.
        """
        for key in self.ctx.components.keys():
            zeopp_label = "zeopp_{}".format(key)
            assert self.ctx[zeopp_label].is_finished_ok

    def should_run_pm(self):
        """
        It checks if there is any accessible Voronoi nodes or not!
        If there is any, it submits a PorousMaterials calculation.
        """
        self.ctx.should_run_comp = []
        self.ctx.should_run_pld = []
        for key in self.ctx.components.keys():
            zeopp_label = "zeopp_{}".format(key)
            visvoro_dir = self.ctx[zeopp_label].outputs.retrieved._repository._get_base_folder().abspath  # pylint: disable=protected-access
            visvoro_path = os.path.join(visvoro_dir, "out.visVoro.voro_accessible")
            with open(visvoro_path, "r") as fobj:
                self.ctx.number_acc_voronoi_nodes[key] = int(fobj.readline().strip())
                if self.ctx.number_acc_voronoi_nodes[key] > 0:
                    self.report(
                        "Found <{}> accessible Voronoi nodes for <{}>".format(
                            self.ctx.number_acc_voronoi_nodes[key], key
                        )
                    )
                    if key == 'PLD':
                        self.ctx.should_run_pld.append(True)
                    else:
                        self.ctx.should_run_comp.append(True)
                else:
                    self.report("No accessible Voronoi nodes for <{}>!: stop".format(key))
                    if key == 'PLD':
                        self.ctx.should_run_pld.append(False)
                    else:
                        self.ctx.should_run_comp.append(False)

        return all(self.ctx.should_run_pld) or all(self.ctx.should_run_comp)

    def run_pm(self):
        """
        It runs a Ev calculation in PorousMaterials.
        """
        pm_input = self.exposed_inputs(PorousMaterialsCalculation, 'porousmaterials')
        pm_input['structure'] = {}
        pm_input['structure'][self.ctx.label] = self.inputs.structure
        pm_input['acc_voronoi_nodes'] = {}
        pm_input['metadata']['label'] = 'Voronoi Energy Calculation'
        pm_input['metadata']['call_link_label'] = 'run_pm'

        if all(self.ctx.should_run_comp) and all(self.ctx.should_run_pld):
            for key in self.ctx.components.keys():
                zeopp_label = "zeopp_{}".format(key)
                voro_label = "{}_{}".format(self.ctx.label, key)
                pm_input['acc_voronoi_nodes'][voro_label] = self.ctx[zeopp_label].outputs.voro_accessible
            pm_input["parameters"] = modify_pm_parameters(
                self.inputs.porousmaterials.parameters, Str('ev_vdw_kh_multicomp_pld_template')
            )

        if all(self.ctx.should_run_comp) and not all(self.ctx.should_run_pld):
            for key in self.inputs.components.keys():
                zeopp_label = "zeopp_{}".format(key)
                voro_label = "{}_{}".format(self.ctx.label, key)
                pm_input['acc_voronoi_nodes'][voro_label] = self.ctx[zeopp_label].outputs.voro_accessible
            pm_input["parameters"] = modify_pm_parameters(
                self.inputs.porousmaterials.parameters, Str('ev_vdw_kh_multicomp_template')
            )

        if not all(self.ctx.should_run_comp) and all(self.ctx.should_run_pld):
            voro_label = "{}_{}".format(self.ctx.label, 'PLD')
            pm_input['acc_voronoi_nodes'][voro_label] = self.ctx["zeopp_PLD"].outputs.voro_accessible
            pm_input["parameters"] = modify_pm_parameters(
                self.inputs.porousmaterials.parameters, Str('ev_vdw_kh_pld_template')
            )

        pm_ev = self.submit(PorousMaterialsCalculation, **pm_input)  # pylint: disable=invalid-name
        self.report("pk: <{}> | Running Voronoi Energy Calculation".format(pm_ev.pk))
        return ToContext(pm_ev=pm_ev)

    def return_results(self):
        """
        Attach the results to the output.
        """
        output_parameters = {}
        all_outputs = {}
        voro_accessible = {}
        # ZeoppCalculation Section
        all_outputs['zeopp_res'] = self.ctx.zeopp_res.outputs.output_parameters

        if all(self.ctx.should_run_visvoro):
            for key in self.ctx.components.keys():
                zeopp_label = "zeopp_{}".format(key)
                if all(self.ctx.should_run_visvoro):
                    if 'voro_accessible' in self.ctx[zeopp_label].outputs:
                        voro_accessible[key] = self.ctx[zeopp_label].outputs.voro_accessible
            self.out('voro_accessible', voro_accessible)

        if all(self.ctx.should_run_comp) or all(self.ctx.should_run_pld):
            all_outputs['pm_out'] = self.ctx.pm_ev.outputs.output_parameters

            if all(self.ctx.should_run_comp):
                for key in self.inputs.components.keys():
                    ev_label = "Ev_vdw_{}_{}_{}".format(self.ctx.label, key, key)
                    all_outputs['pm_ev_' + ev_label] = self.ctx.pm_ev.outputs['ev_output_file__' + ev_label]

            if all(self.ctx.should_run_pld):
                for key in self.inputs.components.keys():
                    ev_label = "Ev_vdw_{}_PLD_{}".format(self.ctx.label, key)
                    all_outputs['pm_ev_' + ev_label] = self.ctx.pm_ev.outputs['ev_output_file__' + ev_label]

        if 'ev_setting' in self.ctx.parameters.get_dict():
            all_outputs['ev_setting'] = List(list=self.ctx.parameters['ev_setting'])
        else:
            all_outputs['ev_setting'] = List(list=[90, 80, 50])

        output_parameters = extract_wrap_results(**all_outputs)

        # Finalizing the results and report!
        self.out("results", output_parameters)
        self.report("Workchain completed successfully! | Result Dict is <{}>".format(self.outputs["results"].pk))


# EOF
