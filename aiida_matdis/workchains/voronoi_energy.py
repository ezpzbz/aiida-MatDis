"""
Voronoi Energy Calculation WorkChain
"""
from __future__ import absolute_import
import os

from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Dict, List, Str
from aiida.engine import calcfunction, if_, ToContext, WorkChain
from aiida_matdis.aide_de_camp import (update_components,
                                       modify_zeopp_parameters,
                                       modify_pm_parameters,
                                       extract_wrap_results,
                                       dict_merge)


ZeoppCalculation = CalculationFactory("zeopp.network")  # pylint: disable=invalid-name
PorousMaterialsCalculation = CalculationFactory("porousmaterials")  # pylint: disable=invalid-name

CifData = DataFactory("cif")  # pylint: disable=invalid-name
NetworkParameters = DataFactory("zeopp.parameters")  # pylint: disable=invalid-name

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

        self.ctx.zeopp_param['zeopp_res'] = self.ctx.default_zeopp_params
        dict_merge(
            zeopp_input, {
                'metadata': {
                    'label': "Zeopp Pore Diameter Calculation",
                    'call_link_label': 'run_zeopp_res'
                },
                'structure': self.inputs.structure,
                'atomic_radii': self.inputs.zeopp.atomic_radii,
                'parameters': modify_zeopp_parameters(self.ctx.parameters, **self.ctx.zeopp_param)

            }
        )

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
        dict_merge(
            zeopp_input, {
                'metadata': {
                    'label': "Zeopp Pore Diameter Calculation",
                },
                'structure': self.inputs.structure,
            }
        )

        zeopp_input['structure'] = self.inputs.structure
        zeopp_input['metadata']['label'] = "Zeopp visVoro Calculation"

        self.ctx.components = update_components(self.inputs.components, self.ctx.zeopp_res.outputs.output_parameters)
        # All together submission!
        for key in self.ctx.components.keys():
            self.ctx.zeopp_param['zeopp_visvoro'] = self.ctx.default_zeopp_params
            self.ctx.zeopp_param['probe'] = Str(key)
            self.ctx.zeopp_param['components'] = self.ctx.components
            dict_merge(
                zeopp_input, {
                    'metadata': {
                        'call_link_label': 'run_zeopp_visvoro_' + key
                    },
                    'parameters': modify_zeopp_parameters(self.ctx.parameters, **self.ctx.zeopp_param)

                }
            )

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
