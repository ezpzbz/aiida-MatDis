#!/usr/bin/env python  # pylint: disable=invalid-name
# -*- coding: utf-8 -*-
"""Run example two-component isotherm calculation with HKUST1 framework."""

from __future__ import absolute_import
from __future__ import print_function

import os
import click

from aiida.engine import run, submit
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Code, Dict

# Workchain objects
WidomGCMCWorkChain = WorkflowFactory('matdis.widom_gcmc')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name
SinglefileData = DataFactory('singlefile')


@click.command('cli')
@click.argument('raspa_code_label')
def main(raspa_code_label):
    """
    Prepare inputs and submit the Isotherm workchain.
    Usage: verdi run run_WidomGCMCChain_alone_HKUST-1_4comp.py raspa37@teslin
    """

    builder = WidomGCMCWorkChain.get_builder()

    builder.metadata.label = "test"

    builder.raspa_base.raspa.code = Code.get_from_string(raspa_code_label)

    options = {
        "resources": {
            "num_machines": 1,
            "tot_num_mpiprocs": 1,
        },
        "max_wallclock_seconds": 1 * 60 * 60,
        "withmpi": False,
    }
    builder.raspa_base.raspa.metadata.options = options
    builder.structure = CifData(file=os.path.abspath('../data/HKUST-1.cif'), label="hkust1")
    builder.geometric_data = load_node(10729)
    builder.block_files = {
        'Xe_block_file': load_node(10720)
    }
    builder.molecules = Dict(dict={
        'comp1': {
            'name': 'xenon',
            'molfraction': 0.65
        },
        'comp2': {
            'name': 'co2',
            'molfraction': 0.05
        },
        'comp3': {
            'name': 'n2',
            'molfraction': 0.06
        },
        'comp4': {
            'name': 'o2',
            'molfraction': 0.24
        },
    })

    for temp in [273,298]:
        builder.parameters = Dict(
            dict={
                'ff_framework': 'UFF',  # Default: UFF
                'temperature': temp,  # (K) Note: higher temperature will have less adsorbate and it is faster
                'raspa_widom_cycles': 100,  # Default: 1e5
                'raspa_gcmc_init_cycles': 100,  # Default: 1e3
                'raspa_gcmc_prod_cycles': 100,  # Default: 1e4
                'pressure_list': [0.1, 1.0],
            })

        submit(builder)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF