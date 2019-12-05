#!/usr/bin/env python  # pylint: disable=invalid-name
# -*- coding: utf-8 -*-
"""Run example two-component isotherm calculation with HKUST1 framework."""

from __future__ import absolute_import
from __future__ import print_function

import os
import click

from aiida.engine import run
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Code, Dict

# Workchain objects
GACMWorkChain = WorkflowFactory('matdis.gacm')  # pylint: disable=invalid-name

# Data objects
CifData = DataFactory('cif')  # pylint: disable=invalid-name
NetworkParameters = DataFactory('zeopp.parameters')  # pylint: disable=invalid-name
SinglefileData = DataFactory('singlefile')


@click.command('cli')
@click.argument('zeopp_code_label')

def main(zeopp_code_label):
    """Prepare inputs and submit the Isotherm workchain.
    Usage: verdi run run_IsothermMultiCompWorkChain_HKUST-1.py raspa@localhost network@localhost"""

    builder = GACMWorkChain.get_builder()

    builder.metadata.label = "test"


    builder.zeopp.code = Code.get_from_string(zeopp_code_label)
    builder.zeopp.atomic_radii = SinglefileData(file=os.path.abspath('../data/UFF.rad'))

    options = {
        "resources": {
            "num_machines": 1,
            "tot_num_mpiprocs": 1,
        },
        "max_wallclock_seconds": 1 * 60 * 60,
        "withmpi": False,
    }

    builder.zeopp.metadata.options = options
    builder.structure = CifData(file=os.path.abspath('../data/HKUST-1.cif'), label="hkust1")
    builder.molecules = List(list=['co2','n2','ch4','xenon'])

    builder.parameters = Dict(
        dict={
            'zeopp_accuracy':'DEF',
            'zeopp_volpo_samples': 10,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_sa_samples': 10,  # Default: 1e5 *NOTE: default is good for standard real-case!
            'zeopp_block_samples': 10,  # Default: 100
        })

    run(builder)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF
