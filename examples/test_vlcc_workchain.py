# -*- coding: utf-8 -*-
"""One-component GEMC through RaspaBaseWorkChain"""

from __future__ import absolute_import
from __future__ import print_function
import sys
import click

from aiida.common import NotExistent
from aiida.engine import run, submit
from aiida.orm import Code, Dict
from aiida.plugins import WorkflowFactory

VLCCWorkChain = WorkflowFactory('matdis.vlcc')  # pylint: disable=invalid-name

@click.command('cli')
@click.argument('codelabel')
def main(codelabel):
    """Run base workchain"""

    # pylint: disable=no-member

    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print("The code '{}' does not exist".format(codelabel))
        sys.exit(1)

    print("Testing RASPA methane GEMC through RaspaBaseWorkChain ...")


    # Constructing builder
    builder = VLCCWorkChain.get_builder()

    # Specifying the code
    builder.raspa_base.raspa.code = code
    builder.raspa_base.fixtures = {
        'fixture_001': ('aiida_raspa.utils', 'check_gemc_box')
    }
    builder.molecule = Str('ch4')
    builder.parameters = Dict(
        dict={
            'raspa_init_cycles': 100,  # Default: 1e3
            'raspa_prod_cycles': 100,  # Default: 1e4
            'box_one_nmols': 150,
            'box_two_nmols': 150,
            'box_one_length': 30,
            'box_two_length': 30,
            'T_min': 280,
            'T_max': 300,
            'dT': 10,
        })

    # Specifying the scheduler options
    builder.raspa_base.raspa.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 1 * 30 * 60,  # 30 min
        "withmpi": False,
    }

    run(builder)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

# EOF
