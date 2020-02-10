# -*- coding: utf-8 -*-
""" Sample run script for VoronoiEnergyWorkChain"""
from __future__ import print_function
from __future__ import absolute_import
import os
import sys

from aiida.common import NotExistent
from aiida.orm import Code, Dict
from aiida.plugins import DataFactory
from aiida.engine import run

VoronoiEnergyWorkChain = WorkflowFactory('matdis.voronoi_energy')

# Reading the structure and convert it to structure data.
SinglefileData = DataFactory('singlefile')  # pylint: disable=invalid-name
CifData = DataFactory('cif')  # pylint: disable=invalid-name

structure = CifData(file=os.path.abspath("./KAXQIL_clean_P1.cif"))  # pylint: disable=invalid-name
structure.label = structure.filename.lower()[:-4]

# Reading code information from system argv
if len(sys.argv) != 3:
    print("Usage: test.py <zeopp_code_name> <julia_code_name>")
    sys.exit(1)

zeopp_codename = sys.argv[1]  # pylint: disable=invalid-name
julia_codename = sys.argv[2]  # pylint: disable=invalid-name

try:
    zeopp_code = Code.get_from_string(zeopp_codename)  # pylint: disable=invalid-name
except NotExistent:
    print("The code '{}' does not exist".format(zeopp_codename))  # pylint: disable=invalid-name
    sys.exit(1)

try:
    julia_code = Code.get_from_string(julia_codename)  # pylint: disable=invalid-name
except NotExistent:
    print("The code '{}' does not exist".format(julia_codename))
    sys.exit(1)

zeopp_atomic_radii_file = SinglefileData(file=os.path.abspath("./UFF.rad"))  # pylint: disable=invalid-name

components = Dict(dict={  # pylint: disable=invalid-name
    "Xe": {
        "probe_radius": 1.985
    },
    "Kr": {
        "probe_radius": 1.82
    },
})

wc_parameters = Dict( # pylint: disable=invalid-name
    dict={
        "pld_min": 3.50,
        "lcd_max": 15.0,
        "visvoro_accuracy": "DEF",
        "pld_accuracy": "S100",
        "pld_based": False,
        'ev_setting': [99, 95, 90, 80, 50],
    })

pm_parameters = Dict( # pylint: disable=invalid-name
    dict={
        'data_path': "/storage/brno9-ceitec/home/pezhman/projects/noble_gas_epfl/xe_kr/data",
        'ff': 'UFF.csv',
        'cutoff': 12.5,
        'mixing': 'Lorentz-Berthelot',
        'framework': structure.filename,
        'frameworkname': structure.filename[:-4],
        'adsorbates': '["Xe","Kr"]',
        'temperature':298.0,
    })

# Constructing builder
builder = VoronoiEnergyWorkChain.get_builder()  # pylint: disable=invalid-name
# VoronoiEnergyWorkChain inputs
builder.structure = structure
builder.parameters = wc_parameters
builder.components = components
builder.metadata.label = "SBMOF-1"  #pylint: disable = no-member
builder.metadata.description = "Test VoronoiEnergyWorkChain with SBMOF1"  #pylint: disable = no-member
# PorousMaterials inputs
builder.porousmaterials.code = julia_code  #pylint: disable = no-member
builder.porousmaterials.parameters = pm_parameters  #pylint: disable = no-member
builder.porousmaterials.metadata.options.resources = { #pylint: disable = no-member
    "num_machines": 1,
    "tot_num_mpiprocs": 1,
}
builder.porousmaterials.metadata.options.max_wallclock_seconds = 1 * 30 * 60  #pylint: disable = no-member
builder.porousmaterials.metadata.options.max_memory_kb = 2000000  #pylint: disable = no-member
builder.porousmaterials.metadata.options.queue_name = "default"
builder.porousmaterials.metadata.options.withmpi = False  #pylint: disable = no-member
# Zeopp inputs
builder.zeopp.code = zeopp_code  #pylint: disable = no-member
builder.zeopp.atomic_radii = zeopp_atomic_radii_file  #pylint: disable = no-member
builder.zeopp.metadata.options.resources = {  #pylint: disable = no-member
    "num_machines": 1,
    "tot_num_mpiprocs": 1,
}
builder.zeopp.metadata.options.max_wallclock_seconds = 1 * 30 * 60  #pylint: disable = no-member
builder.zeopp.metadata.options.queue_name = "default"
builder.zeopp.metadata.options.max_memory_kb = 2000000  #pylint: disable = no-member
builder.zeopp.metadata.options.withmpi = False  #pylint: disable = no-member

run(builder)

# EOF
