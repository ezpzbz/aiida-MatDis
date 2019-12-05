# -*- coding: utf-8 -*-
"""Copied from aiida-lsmo utils"""
from __future__ import absolute_import
from .multiply_unitcell import get_replciation_factors
from .general import (get_molecules_input_dict,
                      get_molecule_dict,
                      get_ff_parameters,
                      get_atomic_radii,
                      get_geometric_output,
                      get_pressure_list,
                      choose_pressure_points,
                      get_output_parameters,
                      get_vlcc_output,
                      get_temperature_points,
                      extract_merge_outputs)
from .other import update_workchain_params, dict_merge
