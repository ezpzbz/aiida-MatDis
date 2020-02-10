# These are very task specifc calcfuntions which would be used
# to extract and calculate the separation performance descriptors.
# I normally apply them on the output_dict of hts workchain during
# the query.

from aiida.engine import calcfunction
from aiida.orm import Dict

@calcfunction
def get_spd_xe_exhaled(components, wc_output):
    """
    Extracting the separation performance descriptors from the
    output_dict of a GCMC simulation for Xenon recovery from exhaled anesthetic
    gas.
    """
    wc_dict = wc_output.get_dict()

    comp1 = components['comp1']['name']
    comp2 = components['comp1']['name']
    comp3 = components['comp1']['name']
    comp4 = components['comp1']['name']

    y1 = wc_dict['mol_fraction'][comp1]
    y2 = wc_dict['mol_fraction'][comp2]
    y3 = wc_dict['mol_fraction'][comp1]
    y4 = wc_dict['mol_fraction'][comp2]

    n_1_des = wc_dict["isotherm"]["loading_absolute_average"][comp1][0]
    n_1_ads = wc_dict["isotherm"]["loading_absolute_average"][comp1][1]
    n_2_des = wc_dict["isotherm"]["loading_absolute_average"][comp2][0]
    n_2_ads = wc_dict["isotherm"]["loading_absolute_average"][comp2][1]

    y1 = wc_dict['mol_fraction'][comp1]
    y2 = wc_dict['mol_fraction'][comp2]

    s_1_2_ads = (n_1_ads / n_2_ads) * (y2 * y1)
    s_1_2_des = (n_1_des / n_2_des) * (y2 * y1)

    wc1 = n_1_ads - n_1_des
    wc2 = n_2_ads - n_2_des

    regen = (wc1 / n_1_ads) * 100

    afm = wc1 * ((s_1_2_ads ** 2) / s_1_2_des)

    output_dict = {
        'sel_1_2_gcmc':
        'S_1_2': s_1_2_ads,
        'wc1': wc1,
    }

    return Dict(dict=output_dict)
