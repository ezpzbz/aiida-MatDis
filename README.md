# AiiDA-MatDis

You may find three types of AiiDA WorkChains in this
repository.

* WorkChains which are developed by myself.
* WorkChains which have been developed on top of [AiiDA LSMO WorkChains](https://github.com/danieleongari/aiida-lsmo) for the purpose of my research but are provided here for now due to considerable deviation from
original design.
* Information about other WorkChains from other groups which we are using for our research .

# Prerequisites
To use `AiiDA-MatDis` workchains use the following packages:
* [`aiida-lsmo`](https://github.com/lsmo-epfl/aiida-lsmo)
* [`aiida-porousmaterials`](https://github.com/pzarabadip/aiida-porousmaterials)
* [`aiida-raspa`](https://github.com/yakutovicha/aiida-raspa)
* [`aiida-zeopp`](https://github.com/ltalirz/aiida-zeopp)

**NOTE** Workchains currently are tested with [`AiiDA Core v1.0.1`](https://github.com/aiidateam/aiida-core/tree/v1.0.1)
# Installation
`git clone https://github.com/pzarabadip/aiida-MatDis`

`pip install -e .`
# Brief Description of WorkChains
* `VoronoiEnergyWorkChain`: Calculation of Voronoi energies using `aiida-zeopp` and `aiida-porousmaterials`
* `VLCCWorkChain`: Construction of vapor-liquid coexistence curves using `aiida-raspa`

**NOTE** Following workchains are developed on top of [`LSMO IsothermWC`](https://github.com/lsmo-epfl/aiida-lsmo) to add multi-component feature with some changes. If you are interested in calculating single-component adsorption isotherms please visit [`aiida-lsmo` repository](https://github.com/lsmo-epfl/aiida-lsmo). You also can find other interesting and very well developed workchains there.

* `HTSWorkChain`: Calculation of multi component adsorption for a few pressure points using `aiida-zeopp` and `aiida-raspa` plugins. It submits all `RASPA` calculations in parallel.
* `HTSMultiTWorkChain`: Similar to `HTSWorkChain` for obtatining the multi-component adsorption values at different temperatures.
* `HTSEvWorkChain`: It is a modified version of `HTSWorkChain` which takes the output dictionary of `VoronoiEnergyWorkChain` and performs the consequent calculations.
* `MultiCompIsothermWorkChain`: It can be used to construct a full range multi-component adsorption isotherm. Here the difference with `HTSWorkChain` is that similar to `IsothermWC` from `aiida-lsmo`, calculations at higher pressure points are using the final configuration from the previous point.

**NOTE** Isotherm wokchains all benefit from `FFBuilder` of `aiida-lsmo` to construct force field definition files.

# Citation
If you benefit from `AiiDA` in your research, please cite [this paper](https://www.sciencedirect.com/science/article/pii/S0927025615005820?via%3Dihub) by *G. Pizzi et. al.*

If you use above-mentioned workchains for gas adsorption, please cite [this paper](https://pubs.acs.org/doi/10.1021/acscentsci.9b00619) by *D. Ongari et. al.*

# Documentation
Documentation for each WorkChain including the references, sources, know-hows, etc are provided in
the [Wiki](https://github.com/pzarabadip/aiida-MatDis/wiki) (In Progress).

# Acknowledgment
I would like to thank the funding received from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Actions and cofinancing by the South Moravian Region under agreement 665860. This software reflects only the authors’ view and the EU is not responsible for any use that may be made of the information it contains.

![aiida-MatDis](ackn_logo.png)
