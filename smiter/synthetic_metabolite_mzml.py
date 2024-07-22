"""Main module."""

import io
import pathlib
import time
import warnings
from pprint import pformat
from typing import Callable, Dict, List, Tuple, Union
from collections import Counter
import re
import numpy as np
from intervaltree import IntervalTree
from loguru import logger
from tqdm import tqdm
from psims.mzml import MzMLWriter
from molmass import Formula
import smiter
from smiter.fragmentation_functions import AbstractFragmentor
from smiter.lib import (
    calc_mz,
    check_mzml_params,
    check_peak_properties,
    peak_properties_to_csv,
)
from smiter.noise_functions import AbstractNoiseInjector
from smiter.peak_distribution import distributions

warnings.filterwarnings("ignore")

def adding_noise_for_each_scan(scan,noise_intensity=200, noise_length=1500): # TODO: noise_intensity, noise_length 作为参数传入
    """
    为每个scan添加噪声
    :param scan: 扫描
    :param noise_intensity: 噪声注入器,noise_length:噪声长度
    :return: 添加噪声后的scan
    """
    noise_i = np.random.uniform(0, noise_intensity, noise_length)
    noise_mz = np.random.uniform(50, 1500, noise_length)
    scan.mz = np.concatenate((scan.mz, noise_mz))
    scan.i = np.concatenate((scan.i, noise_i))
    return scan

def weighted_average(mz_values, intensities):
    """
    计算加权平均值
    :param mz_values: 质荷比列表
    :param intensities: 强度列表
    :return: 加权平均值
    """
    return sum(mz * intensity for mz, intensity in zip(mz_values, intensities)) / sum(intensities), sum(intensities)

def calculate_averages(mz_values, intensities):
    averaged_mz_values = []
    averaged_intensities = []
    i = 0

    while i < len(mz_values) - 1:
        if abs((mz_values[i+1] - mz_values[i]) / mz_values[i]) < 1/20000: #TODO: 20000为分辨率，作为参数传入
            averaged_mz_values.append(weighted_average([mz_values[i], mz_values[i+1]], [intensities[i], intensities[i+1]])[0])
            averaged_intensities.append(weighted_average([mz_values[i], mz_values[i+1]], [intensities[i], intensities[i+1]])[1])
            i += 2  # 跳过下一个值，因为已经计算了加权平均值
        else:
            averaged_mz_values.append(mz_values[i])
            averaged_intensities.append(intensities[i]) 
            i += 1

    # 如果最后一个值没有被计算加权平均值，将其添加到结果列表中
    if i == len(mz_values) - 1:
        averaged_mz_values.append(mz_values[i])
        averaged_intensities.append(intensities[i])
    return averaged_mz_values, averaged_intensities

class Scan(dict):
    """Summary."""

    def __init__(self, data: dict = None):
        """Summary.

        Args:
            dict (TYPE): Description
        """
        if data is not None:
            self.update(data)

    @property
    def mz(self):
        """Summary."""
        v = self.get("mz", None)
        return v

    @mz.setter
    def mz(self, mz):
        """Summary."""
        self["mz"] = mz

    @property
    def i(self):
        """Summary."""
        v = self.get("i", None)
        return v

    @i.setter
    def i(self, i):
        """Summary."""
        self["i"] = i

    @property
    def id(self):
        """Summary."""
        v = self.get("id", None)
        return v

    @property
    def precursor_mz(self):
        """Summary."""
        v = self.get("precursor_mz", None)
        return v

    @property
    def precursor_i(self):
        """Summary."""
        v = self.get("precursor_i", None)
        return v

    @property
    def precursor_charge(self):
        """Summary."""
        v = self.get("precursor_charge", None)
        return v

    @property
    def retention_time(self):
        """Summary."""
        v = self.get("rt", None)
        return v

    @property
    def ms_level(self):
        """Summary.

        Returns:
            TYPE: Description
        """
        v = self.get("ms_level", None)
        return v


def generate_interval_tree(peak_properties):
    """Conctruct an interval tree containing the elution windows of the analytes.

    Args:
        peak_properties (dict): Description

    Returns:
        IntervalTree: Description
    """
    tree = IntervalTree()
    for key, data in peak_properties.items():
        start = data["scan_start_time"]
        end = start + data["peak_width"]
        tree[start:end] = key
    return tree


# @profile
def write_mzml(
    file: Union[str, io.TextIOWrapper],
    peak_properties: Dict[str, dict],
    fragmentor: AbstractFragmentor,
    noise_injector: AbstractNoiseInjector,
    mzml_params: Dict[str, Union[int, float, str]],
) -> str:
    """Write mzML file with chromatographic peaks and fragment spectra for the given molecules.

    Args:
        file (Union[str, io.TextIOWrapper]): Description
        molecules (List[str]): Description
        fragmentation_function (Callable[[str], List[Tuple[float, float]]], optional): Description
        peak_properties (Dict[str, dict], optional): Description
    """
    # check params and raise Exception(s) if necessary
    logger.info("Start generating mzML")
    mzml_params = check_mzml_params(mzml_params)
    peak_properties = check_peak_properties(peak_properties)

    interval_tree = generate_interval_tree(peak_properties)

    filename = file if isinstance(file, str) else file.name
    scans = []

    trivial_names = {}
    charges = set()
    for key, val in peak_properties.items():
        trivial_names[val["chemical_formula"]] = key
        charges.add(val["charge"])
    # trivial_names = {
    # val["chemical_formula"]: key for key, val in peak_properties.items()
    # }
    # dicts are sorted, language specification since python 3.7+

    isotopologue_lib = generate_molecule_isotopologue_lib(
        peak_properties, trivial_names=trivial_names, charges=charges
    )
    scans, scan_dict = generate_scans(
        isotopologue_lib,
        peak_properties,
        interval_tree,
        fragmentor,
        noise_injector,
        mzml_params,
    )
    logger.info("Delete interval tree")
    del interval_tree
    write_scans(file, scans)
    if not isinstance(file, str):
        file_path = file.name
    else:
        file_path = file
    path = pathlib.Path(file_path)
    summary_path = path.parent.resolve() / "molecule_summary.csv"
    peak_properties_to_csv(peak_properties, summary_path)
    return filename


# @profile
def rescale_intensity(
    i: float, rt: float, molecule: str, peak_properties: dict, isotopologue_lib: dict
):
    """Rescale intensity value for a given molecule according to scale factor and distribution function.

    Args:
        i (TYPE): Description
        rt (TYPE): Description
        molecule (TYPE): Description
        peak_properties (TYPE): Description
        isotopologue_lib (TYPE): Description

    Returns:
        TYPE: Description
    """
    scale_func = peak_properties[f"{molecule}"]["peak_function"]


    if scale_func == "gauss":
        mu = (
            peak_properties[f"{molecule}"]["scan_start_time"]
            + 0.5 * peak_properties[f"{molecule}"]["peak_width"]
        )
        dist_scale_factor = [scale_func](
            rt,
            mu=mu,
            sigma=peak_properties[f"{molecule}"]["peak_params"].get(
                "sigma", peak_properties[f"{molecule}"]["peak_width"] / 10
            ),
        )
    elif scale_func == "gamma":
        dist_scale_factor = distributions[scale_func](
            rt,
            a=peak_properties[f"{molecule}"]["peak_params"]["a"],
            scale=peak_properties[f"{molecule}"]["peak_params"]["scale"],
        )
    elif scale_func == "gauss_tail":
        mu = (
            peak_properties[f"{molecule}"]["scan_start_time"]
            + 0.3 * peak_properties[f"{molecule}"]["peak_width"]
        )
        dist_scale_factor = distributions[scale_func](
            rt,
            mu=mu,
            sigma=0.12 * (rt - peak_properties[f"{molecule}"]["scan_start_time"]) + 2,
            scan_start_time=peak_properties[f"{molecule}"]["scan_start_time"],
        )
    elif scale_func is None:
        dist_scale_factor = 1
    
    
    # print(f"dist_scale_factor: {dist_scale_factor}")
    # TODO use ionization_effiency here
    i *= (
        dist_scale_factor
        * peak_properties[f"{molecule}"].get("peak_scaling_factor", 1e3)
        * peak_properties[f"{molecule}"].get("ionization_effiency", 1)
    )
    # print(f"i: {i}")
    return i


def generate_scans(
    isotopologue_lib: dict,
    peak_properties: dict,
    interval_tree: IntervalTree,
    fragmentor: AbstractFragmentor,
    noise_injector: AbstractNoiseInjector,
    mzml_params: dict,
):
    """Summary.

    Args:
        isotopologue_lib (TYPE): Description
        peak_properties (TYPE): Description
        fragmentation_function (A): Description
        mzml_params (TYPE): Description
    """
    logger.info("Initialize chimeric spectra counter")
    chimeric_count = 0
    chimeric = Counter()
    logger.info("Start generating scans")
    t0 = time.time()
    gradient_length = mzml_params["gradient_length"]
    ms_rt_diff = mzml_params.get("ms_rt_diff", 0.03)
    t: float = 0

    mol_scan_dict: Dict[str, Dict[str, list]] = {}
    scans: List[Tuple[Scan, List[Scan]]] = []
    # i: int = 0
    spec_id: int = 1
    de_tracker: Dict[str, int] = {}
    de_stats: dict = {}

    mol_scan_dict = {
        mol: {"ms1_scans": [], "ms2_scans": []} for mol in isotopologue_lib
    }
    molecules = list(isotopologue_lib.keys())

    progress_bar = tqdm(
        total=gradient_length,
        desc="Generating scans",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}",
    )
    while t < gradient_length:
        scan_peaks: List[Tuple[float, float]] = []
        scan_peaks = {}
        mol_i = []
        mol_monoisotopic = {}
        candidates = interval_tree.at(t)
        # print(len(candidates))
        for mol in candidates:
            # if len(candidates) > 1:
            mol = mol.data
            mol_plus = f"{mol}"
            mz = np.array(isotopologue_lib[mol]["mz"])
            # print(mz)
            intensity = np.array(isotopologue_lib[mol]["i"])
            # print(intensity)
            intensity = rescale_intensity(
                intensity, t, mol, peak_properties, isotopologue_lib
            )

            mask = intensity > mzml_params["min_intensity"]
            intensity = intensity[mask]

            # clip max intensity
            intensity = np.clip(
                intensity, a_min=None, a_max=mzml_params.get("max_intensity", 1e10)
            )

            mz = mz[mask]
            mol_peaks = list(zip(mz, intensity))
            mol_peaks = {round(mz, 6): _i for mz, _i in list(zip(mz, intensity))}
            # !FIXED! multiple molecules which share mz should have summed up intensityies for that shared mzs
            if len(mol_peaks) > 0:
                mol_i.append((mol, mz[0], sum(intensity)))
                # scan_peaks.extend(mol_peaks)
                for mz, intensity in mol_peaks.items():
                    if mz in scan_peaks:
                        scan_peaks[mz] += intensity
                    else:
                        scan_peaks[mz] = intensity
                mol_scan_dict[mol]["ms1_scans"].append(spec_id)
                highest_peak = max(mol_peaks.items(), key=lambda x: x[1])
                mol_monoisotopic[mol] = {
                    "mz": highest_peak[0],
                    "i": highest_peak[1],
                }
        scan_peaks = sorted(list(scan_peaks.items()), key=lambda x: x[1])
        if len(scan_peaks) > 0:
            mz, inten = zip(*scan_peaks)
        else:
            mz, inten = [], []
        
        s = Scan(
            {
                "mz": np.array(mz),
                "i": np.array(inten),
                "id": spec_id,
                "rt": t,
                "ms_level": 1,
            }
        )
        prec_scan_id = spec_id
        spec_id += 1

        sorting = s.mz.argsort()
        s.mz = s.mz[sorting]
        s.i = s.i[sorting]

        # add noise
        s = noise_injector.inject_noise(s)

        # i += 1
        scans.append((s, []))
        t += ms_rt_diff
        progress_bar.update(ms_rt_diff)

        if t > gradient_length:
            break

        # fragment_spec_index = 0
        # max_ms2_spectra = mzml_params.get("max_ms2_spectra", 10)
        # if len(mol_i) < max_ms2_spectra:
        #     max_ms2_spectra = len(mol_i)
        # ms2_scan = None
        # mol_i = sorted(mol_i, key=lambda x: x[2], reverse=True)
        # logger.debug(f"All molecules eluting: {len(mol_i)}")
        # logger.debug(f"currently # fragment spectra {len(scans[-1][1])}")

        # mol_i = [
        #     mol
        #     for mol in mol_i
        #     if (de_tracker.get(mol[0], None) is None)
        #     or (t - de_tracker[mol[0]]) > mzml_params["dynamic_exclusion"]
        # ]
        # logger.debug(f"All molecules eluting after DE filtering: {len(mol_i)}")
        # while len(scans[-1][1]) != max_ms2_spectra:
        #     logger.debug(f"Frag spec index {fragment_spec_index}")
        #     if fragment_spec_index > len(mol_i) - 1:
        #         # we evaluated fragmentation for every potential mol
        #         # and all will be skipped
        #         logger.debug(f"All possible mol are skipped due to DE")
        #         break
        #     mol = mol_i[fragment_spec_index][0]
        #     _mz = mol_i[fragment_spec_index][1]
        #     _intensity = mol_i[fragment_spec_index][2]
        #     fragment_spec_index += 1
        #     mol_plus = f"{mol}"
        #     all_mols_in_mz_and_rt_window = [
        #         mol.data
        #         for mol in candidates
        #         if (
        #             abs(isotopologue_lib[mol.data]["mz"][0] - _mz)
        #             < mzml_params["isolation_window_width"]
        #         )
        #     ]
        #     if len(all_mols_in_mz_and_rt_window) > 1:
        #         chimeric_count += 1
        #         chimeric[len(all_mols_in_mz_and_rt_window)] += 1
        #     if mol is None:
        #         # dont add empty MS2 scans but have just a much scans as precursors
        #         breakpoint()
        #         ms2_scan = Scan(
        #             {
        #                 "mz": np.array([]),
        #                 "i": np.array([]),
        #                 "rt": t,
        #                 "id": spec_id,
        #                 "precursor_mz": 0,
        #                 "precursor_i": 0,
        #                 "precursor_charge": 1,
        #                 "precursor_scan_id": prec_scan_id,
        #                 "ms_level": 2,
        #             }
        #         )
        #         spec_id += 1
        #         t += ms_rt_diff
        #         progress_bar.update(ms_rt_diff)

        #         if t > gradient_length:
        #             break
            # elif (peak_properties[mol_plus]["scan_start_time"] <= t) and (
            #     (
            #         peak_properties[mol_plus]["scan_start_time"]
            #         + peak_properties[mol_plus]["peak_width"]
            #     )
            #     >= t
            # ):
            #     # fragment all molecules in isolation and rt window
            #     # check if molecule needs to be fragmented according to dynamic_exclusion rule
            #     if (
            #         de_tracker.get(mol, None) is None
            #         or (t - de_tracker[mol]) > mzml_params["dynamic_exclusion"]
            #     ):
            #         logger.debug("Generate Fragment spec")
            #         de_tracker[mol] = t
            #         if mol not in de_stats:
            #             de_stats[mol] = {"frag_events": 0, "frag_spec_ids": []}
            #         de_stats[mol]["frag_events"] += 1
            #         de_stats[mol]["frag_spec_ids"].append(spec_id)
            #         peaks = fragmentor.fragment(all_mols_in_mz_and_rt_window)
            #         frag_mz = peaks[:, 0]
            #         frag_i = peaks[:, 1]
            #         ms2_scan = Scan(
            #             {
            #                 "mz": frag_mz,
            #                 "i": frag_i,
            #                 "rt": t,
            #                 "id": spec_id,
            #                 "precursor_mz": mol_monoisotopic[mol]["mz"],
            #                 "precursor_i": mol_monoisotopic[mol]["i"],
            #                 "precursor_charge": peak_properties[mol]["charge"],
            #                 "precursor_scan_id": prec_scan_id,
            #                 "ms_level": 2,
                #         }
                #     )
                #     spec_id += 1
                #     ms2_scan.i = rescale_intensity(
                #         ms2_scan.i, t, mol, peak_properties, isotopologue_lib
                #     )
                #     ms2_scan = noise_injector.inject_noise(ms2_scan)
                #     ms2_scan.i *= 0.5
                # else:
                #     logger.debug(f"Skip {mol} due to dynamic exclusion")
                #     continue
                # t += ms_rt_diff
                # progress_bar.update(ms_rt_diff)
                # if t > gradient_length:
            #     #     break
            # else:
            #     logger.debug(f"Skip {mol} since not in RT window")
            #     continue
            # if mol is not None:
            #     mol_scan_dict[mol]["ms2_scans"].append(spec_id)
            # if ms2_scan is None:
            #     # there are molecules in mol_i
            #     # however all molecules are excluded from fragmentation_function
            #     # => Don't do a scan and break the while loop
            #     # => We should rather continue and try to fragment the next mol!
            #     logger.debug(f"Continue and fragment next mol since MS2 scan is None")
            #     continue
            # if (
            #     len(ms2_scan.mz) > -1
            # ):  # TODO -1 to also add empty ms2 specs; 0 breaks tests currently ....
            #     sorting = ms2_scan.mz.argsort()
            #     ms2_scan.mz = ms2_scan.mz[sorting]
            #     ms2_scan.i = ms2_scan.i[sorting]
            #     logger.debug(f"Append MS2 scan with {mol}")
            #     scans[-1][1].append(ms2_scan)
    progress_bar.close()
    t1 = time.time()
    logger.info("Finished generating scans")
    logger.info(f"Generating scans took {t1-t0:.2f} seconds")
    logger.info(f"Found {chimeric_count} chimeric scans")

    return scans, mol_scan_dict


# @profile
def generate_molecule_isotopologue_lib(
    peak_properties: Dict[str, dict],
    charges: List[int] = None,
    trivial_names: Dict[str, str] = None,
):
    """Summary.

    Args:
        molecules (TYPE): Description
    """
    logger.info("Generate Isotopolgue Library")
    start = time.time()
    duplicate_formulas: Dict[str, List[str]] = {}
    for key in peak_properties:
        duplicate_formulas.setdefault(
            peak_properties[key]["chemical_formula"], []
        ).append(key)
    
    if charges is None:
        charges = [1]
    if len(peak_properties) > 0:
        
        molecules = [d["chemical_formula"] for d in peak_properties.values()]
        keys = [d for d in peak_properties.keys()]
        charges = [d["charge"] for d in peak_properties.values()]
        clean_molecules =  [re.sub(r'[+()]', '', item) for item in molecules]
        reduced_lib = {}
        for i in range(len(clean_molecules)):
            f_ = clean_molecules[i]
            triv = keys[i]
            charge = charges[i]
            formula = Formula(f_)+Formula("H"*charge)
            spectrum = formula.spectrum()
            mz = []
            inty = []
            for key, items in sorted(spectrum.items()):
                i = items.intensity
                m = items.mass
                k = items.massnumber
                mz.append(m/charge)
                inty.append(i/100) 
            reduced_lib[triv] = {
                "mz": mz,
                "i": inty,
                        }
    else:
        reduced_lib = {}
    tmp = {}
    for mol in reduced_lib:
        cc = peak_properties[mol]["chemical_formula"]
        for triv in duplicate_formulas[cc]:
            if triv not in reduced_lib:
                tmp[triv] = reduced_lib[mol]
    reduced_lib.update(tmp)
    logger.info(
        f"Generating IsotopologueLibrary took {(time.time() - start)/60} minutes"
    )
    return reduced_lib

# @profile
def write_scans(
    file: Union[str, io.TextIOWrapper], scans: List[Tuple[Scan, List[Scan]]]
) -> None:
    """Generate given scans to mzML file.

    Args:
        file (Union[str, io.TextIOWrapper]): Description
        scans (List[Tuple[Scan, List[Scan]]]): Description

    Returns:
        None: Description
    """
    t0 = time.time()
    logger.info("Start writing Scans")
    ms1_scans = len(scans)
    ms2_scans = 0
    ms2_scan_list = []
    for s in scans:
        ms2_scans += len(s[1])
        ms2_scan_list.append(len(s[1]))
    logger.info("Write {0} MS1 and {1} MS2 scans".format(ms1_scans, ms2_scans))
    id_format_str = "controllerType=0 controllerNumber=1 scan={i}"
    
    with MzMLWriter(file) as writer:
        # Add default controlled vocabularies
        writer.controlled_vocabularies()
        writer.file_description()
        writer.software_list([
        writer.Software(version="0.0.0", id='psims', params=['custom unreleased software tool']),
        writer.Software(version="3.0.24052", id='pwiz', params=['ProteoWizard software'])
        ])
        writer.instrument_configuration_list([
            writer.InstrumentConfiguration(id="IC1", component_list=writer.ComponentList([
                writer.Source(params=['electrospray ionization'], order=1),
                writer.Analyzer(params=['quadrupole'], order=2),
                writer.Analyzer(params=['orbitrap'], order=3),
                writer.Detector(params=['inductive detector'], order=4)
            ])),
    
        ])
        writer.data_processing_list([
            writer.DataProcessing(processing_methods=[
                dict(order=0, software_reference='psims', params=['Conversion to mzML']),
            ], id=1)
        ])
        writer.format()
        # Open the run and spectrum list sections
        time_array = []
        intensity_array = []
        with writer.run(id="Simulated Run", instrument_configuration="IC1"): # add instrument_configuration
            spectrum_count = len(scans) + sum([len(products) for _, products in scans])
            with writer.spectrum_list(count=spectrum_count,data_processing_method = "pwiz_Reader_Thermo_conversion"): # add data_processing_method
                for scan, products in scans:
                    # Write Precursor scan
                    adding_noise_for_each_scan(scan)
                    # 排序并计算加权平均值
                    indices = np.argsort(scan.mz)
                    scan.mz = scan.mz[indices]
                    scan.i = scan.i[indices]
                    mz,i = calculate_averages(scan.mz, scan.i)
                    try:
                        index_of_max_i = np.argmax(i)
                        max_i = i[index_of_max_i]
                        mz_at_max_i = mz[index_of_max_i]
                    except ValueError:
                        mz_at_max_i = 0
                        max_i = 0
                    spec_tic = sum(scan.i)

                    writer.write_spectrum(
                        np.around(mz,5),
                        np.around(i, 5),
                        id=id_format_str.format(i=scan.id),
                        params=[
                            "MS1 Spectrum",
                            {"ms level": 1},
                            {
                                "scan start time": scan.retention_time,
                            #     # "unit_name": "seconds",
                            },
                            {"total ion current": spec_tic},
                            {"base peak m/z": round(mz_at_max_i,5), "unitName": "m/z"},
                            {
                                "base peak intensity": round(max_i,5),
                                "unitName": "number of detector counts",
                            },
                        ],
                        scan_params=[{"scan start time":  round(scan.retention_time, 4)}],
                        scan_window_list=[(50, 1500)], #TODO: 50, 1500 作为参数传入
                        compression="none",
                        
                    )
                    time_array.append( round(scan.retention_time, 4))
                    intensity_array.append(spec_tic)
                    # Write MSn scans
                    for prod in products:
                        writer.write_spectrum(
                            np.around(prod.mz, 5),
                            np.around(prod.i, 5),
                            id=id_format_str.format(i=prod.id),
                            params=[
                                "MSn Spectrum",
                                {"ms level": 2},
                                {
                                    "scan start time": prod.retention_time,
                                    # "unit_name": "seconds",
                                },
                                {"total ion current": sum(np.around(prod.i,5))},
                            ],
                            scan_params=[{"scan start time": round(prod.retention_time,4)}],
                            compression="none",
                            precursor_information={
                                "mz": np.around(prod.precursor_mz,5),
                                "intensity": np.around(prod.precursor_i,5),
                                "charge": prod.precursor_charge,
                                "scan_id": id_format_str.format(i=scan.id),
                                "spectrum_reference": id_format_str.format(i=scan.id),
                                "activation": ["HCD", {"collision energy": 25.0}],
                            },
                        )
            with writer.chromatogram_list(count=1, data_processing_method = "pwiz_Reader_Thermo_conversion"): # add data_processing_method
                writer.write_chromatogram(
                    time_array,
                    intensity_array,
                    id="TIC",
                    chromatogram_type="total ion current",
                    compression="none",
                )
    t1 = time.time()
    logger.info(f"Writing mzML took {(t1-t0)/60:.2f} minutes")
    return
