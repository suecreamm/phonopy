"""QE calculator interface with enhanced input file handling."""

# Copyright (C) 2014 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import re
import glob
from collections import OrderedDict
import numpy as np

from phonopy.file_IO import (
    iter_collect_forces,
    write_FORCE_CONSTANTS,
    write_force_constants_to_hdf5,
)
from phonopy.harmonic.force_constants import distribute_force_constants_by_translations
from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    get_scaled_positions_lines,
)
from phonopy.structure.atoms import PhonopyAtoms, split_symbol_and_index, symbol_map
from phonopy.structure.cells import get_primitive, get_supercell
from phonopy.units import Bohr


def normalize_tag(line):
    """Normalize tag (case-insensitive, handle brackets)"""
    line_lower = line.lower().strip()
    
    tag_match = re.search(r'(atomic_positions|cell_parameters|atomic_species|k_points|hubbard)', line_lower)
    if not tag_match:
        return line
    
    tag_name = tag_match.group(1).upper()
    
    param_match = re.search(r'(?:\(|\{)([^})]*)(?:\)|\})', line_lower)
    if param_match:
        param = param_match.group(1).strip()
    else:
        parts = line_lower.split(tag_name.lower(), 1)
        if len(parts) > 1 and parts[1].strip():
            param = parts[1].strip()
        else:
            param = ""
    
    if param:
        return f"{tag_name} ({param})"
    else:
        return tag_name


def count_atomic_positions(lines):
    """Count atoms in ATOMIC_POSITIONS section"""
    in_positions = False
    count = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('!') or line.startswith('#'):
            continue
            
        if re.search(r'atomic_positions', line.lower()):
            in_positions = True
            continue
        elif in_positions and re.search(r'(cell_parameters|k_points|hubbard)', line.lower()):
            break
        elif in_positions and not line.startswith('!') and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                count += 1
    
    return count


def extract_section(lines, section_name):
    """Extract a specific section"""
    section_pattern = re.compile(rf'{section_name}', re.IGNORECASE)
    section_lines = []
    in_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if section_pattern.search(line):
            in_section = True
            continue
        
        if in_section and re.search(r'(atomic_positions|cell_parameters|atomic_species|k_points|hubbard)', line.lower()):
            break
            
        if in_section:
            section_lines.append(line)
    
    return section_lines


def merge_qe_files(template_file, supercell_file, output_file, dimensions=None):
    """Merge QE input files and update nbnd"""
    with open(template_file, 'r') as f:
        template_lines = f.readlines()
    
    with open(supercell_file, 'r') as f:
        supercell_lines = f.readlines()
    
    merged_lines = []
    in_system = False
    nbnd_updated = False
    
    for line in template_lines:
        if re.search(r'&\s*system', line.lower()):
            in_system = True
            merged_lines.append(line)
            continue
        
        if in_system and re.search(r'/', line.strip()):
            in_system = False
            if not nbnd_updated and dimensions is not None:
                default_nbnd = 10
                prod_dim = np.prod(dimensions)
                merged_lines.append(f"  nbnd = {int(default_nbnd * prod_dim)}\n")
            merged_lines.append(line)
            continue
        
        if in_system and re.search(r'nbnd\s*=', line.lower()) and dimensions is not None:
            nbnd_value = re.search(r'nbnd\s*=\s*(\d+)', line.lower())
            if nbnd_value:
                old_nbnd = int(nbnd_value.group(1))
                prod_dim = np.prod(dimensions)
                new_nbnd = int(old_nbnd * prod_dim)
                merged_line = re.sub(r'nbnd\s*=\s*\d+', f'nbnd = {new_nbnd}', line)
                merged_lines.append(merged_line)
                nbnd_updated = True
                continue
        
        if re.search(r'ibrav\s*=', line.lower()):
            merged_lines.append(line)
            continue
            
        elif re.search(r'nat\s*=', line.lower()):
            merged_lines.append(line)
            continue
            
        elif not re.search(r'(atomic_positions|cell_parameters|k_points)', line.lower()):
            merged_lines.append(line)
    
    # Add ATOMIC_SPECIES section
    atomic_species_found = False
    for i, line in enumerate(template_lines):
        if re.search(r'atomic_species', line.lower()):
            atomic_species_found = True
            merged_lines.append(normalize_tag(line) + '\n')
            j = i + 1
            while j < len(template_lines) and not re.search(r'(atomic_positions|cell_parameters|k_points|hubbard)', template_lines[j].lower()):
                merged_lines.append(template_lines[j])
                j += 1
            break
    
    # Add CELL_PARAMETERS
    cell_params_found = False
    for line in supercell_lines:
        if re.search(r'cell_parameters', line.lower()):
            cell_params_found = True
            merged_lines.append(normalize_tag(line) + '\n')
            continue
        
        if cell_params_found:
            if re.search(r'(atomic_positions|atomic_species|k_points|hubbard)', line.lower()):
                cell_params_found = False
            else:
                merged_lines.append(line)
    
    # Add ATOMIC_POSITIONS
    atomic_pos_found = False
    atomic_pos_lines = []
    for line in supercell_lines:
        if re.search(r'atomic_positions', line.lower()):
            atomic_pos_found = True
            atomic_pos_lines.append(normalize_tag(line) + '\n')
            continue
        
        if atomic_pos_found:
            if re.search(r'(cell_parameters|atomic_species|k_points|hubbard)', line.lower()):
                atomic_pos_found = False
            else:
                atomic_pos_lines.append(line)
    
    merged_lines.extend(atomic_pos_lines)
    
    # Count atoms and update nat
    nat_count = count_atomic_positions(atomic_pos_lines)
    
    for i, line in enumerate(merged_lines):
        if re.search(r'nat\s*=', line.lower()):
            merged_lines[i] = re.sub(r'nat\s*=\s*\d+', f'nat = {nat_count}', line)
    
    # Add K_POINTS
    k_points_found = False
    for line in template_lines:
        if re.search(r'k_points', line.lower()):
            k_points_found = True
            merged_lines.append(normalize_tag(line) + '\n')
            i = template_lines.index(line) + 1
            while i < len(template_lines) and not re.search(r'(atomic_positions|cell_parameters|atomic_species|hubbard)', template_lines[i].lower()):
                merged_lines.append(template_lines[i])
                i += 1
            break
    
    # Add HUBBARD
    hubbard_found = False
    for line in template_lines:
        if re.search(r'hubbard', line.lower()):
            hubbard_found = True
            merged_lines.append(normalize_tag(line) + '\n')
            i = template_lines.index(line) + 1
            while i < len(template_lines) and not re.search(r'(atomic_positions|cell_parameters|atomic_species|k_points)', template_lines[i].lower()):
                merged_lines.append(template_lines[i])
                i += 1
            break
    
    with open(output_file, 'w') as f:
        f.writelines(merged_lines)
    
    print(f"File successfully merged: {output_file}")
    print(f"Total atoms: {nat_count}")
    if dimensions is not None and nbnd_updated:
        print(f"nbnd value updated (dimensions: {dimensions})")


def process_supercell_files(template_file, dimensions=None, pattern="supercell-*.in"):
    """Process all supercell-*.in files"""
    supercell_files = glob.glob(pattern)
    
    for supercell_file in sorted(supercell_files):
        output_file = supercell_file
        backup_file = supercell_file + ".bak"
        if os.path.exists(supercell_file):
            os.rename(supercell_file, backup_file)
        
        merge_qe_files(template_file, backup_file, output_file, dimensions)
        print(f"Processed: {supercell_file}")


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "Forces acting on atoms"
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        pwscf_forces = iter_collect_forces(
            filename, num_atoms, hook, [6, 7, 8], word="force"
        )
        if check_forces(pwscf_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                pwscf_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(pwscf_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_pwscf(filename):
    """Flexible QE file parsing"""
    with open(filename) as f:
        pwscf_in = PwscfIn(f.readlines())
    tags = pwscf_in.get_tags()
    
    for tag in ["cell_parameters", "atomic_positions", "atomic_species"]:
        if tag not in tags:
            print(f"Warning: {tag} tag not found. Proceeding with empty data.")
    
    if "cell_parameters" not in tags:
        print("Error: cell_parameters tag is required.")
        return None, None
    
    lattice = tags["cell_parameters"]
    
    if "atomic_positions" not in tags:
        print("Error: atomic_positions tag is required.")
        return None, None
    
    if pwscf_in.cartesian_positions:
        positions = [pos[1] for pos in tags["atomic_positions"]]
        scaled_positions = None
    else:
        positions = None
        scaled_positions = [pos[1] for pos in tags["atomic_positions"]]
    
    species = [pos[0] for pos in tags["atomic_positions"]]

    if "atomic_species" not in tags:
        mass_map = {s: 1.0 for s in set(species)}
        pp_map = {s: f"{s}_PP_filename" for s in set(species)}
    else:
        mass_map = {}
        pp_map = {}
        for vals in tags["atomic_species"]:
            mass_map[vals[0]] = vals[1]
            pp_map[vals[0]] = vals[2]
    
    masses = []
    pp_all_filenames = []
    for s in species:
        if s in mass_map:
            masses.append(mass_map[s])
        else:
            print(f"Warning: Mass for element {s} not defined. Using default 1.0.")
            masses.append(1.0)
        
        if s in pp_map:
            pp_all_filenames.append(pp_map[s])
        else:
            print(f"Warning: Pseudopotential for element {s} not defined. Using default.")
            pp_all_filenames.append(f"{s}_PP_filename")

    use_given_masses = False
    for symnum in species:
        symbol, num = split_symbol_and_index(symnum)
        if symbol not in symbol_map:
            print(f"Warning: Element {symbol} is not supported.")
        if num > 0:
            use_given_masses = True

    if use_given_masses:
        cell = PhonopyAtoms(
            symbols=species,
            cell=lattice,
            positions=positions,
            scaled_positions=scaled_positions,
            masses=masses,
        )
    else:
        cell = PhonopyAtoms(
            symbols=species,
            cell=lattice,
            positions=positions,
            scaled_positions=scaled_positions,
        )

    unique_symbols = []
    pp_filenames = {}
    for i, symnum in enumerate(cell.symbols):
        if symnum not in unique_symbols:
            unique_symbols.append(symnum)
            pp_filenames[symnum] = pp_all_filenames[i]

    return cell, pp_filenames


def write_pwscf(filename, cell, pp_filenames):
    """Write cell to file."""
    f = open(filename, "w")
    f.write(get_pwscf_structure(cell, pp_filenames=pp_filenames))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pp_filenames,
    pre_filename="supercell",
    width=3,
    template_file=None,
    dimensions=None
):
    """Generate supercell files and merge with template"""
    write_pwscf("%s.in" % pre_filename, supercell, pp_filenames)
    
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=pre_filename, width=width
        )
        write_pwscf(filename, cell, pp_filenames)
        
        if template_file and os.path.exists(template_file):
            merged_filename = filename + ".merged"
            merge_qe_files(template_file, filename, merged_filename, dimensions)
            os.rename(filename, filename + ".orig")
            os.rename(merged_filename, filename)
            print(f"Created merged file: {filename}")


def get_pwscf_structure(cell, pp_filenames=None):
    """Return QE structure in text."""
    lattice = cell.cell
    positions = cell.scaled_positions
    masses = cell.masses
    chemical_symbols = cell.symbols
    unique_symbols = []
    atomic_species = []
    for symbol, m in zip(chemical_symbols, masses):
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
            atomic_species.append((symbol, m))

    lines = ""
    lines += "!    ibrav = 0, nat = %d, ntyp = %d\n" % (
        len(positions),
        len(unique_symbols),
    )
    lines += "CELL_PARAMETERS bohr\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "ATOMIC_SPECIES\n"
    for symbol, mass in atomic_species:
        if pp_filenames is None:
            lines += " %2s %10.5f   %s_PP_filename\n" % (symbol, mass, symbol)
        else:
            lines += " %2s %10.5f   %s\n" % (symbol, mass, pp_filenames[symbol])
    lines += "ATOMIC_POSITIONS crystal\n"
    for i, (symbol, pos_line) in enumerate(
        zip(chemical_symbols, get_scaled_positions_lines(positions).split("\n"))
    ):
        lines += (" %2s " % symbol) + pos_line
        if i < len(chemical_symbols) - 1:
            lines += "\n"

    return lines


class PwscfIn:
    """Flexible QE input file parser"""

    _set_methods = OrderedDict(
        [
            ("ibrav", "_set_ibrav"),
            ("celldm(1)", "_set_celldm1"),
            ("nat", "_set_nat"),
            ("ntyp", "_set_ntyp"),
            ("atomic_species", "_set_atom_types"),
            ("atomic_positions", "_set_positions"),
            ("cell_parameters", "_set_lattice"),
            ("k_points", "_set_kpoints"),
        ]
    )

    def __init__(self, lines):
        """Init method."""
        self._tags = {}
        self._current_tag_name = None
        self._values = None
        self._cartesian_positions = False
        self._collect(lines)

    @property
    def cartesian_positions(self):
        """Return True if positions are in Cartesian coordinates."""
        return self._cartesian_positions

    def get_tags(self):
        """Return tags."""
        return self._tags

    def _collect(self, lines):
        elements = {}
        tag_name = None

        for line in lines:
            _line = line.split("!")[0]
            
            lower_line = _line.lower()
            if "atomic_positions" in lower_line or "cell_parameters" in lower_line:
                words = []
                if "atomic_positions" in lower_line:
                    tag_name = "atomic_positions"
                    words.append(tag_name)
                    unit = lower_line.replace("atomic_positions", "").strip()
                    unit = unit.replace("(", "").replace(")", "").replace("{", "").replace("}", "").strip()
                    if unit:
                        words.append(unit)
                    else:
                        words.append("crystal")
                elif "cell_parameters" in lower_line:
                    tag_name = "cell_parameters"
                    words.append(tag_name)
                    unit = lower_line.replace("cell_parameters", "").strip()
                    unit = unit.replace("(", "").replace(")", "").replace("{", "").replace("}", "").strip()
                    if unit:
                        words.append(unit)
                    else:
                        words.append("bohr")
            elif "atomic_species" in lower_line:
                words = _line.split()
                tag_name = "atomic_species"
            elif "k_points" in lower_line:
                tag_name = "k_points"
                words = [tag_name]
                unit = lower_line.replace("k_points", "").strip()
                unit = unit.replace("(", "").replace(")", "").replace("{", "").replace("}", "").strip()
                if unit:
                    words.append(unit)
            else:
                line_replaced = _line.replace("=", " ").replace(",", " ")
                words = line_replaced.split()

            for val in words:
                if val.lower() in self._set_methods:
                    tag_name = val.lower()
                    elements[tag_name] = [val]
                elif tag_name is not None:
                    elements[tag_name].append(val)

        # Check required tags and set defaults
        missing_tags = []
        for tag_name in ["ibrav", "nat", "ntyp"]:
            if tag_name not in elements:
                missing_tags.append(tag_name)
        
        if missing_tags:
            if "ibrav" in missing_tags:
                elements["ibrav"] = ["ibrav", "0"]
            if "ntyp" in missing_tags:
                if "atomic_species" in elements:
                    ntyp = len(elements["atomic_species"]) // 3
                    elements["ntyp"] = ["ntyp", str(ntyp)]
                else:
                    elements["ntyp"] = ["ntyp", "1"]
            if "nat" in missing_tags:
                if "atomic_positions" in elements:
                    nat = (len(elements["atomic_positions"]) - 1) // 4
                    elements["nat"] = ["nat", str(nat)]
                else:
                    elements["nat"] = ["nat", "1"]

        # Set tag values
        for tag_name in self._set_methods:
            if tag_name in elements:
                self._current_tag_name = elements[tag_name][0]
                self._values = elements[tag_name][1:]
                if tag_name in self._set_methods.keys():
                    try:
                        getattr(self, self._set_methods[tag_name])()
                    except Exception as e:
                        print(f"Warning: Error setting {tag_name} ({str(e)}). Using default values.")

    def _set_ibrav(self):
        try:
            ibrav = int(self._values[0])
            if ibrav != 0:
                print(f"Warning: ibrav={ibrav} is not supported. Setting to ibrav=0.")
                ibrav = 0
        except (ValueError, IndexError):
            print("Warning: Could not parse ibrav value. Setting to ibrav=0.")
            ibrav = 0

        self._tags["ibrav"] = ibrav

    def _set_celldm1(self):
        try:
            self._tags["celldm(1)"] = float(self._values[0])
        except (ValueError, IndexError):
            print("Warning: Could not parse celldm(1) value. Using default 1.0.")
            self._tags["celldm(1)"] = 1.0

    def _set_nat(self):
        try:
            self._tags["nat"] = int(self._values[0])
        except (ValueError, IndexError):
            if "atomic_positions" in self._tags:
                self._tags["nat"] = len(self._tags["atomic_positions"])
            else:
                print("Warning: Could not parse nat value. Using default 1.")
                self._tags["nat"] = 1

    def _set_ntyp(self):
        try:
            self._tags["ntyp"] = int(self._values[0])
        except (ValueError, IndexError):
            if "atomic_species" in self._tags:
                self._tags["ntyp"] = len(self._tags["atomic_species"])
            else:
                print("Warning: Could not parse ntyp value. Using default 1.")
                self._tags["ntyp"] = 1

    def _set_lattice(self):
        """Process CELL_PARAMETERS"""
        unit = self._values[0].lower() if self._values else "bohr"
        factor = 1.0
        
        if unit == "alat":
            if "celldm(1)" not in self._tags:
                print("Warning: celldm(1) not defined but alat unit is used. Setting default 1.0.")
                self._tags["celldm(1)"] = 1.0
            factor = self._tags["celldm(1)"]
        elif "angstrom" in unit:
            factor = 1.0 / Bohr
        elif "bohr" in unit:
            factor = 1.0
        else:
            print(f"Warning: Unit '{unit}' is not supported. Using bohr.")
            unit = "bohr"
            factor = 1.0

        if len(self._values[1:]) < 9:
            print(f"Warning: Not enough values in CELL_PARAMETERS. Using default values.")
            lattice = np.eye(3) * 10.0
        else:
            try:
                lattice = np.reshape([float(x) for x in self._values[1:10]], (3, 3))
            except ValueError:
                print("Warning: Could not parse CELL_PARAMETERS values. Using default values.")
                lattice = np.eye(3) * 10.0
        
        self._tags["cell_parameters"] = lattice * factor

    def _set_positions(self):
        """Process ATOMIC_POSITIONS"""
        unit = self._values[0].lower() if self._values else "crystal"
        factor = 1.0
        
        if "angstrom" in unit:
            factor = 1.0 / Bohr
            self._cartesian_positions = True
        elif "bohr" in unit:
            self._cartesian_positions = True
        elif "crystal" not in unit:
            print(f"Warning: Unit '{unit}' is not supported. Using crystal.")
            unit = "crystal"
            self._cartesian_positions = False

        natom = self._tags.get("nat", 0)
        pos_vals = self._values[1:]
        
        positions = []
        i = 0
        
        while i < len(pos_vals):
            if i + 4 <= len(pos_vals):
                species = pos_vals[i]
                try:
                    coords = [factor * float(x) for x in pos_vals[i+1:i+4]]
                    positions.append([species, coords])
                except ValueError:
                    print(f"Warning: Could not parse coordinates for atom #{i//4+1}.")
                i += 4
            else:
                print(f"Warning: Incomplete data in ATOMIC_POSITIONS section.")
                break
        
        if natom > 0 and len(positions) != natom:
            print(f"Warning: nat={natom} but found {len(positions)} atomic positions.")
            self._tags["nat"] = len(positions)
        
        self._tags["atomic_positions"] = positions

    def _set_atom_types(self):
        """Process ATOMIC_SPECIES"""
        num_types = self._tags.get("ntyp", 0)
        species = []
        
        for i in range(0, len(self._values), 3):
            if i + 3 <= len(self._values):
                try:
                    mass = float(self._values[i+1])
                except ValueError:
                    print(f"Warning: Could not parse mass for element '{self._values[i]}'. Using default 1.0.")
                    mass = 1.0
                    
                species.append([
                    self._values[i],
                    mass,
                    self._values[i+2],
                ])
        
        if num_types > 0 and len(species) != num_types:
            print(f"Warning: ntyp={num_types} but found {len(species)} element types.")
            self._tags["ntyp"] = len(species)
            
        self._tags["atomic_species"] = species

    def _set_kpoints(self):
        """Process K_POINTS"""
        self._tags["k_points"] = self._values


class PH_Q2R:
    """Parse QE/q2r output and create supercell force constants array.

    A simple usage is as follows:

    ---------
    #!/usr/bin/env python

    cell, _ = read_pwscf(primcell_filename)
    q2r = PH_Q2R(q2r_filename)
    q2r.run(cell)
    q2r.write_force_constants()
    ---------

    To save memory/storage space of force constants, the shape of
    force constants array is (n_uatom, n_satom, 3, 3), where u_atom is
    the number of atoms in unit cell and n_satom is the number of
    atoms in super cell, i.e., u_atom * prod(dim). When using this
    force constants data from phonopy with primitive cell that is
    differnt from unit cell, force constants have to be regenerated
    for the primitive cell, which is not done in this class.

    Treatment of non-analytical term correction (NAC) is different
    between phonopy and QE. For insulator, QE automatically calculate
    dielectric constant and Born effective charges at PH calculation
    when q-point mesh sampling mode ('ldisp = .true.'). These data are
    written in the Gamma point dynamical matrix file (probably
    numbered as 1 among files). When running q2r.x, these files are
    read including the dielectric constant and Born effective charges,
    and the real space force constants where QE-NAC treatment is done
    are written to the q2r output file. This is not that phonopy
    expects. Therefore the dielectric constant and Born effective
    charges data have to be removed manually from the Gamma point
    dynamical matrix file before running q2r.x. Alternatively Gamma
    point only PH calculation with 'epsil = .false.' can generate the
    dynamical matrix file without the dielectric constant and Born
    effective charges data. So it is possible to replace the Gamma
    point file by this Gamma point only file to run q2r.x for phonopy.

    Attributes
    ----------
    fc : ndarray
        Force constants in either compact or full matrix.
        dtype='double'
        shape=(natom_prim, natom_super, 3, 3) for compact fc or
              (natom_super, natom_super, 3, 3) for full fc
    dimenstion : ndarray
        Supercell dimensions (not matrix)
        dtype='intc'
        shape=(3,)
    epsilon : ndarray
        Dielectric constant tensor
        dtype='double'
        shape=(3, 3)
    born : ndarray
        Born effective charges
        dtype='double'
        shape=(natom_prim, 3, 3)
    primitive : Primitive
        Primitive cell
    supercell : Supercell
        Supercell

    """

    def __init__(self, filename, symprec=1e-5):
        """Init method."""
        self.fc = None
        self.dimension = None
        self.epsilon = None
        self.borns = None
        self.primitive = None
        self.supercell = None
        self._symprec = symprec
        self._filename = filename

    def run(self, cell, is_full_fc=False, parse_fc=True):
        """Read force constants from QE output"""
        with open(self._filename) as f:
            fc_dct = self._parse_q2r(f)
            self.dimension = fc_dct["dimension"]
            self.epsilon = fc_dct["dielectric"]
            self.borns = fc_dct["born"]
            if parse_fc:
                (self.fc, self.primitive, self.supercell) = self._arrange_supercell_fc(
                    cell, fc_dct["fc"], is_full_fc=is_full_fc
                )

    def write_force_constants(self, fc_format="hdf5"):
        """Write force constatns to file in hdf5."""
        if self.fc is not None:
            if fc_format == "hdf5":
                write_force_constants_to_hdf5(self.fc, p2s_map=self.primitive.p2s_map)
            else:
                write_FORCE_CONSTANTS(self.fc)

    def _parse_q2r(self, f):
        """Parse q2r output file."""
        natom, dim, epsilon, borns = self._parse_parameters(f)
        fc_dct = {
            "fc": self._parse_fc(f, natom, dim),
            "dimension": dim,
            "dielectric": epsilon,
            "born": borns,
        }
        return fc_dct

    def _parse_parameters(self, f):
        line = f.readline()
        ntype, natom, ibrav = (int(x) for x in line.split()[:3])
        if ibrav == 0:
            for _ in range(3):
                line = f.readline()
        for _ in range(ntype + natom):
            line = f.readline()
        line = f.readline()
        if line.strip() == "T":
            epsilon, borns = self._parse_born(f, natom)
        else:
            epsilon = None
            borns = None
        line = f.readline()
        dim = np.array([int(x) for x in line.split()], dtype="intc")

        return natom, dim, epsilon, borns

    def _parse_born(self, f, natom):
        epsilon = np.zeros((3, 3), dtype="double", order="C")
        borns = np.zeros((natom, 3, 3), dtype="double", order="C")
        for i in range(3):
            line = f.readline()
            epsilon[i, :] = [float(x) for x in line.split()]
        for i in range(natom):
            line = f.readline()
            for j in range(3):
                line = f.readline()
                borns[i, j, :] = [float(x) for x in line.split()]
        return epsilon, borns

    def _parse_fc(self, f, natom, dim):
        """Parse force constants"""
        ndim = np.prod(dim)
        fc = np.zeros((natom, natom * ndim, 3, 3), dtype="double", order="C")
        for k, ll, i, j in np.ndindex((3, 3, natom, natom)):
            line = f.readline()
            for i_dim in range(ndim):
                line = f.readline()
                fc[j, i * ndim + i_dim, ll, k] = float(line.split()[3])
        return fc

    def _arrange_supercell_fc(self, cell, q2r_fc, is_full_fc=False):
        dim = self.dimension
        q2r_spos = self._get_q2r_positions(cell)
        scell = get_supercell(cell, np.diag(dim))
        pcell = get_primitive(scell, np.diag(1.0 / dim))

        diff = cell.get_scaled_positions() - pcell.get_scaled_positions()
        diff -= np.rint(diff)
        assert (np.abs(diff) < 1e-8).all()
        assert scell.get_number_of_atoms() == len(q2r_spos)

        site_map = self._get_site_mapping(
            scell.get_scaled_positions(), q2r_spos, scell.get_cell()
        )
        natom = pcell.get_number_of_atoms()
        ndim = np.prod(dim)
        natom_s = natom * ndim

        if is_full_fc:
            fc = np.zeros((natom_s, natom_s, 3, 3), dtype="double", order="C")
            p2s = pcell.get_primitive_to_supercell_map()
            fc[p2s, :] = q2r_fc[:, site_map]
            distribute_force_constants_by_translations(fc, pcell)
        else:
            fc = np.zeros((natom, natom_s, 3, 3), dtype="double", order="C")
            fc[:, :] = q2r_fc[:, site_map]

        return fc, pcell, scell

    def _get_q2r_positions(self, cell):
        dim = self.dimension
        natom = cell.get_number_of_atoms()
        ndim = np.prod(dim)
        spos = np.zeros((natom * np.prod(dim), 3), dtype="double", order="C")
        trans = [x[::-1] for x in np.ndindex(tuple(dim[::-1]))]
        for i, p in enumerate(cell.get_scaled_positions()):
            spos[i * ndim : (i + 1) * ndim] = (trans + p) / dim
        return spos

    def _get_site_mapping(self, spos, q2r_spos, lattice):
        site_map = []
        for _, p in enumerate(spos):
            diff = q2r_spos - p
            diff -= np.rint(diff)
            distances = np.sqrt(np.sum(np.dot(diff, lattice) ** 2, axis=1))
            indices = np.where(distances < self._symprec)[0]
            assert len(indices) == 1, "%s" % indices
            site_map.append(indices[0])

        assert len(np.unique(site_map)) == len(spos)

        return np.array(site_map)
