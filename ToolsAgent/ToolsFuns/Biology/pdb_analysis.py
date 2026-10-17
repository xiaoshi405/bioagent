# -*- coding: utf-8 -*-
"""Local PDB structure and chain analysis."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

from app.core.config import Config
from app.core.logger import get_logger


logger = get_logger("PDBAnalysis", "pdb_analysis.log")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_PDB_FILE_BYTES = Config().MAX_PDB_UPLOAD_MB * 1024 * 1024
MAX_ANM_RESIDUES = 500
WATER_NAMES = {"HOH", "WAT", "DOD"}
MODIFIED_RESIDUE_MAP = {"MSE": "M", "SEC": "U", "PYL": "O"}


def _error(message: str) -> dict[str, Any]:
    try:
        return {"status": "error", "message": str(message)}
    except Exception as exc:
        return {"status": "error", "message": f"构造PDB错误响应失败: {exc}"}


def _resolve_pdb_path(file_path: str) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Check the PDB path, project boundary, extension and size limit."""
    try:
        if not isinstance(file_path, str) or not file_path.strip():
            return None, _error("PDB文件路径不能为空")

        path = Path(file_path.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve()

        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return None, _error("PDB文件必须位于项目目录内")

        if resolved.suffix.lower() not in {".pdb", ".ent"}:
            return None, _error("仅支持 .pdb 或 .ent 结构文件")
        if not resolved.is_file():
            return None, _error(f"PDB文件不存在: {resolved.name}")
        file_size = resolved.stat().st_size
        if file_size <= 0:
            return None, _error("PDB文件为空")
        if file_size > MAX_PDB_FILE_BYTES:
            return None, _error(
                f"PDB文件超过大小上限 {MAX_PDB_FILE_BYTES // (1024 * 1024)} MB"
            )
        return resolved, None
    except Exception as exc:
        logger.error("校验PDB文件失败: %s", exc, exc_info=True)
        return None, _error(f"校验PDB文件失败: {exc}")


def _load_structure(file_path: str):
    """Load a PDB structure or return a structured error."""
    try:
        path, error = _resolve_pdb_path(file_path)
        if error or path is None:
            return None, None, error or _error("PDB路径校验失败")
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        structure = parser.get_structure(path.stem, str(path))
        models = list(structure.get_models())
        if not models:
            return None, None, _error("PDB文件中没有可解析的模型")
        return structure, path, None
    except Exception as exc:
        logger.error("解析PDB结构失败: %s", exc, exc_info=True)
        return None, None, _error(f"解析PDB结构失败: {exc}")


def _residue_to_letter(residue) -> str:
    """Map standard and supported modified residues to one-letter codes."""
    try:
        residue_name = str(residue.get_resname()).strip().upper()
        if residue_name in MODIFIED_RESIDUE_MAP:
            return MODIFIED_RESIDUE_MAP[residue_name]
        return seq1(residue_name, undef_code="X")
    except Exception as exc:
        logger.warning("残基转换失败，使用X占位: %s", exc)
        return "X"


def _protein_residues(chain) -> list:
    """Return standard and supported modified amino acid residues."""
    try:
        return [residue for residue in chain.get_residues() if is_aa(residue, standard=False)]
    except Exception as exc:
        logger.error("提取蛋白质残基失败: %s", exc, exc_info=True)
        return []


def _parse_secondary_structure_annotations(
    path: Path, protein_residue_keys: set[tuple[str, int]]
) -> dict[str, Any]:
    """Read HELIX and SHEET annotations from a PDB file."""
    try:
        helix_keys: set[tuple[str, int]] = set()
        sheet_keys: set[tuple[str, int]] = set()
        helix_records = 0
        sheet_records = 0

        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("HELIX") and len(line) >= 38:
                start_chain = line[19].strip() or "_"
                end_chain = line[31].strip() or "_"
                if start_chain != end_chain:
                    continue
                start_residue = int(line[21:25].strip())
                end_residue = int(line[33:37].strip())
                helix_records += 1
                for residue_number in range(start_residue, end_residue + 1):
                    key = (start_chain, residue_number)
                    if key in protein_residue_keys:
                        helix_keys.add(key)
            elif line.startswith("SHEET") and len(line) >= 38:
                start_chain = line[21].strip() or "_"
                end_chain = line[32].strip() or "_"
                if start_chain != end_chain:
                    continue
                start_residue = int(line[22:26].strip())
                end_residue = int(line[33:37].strip())
                sheet_records += 1
                for residue_number in range(start_residue, end_residue + 1):
                    key = (start_chain, residue_number)
                    if key in protein_residue_keys:
                        sheet_keys.add(key)

        # Prefer HELIX when annotations overlap to avoid double counting.
        sheet_keys -= helix_keys
        total = len(protein_residue_keys)
        other_count = max(0, total - len(helix_keys) - len(sheet_keys))
        denominator = max(1, total)
        return {
            "method": "PDB HELIX/SHEET annotation records",
            "annotation_records_present": bool(helix_records or sheet_records),
            "helix_record_count": helix_records,
            "sheet_record_count": sheet_records,
            "helix_residue_count": len(helix_keys),
            "sheet_residue_count": len(sheet_keys),
            "other_or_unassigned_residue_count": other_count,
            "helix_percent": round(len(helix_keys) / denominator * 100.0, 2),
            "sheet_percent": round(len(sheet_keys) / denominator * 100.0, 2),
            "other_or_unassigned_percent": round(other_count / denominator * 100.0, 2),
        }
    except Exception as exc:
        logger.error("解析PDB二级结构注释失败: %s", exc, exc_info=True)
        return {
            "method": "PDB HELIX/SHEET annotation records",
            "status": "error",
            "message": f"解析二级结构注释失败: {exc}",
        }


def extract_pdb_sequences(file_path: str) -> dict[str, Any]:
    """Extract chain sequences from the first PDB model."""
    try:
        structure, path, error = _load_structure(file_path)
        if error or structure is None or path is None:
            return error or _error("PDB解析失败")

        first_model = next(structure.get_models())
        chains = {}
        total_residues = 0
        for chain in first_model.get_chains():
            residues = _protein_residues(chain)
            if not residues:
                continue
            chain_id = str(chain.id).strip() or "_"
            sequence = "".join(_residue_to_letter(residue) for residue in residues)
            chains[chain_id] = {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "unknown_residue_count": sequence.count("X"),
            }
            total_residues += len(sequence)

        if not chains:
            return _error("PDB文件中没有可提取的蛋白质链")
        logger.info("PDB序列提取完成 | file=%s | chains=%s", path.name, len(chains))
        return {
            "status": "success",
            "file_name": path.name,
            "model_used": int(first_model.id),
            "chain_count": len(chains),
            "total_residue_count": total_residues,
            "chains": chains,
            "method": "Bio.PDB residue mapping from the first model",
        }
    except Exception as exc:
        logger.error("提取PDB序列失败: %s", exc, exc_info=True)
        return _error(f"提取PDB序列失败: {exc}")


def select_pdb_chain(file_path: str, chain_id: str) -> dict[str, Any]:
    """Return the sequence of the explicitly requested chain."""
    try:
        if not isinstance(chain_id, str) or not chain_id.strip():
            return _error("PDB链选择必须提供非空chain_id")

        requested_chain_id = chain_id.strip()
        sequence_map = extract_pdb_sequences(file_path)
        if not isinstance(sequence_map, dict):
            return _error("PDB链序列提取结果格式无效")
        if sequence_map.get("status") == "error":
            return sequence_map

        chains = sequence_map.get("chains")
        if not isinstance(chains, dict) or not chains:
            return _error("PDB链序列映射为空")

        available_chain_ids = sorted(str(value) for value in chains)
        if requested_chain_id not in chains:
            return _error(
                f"PDB中不存在链 {requested_chain_id}；"
                f"可用链: {', '.join(available_chain_ids)}"
            )

        selected = chains.get(requested_chain_id)
        if not isinstance(selected, dict):
            return _error(f"PDB链 {requested_chain_id} 的序列记录格式无效")

        sequence = selected.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            return _error(f"PDB链 {requested_chain_id} 不包含可用蛋白质序列")

        logger.info(
            "PDB链选择完成 | file=%s | chain=%s | length=%s",
            sequence_map.get("file_name", Path(str(file_path)).name),
            requested_chain_id,
            len(sequence),
        )
        return {
            "status": "success",
            "file_name": sequence_map.get("file_name", Path(str(file_path)).name),
            "model_used": sequence_map.get("model_used"),
            "chain_id": requested_chain_id,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "unknown_residue_count": int(
                selected.get("unknown_residue_count", sequence.count("X"))
            ),
            "available_chain_ids": available_chain_ids,
            "method": (
                "Bio.PDB first-model chain extraction with explicit chain_id selection"
            ),
        }
    except Exception as exc:
        logger.error("选择PDB蛋白质链失败: %s", exc, exc_info=True)
        return _error(f"选择PDB蛋白质链失败: {exc}")


def analyze_pdb_structure(file_path: str) -> dict[str, Any]:
    """Summarize chains, residues, atoms, ligands and geometry."""
    try:
        structure, path, error = _load_structure(file_path)
        if error or structure is None or path is None:
            return error or _error("PDB解析失败")

        models = list(structure.get_models())
        first_model = models[0]
        chain_summaries = {}
        protein_residue_keys: set[tuple[str, int]] = set()
        protein_coordinates = []
        protein_atom_count = 0
        hetero_residue_count = 0
        water_count = 0

        for chain in first_model.get_chains():
            chain_id = str(chain.id).strip() or "_"
            protein_residues = _protein_residues(chain)
            if protein_residues:
                sequence = "".join(_residue_to_letter(residue) for residue in protein_residues)
                chain_summaries[chain_id] = {
                    "residue_count": len(protein_residues),
                    "sequence_length": len(sequence),
                    "unknown_residue_count": sequence.count("X"),
                }
                for residue in protein_residues:
                    protein_residue_keys.add((chain_id, int(residue.id[1])))
                    for atom in residue.get_atoms():
                        protein_coordinates.append(np.asarray(atom.coord, dtype=float))
                        protein_atom_count += 1

            for residue in chain.get_residues():
                if is_aa(residue, standard=False):
                    continue
                if str(residue.get_resname()).strip().upper() in WATER_NAMES:
                    water_count += 1
                elif str(residue.id[0]).strip():
                    hetero_residue_count += 1

        if not protein_coordinates:
            return _error("PDB文件中没有蛋白质原子坐标")

        coordinates = np.vstack(protein_coordinates)
        center = coordinates.mean(axis=0)
        radius_of_gyration = float(
            np.sqrt(np.mean(np.sum((coordinates - center) ** 2, axis=1)))
        )
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        secondary_structure = _parse_secondary_structure_annotations(
            path, protein_residue_keys
        )

        logger.info(
            "PDB结构摘要完成 | file=%s | chains=%s | residues=%s | atoms=%s",
            path.name,
            len(chain_summaries),
            len(protein_residue_keys),
            protein_atom_count,
        )
        return {
            "status": "success",
            "file_name": path.name,
            "model_count": len(models),
            "model_used": int(first_model.id),
            "chain_count": len(chain_summaries),
            "chains": chain_summaries,
            "protein_residue_count": len(protein_residue_keys),
            "protein_atom_count": protein_atom_count,
            "hetero_residue_count": hetero_residue_count,
            "water_count": water_count,
            "center_of_geometry_angstrom": [round(float(value), 3) for value in center],
            "radius_of_gyration_angstrom": round(radius_of_gyration, 3),
            "bounding_box_angstrom": {
                "minimum": [round(float(value), 3) for value in minimum],
                "maximum": [round(float(value), 3) for value in maximum],
            },
            "secondary_structure_annotation": secondary_structure,
            "method": "Bio.PDB first-model parsing and unweighted atomic geometry",
        }
    except Exception as exc:
        logger.error("PDB结构分析失败: %s", exc, exc_info=True)
        return _error(f"PDB结构分析失败: {exc}")


def calculate_protein_anm(
    file_path: str,
    n_modes: int = 10,
    cutoff: float = 12.0,
    chain_id: Optional[str] = None,
) -> dict[str, Any]:
    """Compute the lowest nonzero ANM modes from one chain of C-alpha atoms."""
    try:
        if not isinstance(n_modes, int) or not 1 <= n_modes <= 20:
            return _error("n_modes必须是1到20之间的整数")
        cutoff_value = float(cutoff)
        if not 4.0 <= cutoff_value <= 20.0:
            return _error("cutoff必须在4.0到20.0埃之间")

        structure, path, error = _load_structure(file_path)
        if error or structure is None or path is None:
            return error or _error("PDB解析失败")

        first_model = next(structure.get_models())
        candidates = []
        requested_chain = str(chain_id).strip() if chain_id else None
        for chain in first_model.get_chains():
            current_chain_id = str(chain.id).strip() or "_"
            if requested_chain and current_chain_id != requested_chain:
                continue
            ca_atoms = []
            for residue in _protein_residues(chain):
                if "CA" in residue:
                    ca_atoms.append(residue["CA"])
            if ca_atoms:
                candidates.append((current_chain_id, ca_atoms))

        if not candidates:
            if requested_chain:
                return _error(f"链 {requested_chain} 中没有可用于ANM的Cα原子")
            return _error("PDB文件中没有可用于ANM的Cα原子")

        # Default to the longest chain to keep the dense ANM matrix bounded.
        selected_chain, ca_atoms = max(candidates, key=lambda item: len(item[1]))
        residue_count = len(ca_atoms)
        if residue_count < 4:
            return _error("ANM至少需要4个Cα原子")
        if residue_count > MAX_ANM_RESIDUES:
            return _error(
                f"单链Cα数量超过ANM上限 {MAX_ANM_RESIDUES}，请指定较短的链"
            )

        coordinates = np.vstack(
            [np.asarray(atom.coord, dtype=float) for atom in ca_atoms]
        )
        hessian = np.zeros((3 * residue_count, 3 * residue_count), dtype=float)
        contact_count = 0
        cutoff_squared = cutoff_value ** 2

        for first_index in range(residue_count - 1):
            for second_index in range(first_index + 1, residue_count):
                displacement = coordinates[second_index] - coordinates[first_index]
                distance_squared = float(np.dot(displacement, displacement))
                if distance_squared <= 0.0 or distance_squared > cutoff_squared:
                    continue
                block = np.outer(displacement, displacement) / distance_squared
                first_slice = slice(3 * first_index, 3 * first_index + 3)
                second_slice = slice(3 * second_index, 3 * second_index + 3)
                hessian[first_slice, first_slice] += block
                hessian[second_slice, second_slice] += block
                hessian[first_slice, second_slice] -= block
                hessian[second_slice, first_slice] -= block
                contact_count += 1

        if contact_count == 0:
            return _error("给定cutoff下没有形成Cα接触网络")

        eigenvalues = np.linalg.eigvalsh(hessian)
        tolerance = max(1e-8, float(np.max(np.abs(eigenvalues))) * 1e-8)
        positive_eigenvalues = eigenvalues[eigenvalues > tolerance]
        if positive_eigenvalues.size == 0:
            return _error("ANM Hessian没有得到非零模态")

        selected_values = positive_eigenvalues[:n_modes]
        logger.info(
            "ANM计算完成 | file=%s | chain=%s | residues=%s | modes=%s",
            path.name,
            selected_chain,
            residue_count,
            len(selected_values),
        )
        return {
            "status": "success",
            "file_name": path.name,
            "chain_id": selected_chain,
            "residue_count": residue_count,
            "ca_atom_count": residue_count,
            "cutoff_angstrom": cutoff_value,
            "contact_count": contact_count,
            "requested_mode_count": n_modes,
            "returned_mode_count": int(len(selected_values)),
            "zero_or_near_zero_mode_count": int(len(eigenvalues) - len(positive_eigenvalues)),
            "eigenvalues": [round(float(value), 8) for value in selected_values],
            "method": "C-alpha anisotropic network model, gamma=1, dense NumPy eigensolver",
        }
    except Exception as exc:
        logger.error("计算PDB ANM失败: %s", exc, exc_info=True)
        return _error(f"计算PDB ANM失败: {exc}")
