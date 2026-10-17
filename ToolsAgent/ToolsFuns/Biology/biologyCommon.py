# -*- coding: utf-8 -*-
"""Local protein sequence analysis tools."""
from functools import wraps
import re
import traceback
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

# Kyte & Doolittle, J. Mol. Biol. 1982, 157, 105-132.
KD_SCALE = ProteinAnalysis("").kd_scale if hasattr(ProteinAnalysis(""), 'kd_scale') else {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# Small neutral residues allowed at cleavage positions -3 and -1.
_SMALL_NEUTRAL = set("AGSTVC")
_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _normalize_sequence(sequence):
    """Normalize whitespace and case; accept the 20 standard amino acids."""
    try:
        if not isinstance(sequence, str):
            return None, {"status": "error", "message": "序列必须是字符串"}

        normalized = re.sub(r"\s+", "", sequence).upper()
        if not normalized:
            return None, {"status": "error", "message": "序列不能为空"}

        invalid = sorted(set(normalized) - _VALID_AMINO_ACIDS)
        if invalid:
            invalid_text = ", ".join(invalid)
            return None, {
                "status": "error",
                "message": f"序列包含非法或非标准氨基酸字符: {invalid_text}",
            }
        return normalized, None
    except Exception as exc:
        logger.error("蛋白质序列校验失败: %s", exc, exc_info=True)
        return None, {"status": "error", "message": f"序列校验失败: {exc}"}


def _defensive_tool(func):
    """Validate sequence input and return structured tool errors."""
    @wraps(func)
    def wrapper(sequence):
        try:
            normalized, error = _normalize_sequence(sequence)
            if error:
                return error
            return func(normalized)
        except Exception as exc:
            logger.error("%s 执行失败: %s", func.__name__, exc, exc_info=True)
            return {"status": "error", "message": f"{func.__name__} 执行失败: {exc}"}

    return wrapper


@_defensive_tool
def compute_pI_mw(sequence: str):
    """Compute pI and molecular weight locally with Biopython."""
    try:
        analyser = ProteinAnalysis(sequence)
        pI = round(analyser.isoelectric_point(), 2)
        Mw = round(analyser.molecular_weight(), 2)
        logger.info("本地计算成功: pI=%s, Mw=%s", pI, Mw)
    except Exception as exc:
        logger.error("pI/Mw 本地计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"pI/Mw 计算失败: {exc}"}

    return {
        "status": "success",
        "sequence_length": len(sequence),
        "pI": pI,
        "molecular_weight": Mw
    }


@_defensive_tool
def compute_gravy(sequence: str):
    """Compute GRAVY on the Kyte-Doolittle scale."""
    try:
        gravy = round(ProteinAnalysis(sequence).gravy(), 3)
        return {
            "status": "success",
            "sequence_length": len(sequence),
            "gravy": gravy,
            "method": "Biopython ProteinAnalysis.gravy (Kyte-Doolittle scale)",
        }
    except Exception as exc:
        logger.error("GRAVY 计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"GRAVY 计算失败: {exc}"}


@_defensive_tool
def compute_secondary_structure_fraction(sequence: str):
    """Estimate helix, turn and sheet propensities from residue composition."""
    try:
        helix, turn, sheet = ProteinAnalysis(sequence).secondary_structure_fraction()
        return {
            "status": "success",
            "sequence_length": len(sequence),
            "helix_fraction": round(helix, 4),
            "turn_fraction": round(turn, 4),
            "sheet_fraction": round(sheet, 4),
            "method": "Biopython ProteinAnalysis.secondary_structure_fraction",
            "note": (
                "结果是基于残基组成的二级结构倾向，不是基于三维结构的真实指派；"
                "三类残基集合可能重叠，比例之和不要求等于1。"
            ),
        }
    except Exception as exc:
        logger.error("二级结构倾向计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"二级结构倾向计算失败: {exc}"}


@_defensive_tool
def compute_amino_acid_composition(sequence: str):
    """Count the 20 standard amino acids and their percentages."""
    try:
        raw_counts = ProteinAnalysis(sequence).count_amino_acids()
        counts = {amino_acid: int(raw_counts.get(amino_acid, 0))
                  for amino_acid in _AMINO_ACID_ORDER}
        percentages = {
            amino_acid: round(counts[amino_acid] / len(sequence) * 100.0, 4)
            for amino_acid in _AMINO_ACID_ORDER
        }
        return {
            "status": "success",
            "sequence_length": len(sequence),
            "counts": counts,
            "percentages": percentages,
            "units": "percent",
            "method": "Biopython count_amino_acids + normalized percentage",
        }
    except Exception as exc:
        logger.error("氨基酸组成计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"氨基酸组成计算失败: {exc}"}


@_defensive_tool
def compute_instability_index(sequence: str):
    """Compute the instability index; use 40 as the empirical cutoff."""
    try:
        instability_index = round(ProteinAnalysis(sequence).instability_index(), 2)
        result = {
            "status": "success",
            "sequence_length": len(sequence),
            "instability_index": instability_index,
            "is_stable": instability_index < 40.0,
            "threshold": 40.0,
            "method": "Biopython ProteinAnalysis.instability_index",
        }
        if len(sequence) < 10:
            result["note"] = "序列短于10个残基，经验稳定性判断需谨慎解释。"
        return result
    except Exception as exc:
        logger.error("不稳定性指数计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"不稳定性指数计算失败: {exc}"}


@_defensive_tool
def compute_extinction_coefficient(sequence: str):
    """Compute reduced and oxidized molar extinction coefficients at 280 nm."""
    try:
        reduced, oxidized = ProteinAnalysis(sequence).molar_extinction_coefficient()
        return {
            "status": "success",
            "sequence_length": len(sequence),
            "reduced_molar_extinction_coefficient": int(reduced),
            "oxidized_molar_extinction_coefficient": int(oxidized),
            "wavelength_nm": 280,
            "units": "M^-1 cm^-1",
            "method": "Biopython ProteinAnalysis.molar_extinction_coefficient",
        }
    except Exception as exc:
        logger.error("消光系数计算失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"消光系数计算失败: {exc}"}


@_defensive_tool
def check_protein_toxicity(sequence: str):
    """Screen against the local toxicity database using Smith-Waterman alignment."""
    try:
        from app.tools.safety.toxicity_checker import check_protein_toxicity_bio

        result = check_protein_toxicity_bio(sequence)
        if not isinstance(result, dict):
            return {"status": "error", "message": "毒性检测返回格式无效"}
        return result
    except Exception as exc:
        logger.error("毒性检测失败: %s", exc, exc_info=True)
        return {"status": "error", "message": f"毒性检测失败: {exc}"}


@_defensive_tool
def predict_signalpeptide(sequence: str):
    """Screen for an N-terminal signal peptide using hydrophobicity and the (-3,-1) rule."""
    if len(sequence) < 20:
        return {"status": "success", "has_signal_peptide": False,
                "confidence": "N/A", "note": "序列过短 (<20 aa)，无法分析"}

    # Scan the N-terminal signal peptide region.
    scan = sequence[:min(70, len(sequence))]

    # Use an 8-residue hydrophobicity window.
    h_window = 8
    h_scores = ProteinAnalysis(scan).protein_scale(KD_SCALE, h_window)
    best_h_start = max(range(1, len(h_scores)), key=lambda i: h_scores[i], default=0)
    best_h_score = h_scores[best_h_start]
    best_h_end = best_h_start + h_window

    # Check candidate cleavage sites against the (-3,-1) rule.
    cleavage_site = None
    best_cs_score = 0
    for pos in range(best_h_end + 2, min(best_h_end + 15, len(sequence))):
        cs = 0
        if sequence[pos - 2] in _SMALL_NEUTRAL:
            cs += 2
        if sequence[pos] in _SMALL_NEUTRAL:
            cs += 2
        if sequence[pos - 1] != 'P':  # Exclude proline at position -2.
            cs += 1
        if cs > best_cs_score:
            best_cs_score = cs
            cleavage_site = pos + 1  # 1-indexed

    # Check N-region positive charges.
    n_region = sequence[:best_h_start] if best_h_start > 0 else sequence[:3]
    n_charge = n_region.count('R') + n_region.count('K')

    total = 0
    evidence = []

    if sequence[0] == 'M':
        total += 1; evidence.append("N端Met(+1)")
    if best_h_score > 1.5:
        total += 3; evidence.append(f"强疏水核心(KD={best_h_score:.2f}, +3)")
    elif best_h_score > 0.8:
        total += 2; evidence.append(f"中等疏水核心(KD={best_h_score:.2f}, +2)")
    if best_cs_score >= 4:
        total += 3; evidence.append("(-3,-1)规则完全匹配(+3)")
    elif best_cs_score >= 3:
        total += 2; evidence.append("(-3,-1)规则部分匹配(+2)")
    if n_charge >= 1:
        total += 1; evidence.append(f"n-区正电荷×{n_charge}(+1)")

    has_sp = total >= 5
    conf = "high" if total >= 7 else ("medium" if total >= 5 else "low")

    result = {
        "status": "success",
        "has_signal_peptide": has_sp,
        "confidence": conf,
        "score": f"{total}/8",
        "evidence": evidence,
        "method": "von Heijne 三区域模型 + Kyte-Doolittle 疏水性 + (-3,-1) 切割位点规则"
    }
    if has_sp and cleavage_site:
        result["cleavage_site"] = cleavage_site
        result["signal_peptide_sequence"] = sequence[:cleavage_site]
        result["h_region"] = {"start": best_h_start + 1, "end": best_h_end,
                              "avg_hydropathy": round(best_h_score, 2)}
    return result


@_defensive_tool
def predict_transmembrane(sequence: str):
    """Screen transmembrane regions using a 19-residue Kyte-Doolittle window."""
    METHOD = "Kyte-Doolittle hydropathy (Biopython protein_scale), window=19, threshold=1.6"
    if len(sequence) < 20:
        return {"status": "success", "transmembrane_regions": [], "count": 0,
                "note": "序列过短", "method": METHOD}

    window, threshold = 19, 1.6
    half = window // 2

    scores = ProteinAnalysis(sequence).protein_scale(KD_SCALE, window)
    above = [i + half for i, s in enumerate(scores) if s > threshold]

    if not above:
        return {"status": "success", "transmembrane_regions": [], "count": 0,
                "topology": "可能为水溶性蛋白", "method": METHOD}

    # Merge nearby peaks, allowing gaps of up to five residues.
    regions = []
    start = end = above[0]
    for pos in above[1:]:
        if pos - end <= 5:
            end = pos
        else:
            regions.append((start, end))
            start = end = pos
    regions.append((start, end))

    # Expand peaks to their window bounds; require at least 15 residues.
    valid = []
    for s, e in regions:
        a_start = max(0, s - half)
        a_end = min(len(sequence) - 1, e + half)
        length = a_end - a_start + 1
        if length >= 15:
            seg = sequence[a_start:a_end + 1]
            avg_kd = round(ProteinAnalysis(seg).protein_scale(KD_SCALE, min(len(seg), window))[len(seg)//2 - (min(len(seg),window)//2)], 2) \
                     if len(seg) >= window else round(sum(scores[max(0,a_start-half):a_end-half+1]) / max(1, a_end-a_start), 2)
            valid.append({
                "start": a_start + 1, "end": a_end + 1,
                "length": length, "sequence": seg,
                "avg_hydropathy": avg_kd, "type": "跨膜螺旋"
            })

    n = len(valid)
    topo = ("可能为水溶性蛋白" if n == 0 else
            "单次跨膜蛋白 (I型或II型)" if n == 1 else
            f"多次跨膜蛋白 ({n}次跨膜)" if n <= 7 else
            f"高度跨膜蛋白 ({n}次跨膜)")

    return {"status": "success", "transmembrane_regions": valid, "count": n,
            "topology": topo, "method": METHOD}


@_defensive_tool
def predict_solubility(sequence: str):
    """Estimate solubility from GRAVY, instability, pI and charged residue content."""
    if len(sequence) < 10:
        return {"status": "error", "message": "序列过短 (<10 aa)，无法分析"}

    try:
        pa = ProteinAnalysis(sequence)
        gravy = round(pa.gravy(), 3)
        instability = round(pa.instability_index(), 2)
        pI = round(pa.isoelectric_point(), 2)
        mw = round(pa.molecular_weight(), 2)

        aa_cnt = pa.count_amino_acids()
        charged = sum(aa_cnt.get(aa, 0) for aa in "DEKRH")
        charged_frac = round(charged / len(sequence), 3)

        score = 50
        if   gravy < -0.5: score += 20
        elif gravy < 0:    score += 10
        elif gravy > 0.5:  score -= 15

        if   instability < 30: score += 15
        elif instability < 40: score += 10
        elif instability > 50: score -= 10

        pi_diff = abs(pI - 7.0)
        if   pi_diff > 2: score += 10
        elif pi_diff > 1: score += 5
        else:             score -= 5

        if   charged_frac > 0.25: score += 10
        elif charged_frac > 0.15: score += 5

        if   len(sequence) < 200: score += 5
        elif len(sequence) > 500: score -= 5

        score = max(0, min(100, score))
        soluble = score >= 50

        level = ("高溶解度" if score >= 75 else
                 "中等溶解度" if score >= 50 else
                 "低溶解度" if score >= 30 else
                 "极低溶解度（建议添加增溶标签如 MBP/SUMO）")

        return {
            "status": "success",
            "soluble": soluble,
            "solubility_score": score,
            "solubility_level": level,
            "pI": pI,
            "parameters": {
                "GRAVY": gravy,
                "instability_index": instability,
                "pI": pI,
                "molecular_weight": mw,
                "charged_residue_fraction": charged_frac,
                "sequence_length": len(sequence)
            },
            "method": "Multi-parameter scoring (GRAVY + Instability Index + pI + charged fraction)"
        }
    except Exception as e:
        return {"status": "error", "message": f"溶解度分析异常: {str(e)}"}


@_defensive_tool
def protein_analysis(sequence: str):
    """Run the combined sequence analyses."""
    return {
        "status": "success",
        "basic_info": compute_pI_mw(sequence),
        "signal_peptide": predict_signalpeptide(sequence),
        "transmembrane": predict_transmembrane(sequence),
        "solubility": predict_solubility(sequence)
    }


if __name__ == "__main__":
    test_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    try:
        result = protein_analysis(test_seq)
        print("蛋白分析结果：")
        print(result)
    except Exception:
        traceback.print_exc()
