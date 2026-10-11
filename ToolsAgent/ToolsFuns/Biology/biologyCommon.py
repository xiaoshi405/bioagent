# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:08:56 2026
@author: huawei

修改记录：
- 2026-07-17：增加统一序列校验和异常兜底；pI/Mw 改为 Biopython 本地计算，
  避免正式链路依赖 Expasy 网络页面。
- 2026-07-19：新增 GRAVY、二级结构倾向、氨基酸组成、不稳定性指数、
  消光系数和毒性相似性检测六个离线原子工具。
"""
#生物工具算法文件
from functools import wraps
import re
import traceback
import logging
# 引入 Biopython 做本地兜底计算
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

# Biopython 内置的 Kyte-Doolittle 疏水性指数表（protein_scale 调用时传入）
# 参考: Kyte & Doolittle, J. Mol. Biol. 1982, 157, 105-132
KD_SCALE = ProteinAnalysis("").kd_scale if hasattr(ProteinAnalysis(""), 'kd_scale') else {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# 信号肽 (-3,-1) 规则：切割位点处允许的小侧链/中性氨基酸
_SMALL_NEUTRAL = set("AGSTVC")
_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _normalize_sequence(sequence):
    """规范化并校验蛋白质序列，仅接受20种标准氨基酸字母。"""
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
    """为生物工具统一增加输入校验和异常兜底，防止工具异常导致主进程崩溃。"""
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
    """
    计算等电点 (pI) 和分子量 (Mw)
    使用 Biopython ProteinAnalysis 在本地确定性计算，不依赖外部网络。
    """
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
    """使用 Kyte-Doolittle 标度计算蛋白质平均疏水性（GRAVY）。"""
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
    """计算由氨基酸组成推导的螺旋、转角和折叠残基倾向比例。"""
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
    """统计20种标准氨基酸的计数和百分比。"""
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
    """计算蛋白质不稳定性指数，并按40这一经验阈值给出稳定性判断。"""
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
    """计算280 nm处还原态和氧化态蛋白质摩尔消光系数。"""
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
    """使用本地毒性蛋白数据库和 Smith-Waterman 比对检测高相似序列。"""
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
    """
    信号肽预测 — 基于 von Heijne 三区域模型与 (-3,-1) 切割位点规则

    信号肽由三个区域组成:
      n-区 (1-5 aa): N 端，携带正电荷 (Arg/Lys)
      h-区 (7-15 aa): 疏水核心，Kyte-Doolittle 平均疏水值需显著 > 0
      c-区 (3-7 aa): 切割位点区域，遵循 (-3,-1) 规则
    """
    if len(sequence) < 20:
        return {"status": "success", "has_signal_peptide": False,
                "confidence": "N/A", "note": "序列过短 (<20 aa)，无法分析"}

    # --- Step 1: N 端扫描范围 (信号肽一般 15-30 aa，最长不超过 70) ---
    scan = sequence[:min(70, len(sequence))]

    # --- Step 2: 用 Biopython protein_scale() 计算 h-region 疏水曲线，窗口 8 残基 ---
    h_window = 8
    # protein_scale 返回长度为 len(seq)-window+1 的列表，每个值对应以该位置为中心的窗口均值
    h_scores = ProteinAnalysis(scan).protein_scale(KD_SCALE, h_window)
    # 从位置 1 开始（跳过 n-region 的第 0 个氨基酸）找最高疏水性中心
    best_h_start = max(range(1, len(h_scores)), key=lambda i: h_scores[i], default=0)
    best_h_score = h_scores[best_h_start]
    best_h_end = best_h_start + h_window

    # --- Step 3: 搜索 (-3,-1) 切割位点 ---
    cleavage_site = None
    best_cs_score = 0
    for pos in range(best_h_end + 2, min(best_h_end + 15, len(sequence))):
        cs = 0
        if sequence[pos - 2] in _SMALL_NEUTRAL:   # -3 位
            cs += 2
        if sequence[pos] in _SMALL_NEUTRAL:        # -1 位
            cs += 2
        if sequence[pos - 1] != 'P':               # -2 位排除 Pro
            cs += 1
        if cs > best_cs_score:
            best_cs_score = cs
            cleavage_site = pos + 1  # 1-indexed

    # --- Step 4: n-region 正电荷检查 ---
    n_region = sequence[:best_h_start] if best_h_start > 0 else sequence[:3]
    n_charge = n_region.count('R') + n_region.count('K')

    # --- Step 5: 综合评分 (满分 8) ---
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
    """
    跨膜区预测 — Kyte-Doolittle 疏水性滑动窗口分析

    参考: Kyte & Doolittle, J. Mol. Biol. (1982) 157, 105-132
    窗口: 19 残基 | 阈值: 平均 KD > 1.6 | 最小跨膜长度: 15 残基
    """
    METHOD = "Kyte-Doolittle hydropathy (Biopython protein_scale), window=19, threshold=1.6"
    if len(sequence) < 20:
        return {"status": "success", "transmembrane_regions": [], "count": 0,
                "note": "序列过短", "method": METHOD}

    window, threshold = 19, 1.6
    half = window // 2

    # Step 1: 直接调用 Biopython protein_scale() 得到全序列疏水性曲线
    # 返回长度为 len(seq)-window+1 的列表，索引 i 对应原序列第 i+half 位
    scores = ProteinAnalysis(sequence).protein_scale(KD_SCALE, window)
    above = [i + half for i, s in enumerate(scores) if s > threshold]

    if not above:
        return {"status": "success", "transmembrane_regions": [], "count": 0,
                "topology": "可能为水溶性蛋白", "method": METHOD}

    # Step 2: 合并相邻超阈值位置为连续区域 (允许 ≤5 残基间隔)
    regions = []
    start = end = above[0]
    for pos in above[1:]:
        if pos - end <= 5:
            end = pos
        else:
            regions.append((start, end))
            start = end = pos
    regions.append((start, end))

    # Step 3: 扩展至窗口覆盖范围，过滤 < 15 残基的区域
    # 平均疏水性直接从 Biopython 曲线读取，无需重算
    valid = []
    for s, e in regions:
        a_start = max(0, s - half)
        a_end = min(len(sequence) - 1, e + half)
        length = a_end - a_start + 1
        if length >= 15:
            seg = sequence[a_start:a_end + 1]
            # 用 Biopython 算该片段的平均疏水值
            avg_kd = round(ProteinAnalysis(seg).protein_scale(KD_SCALE, min(len(seg), window))[len(seg)//2 - (min(len(seg),window)//2)], 2) \
                     if len(seg) >= window else round(sum(scores[max(0,a_start-half):a_end-half+1]) / max(1, a_end-a_start), 2)
            valid.append({
                "start": a_start + 1, "end": a_end + 1,
                "length": length, "sequence": seg,
                "avg_hydropathy": avg_kd, "type": "跨膜螺旋"
            })

    # Step 4: 拓扑推断
    n = len(valid)
    topo = ("可能为水溶性蛋白" if n == 0 else
            "单次跨膜蛋白 (I型或II型)" if n == 1 else
            f"多次跨膜蛋白 ({n}次跨膜)" if n <= 7 else
            f"高度跨膜蛋白 ({n}次跨膜)")

    return {"status": "success", "transmembrane_regions": valid, "count": n,
            "topology": topo, "method": METHOD}


@_defensive_tool
def predict_solubility(sequence: str):
    """
    蛋白溶解度预测 — 基于 GRAVY、不稳定性指数、等电点与带电残基比例的多参数评估

    评分体系 (满分 100):
      基础分 50 + GRAVY贡献(±20) + 稳定性贡献(±15) + pI贡献(±10)
      + 带电残基比例贡献(+10) + 长度贡献(±5)
    """
    if len(sequence) < 10:
        return {"status": "error", "message": "序列过短 (<10 aa)，无法分析"}

    try:
        pa = ProteinAnalysis(sequence)
        gravy = round(pa.gravy(), 3)
        instability = round(pa.instability_index(), 2)
        pI = round(pa.isoelectric_point(), 2)
        mw = round(pa.molecular_weight(), 2)

        # 带电残基比例 (D, E, K, R, H)
        aa_cnt = pa.count_amino_acids()
        charged = sum(aa_cnt.get(aa, 0) for aa in "DEKRH")
        charged_frac = round(charged / len(sequence), 3)

        # --- 综合评分 ---
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
    """ 一站式蛋白分析 """
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
