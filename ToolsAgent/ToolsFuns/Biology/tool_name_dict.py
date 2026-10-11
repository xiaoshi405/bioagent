# -*- coding: utf-8 -*-
"""
生物工具注册表，作为 Planner、KG 和执行层的工具元数据来源。

修改记录：
- 2026-09-05：为可替换的 Artifact 适配工具增加重规划策略元数据，
  使 F1 能从注册表派生必须保留的目标工具，避免按题号硬编码。
- 2026-08-15：为记录选择工具显式声明重复无效步骤剪枝许可；修复器不得据此
  删除单次选择、有效选择或输出被下游消费的选择步骤。
- 2026-07-19：补充五个 Biopython 序列描述工具和本地毒性相似性筛查；
  修正 pI/Mw 的离线计算描述，并为 KG 增加明确输出类型。
- 2026-07-20：注册三个PDB文件工具，并增加PDB文件输入类型以支持结构到序列链路。
- 2026-07-23：为全部序列工具显式注册输入类型，避免类型系统依赖隐式默认值。
- 2026-07-24：注册单记录Protein FASTA解析工具和文件输入类型。
- 2026-07-24：修正PDB多链提取输出类型，并注册显式链选择工具。
- 2026-07-26：为全部外部工具增加output_bindings，作为Artifact Pool从
  工具JSON结果中提取类型化对象的唯一字段绑定来源。
- 2026-08-14：按领域模块注册多记录FASTA解析、显式记录选择、两记录
  比对与比对统计工具；确定性桥接元数据与I/O签名同源维护。
"""


_SEQUENCE_PARAMETER = {"sequence": "str（标准单字母氨基酸序列）"}
_PDB_PARAMETER = {"file_path": "str（已上传到项目目录的.pdb文件路径）"}
_PDB_CHAIN_PARAMETER = {
    "file_path": "str（已上传到项目目录的.pdb文件路径）",
    "chain_id": "str（用户明确指定的PDB链ID，区分大小写）",
}
_FASTA_PARAMETER = {
    "file_path": "str（已上传到项目目录的单记录.fasta/.fa/.faa文件路径）"
}
_MULTI_FASTA_PARAMETER = {
    "file_path": "str（已上传到项目目录的多记录.fasta/.fa/.faa文件路径）"
}
_FASTA_RECORD_PARAMETER = {
    "collection": "dict（parse_multi_protein_fasta输出的具名蛋白质序列集合）",
    "record_id": "str（用户显式提供、区分大小写的精确FASTA记录ID）",
}
_FASTA_RECORD_PAIR_PARAMETER = {
    "collection": "dict（parse_multi_protein_fasta输出的具名蛋白质序列集合）",
    "record_ids": "list[str]（用户显式提供的两个不同FASTA记录ID）",
}
_ALIGNMENT_PARAMETER = {
    "alignment": "dict（align_fasta_records输出的成对蛋白质比对对象）"
}
_SEQUENCE_INPUT_TYPES = ["amino acid sequence"]
_BIOLOGY_FILE_PATH = "ToolsAgent/ToolsFuns/Biology/biologyCommon.py"
_PDB_FILE_PATH = "ToolsAgent/ToolsFuns/Biology/pdb_analysis.py"
_FASTA_FILE_PATH = "ToolsAgent/ToolsFuns/Biology/fasta_analysis.py"
_FASTA_COLLECTION_FILE_PATH = "ToolsAgent/ToolsFuns/Biology/fasta_collection.py"
_PROTEIN_ALIGNMENT_FILE_PATH = "ToolsAgent/ToolsFuns/Biology/protein_alignment.py"


BIOLOGY_TOOLS_DESC = {
    "compute_pI_mw": {
        "description": "使用Biopython在本地计算氨基酸序列的等电点(pI)和分子量(Mw)",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_pi_mw",
        "return_type": "dict{status, sequence_length, pI, molecular_weight}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["pI", "molecular weight"],
        "output_bindings": {"pI": "pI", "molecular weight": "molecular_weight"},
    },
    "compute_gravy": {
        "description": "使用Kyte-Doolittle标度在本地计算蛋白质平均疏水性GRAVY值",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_gravy",
        "return_type": "dict{status, sequence_length, gravy, method}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["GRAVY hydropathy score"],
        "output_bindings": {"GRAVY hydropathy score": "gravy"},
    },
    "compute_secondary_structure_fraction": {
        "description": "根据氨基酸组成计算螺旋、转角和折叠残基倾向比例，不等同于三维结构指派",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_secondary_structure_fraction",
        "return_type": "dict{status, sequence_length, helix_fraction, turn_fraction, sheet_fraction}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["secondary structure propensity fractions"],
        "output_bindings": {"secondary structure propensity fractions": "$"},
    },
    "compute_amino_acid_composition": {
        "description": "统计蛋白质序列中20种标准氨基酸的计数和百分比组成",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_amino_acid_composition",
        "return_type": "dict{status, sequence_length, counts, percentages, units}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["amino acid composition"],
        "output_bindings": {"amino acid composition": "$"},
    },
    "compute_instability_index": {
        "description": "使用Biopython在本地计算蛋白质不稳定性指数，并按40阈值给出经验稳定性判断",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_instability_index",
        "return_type": "dict{status, sequence_length, instability_index, is_stable, threshold}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["protein instability index"],
        "output_bindings": {"protein instability index": "instability_index"},
    },
    "compute_extinction_coefficient": {
        "description": "计算蛋白质在280 nm处还原态和氧化态的摩尔消光系数",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/compute_extinction_coefficient",
        "return_type": "dict{status, sequence_length, reduced_molar_extinction_coefficient, oxidized_molar_extinction_coefficient, units}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["molar extinction coefficient"],
        "output_bindings": {"molar extinction coefficient": "$"},
    },
    "predict_signalpeptide": {
        "description": "基于von Heijne区域模型和疏水性规则筛查蛋白质信号肽及候选切割位点",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/predict_signalpeptide",
        "return_type": "dict{status, has_signal_peptide, confidence, cleavage_site, method}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["signal peptide prediction"],
        "output_bindings": {"signal peptide prediction": "$"},
    },
    "predict_transmembrane": {
        "description": "使用Kyte-Doolittle滑动窗口筛查蛋白质序列中的候选跨膜螺旋区域",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/predict_transmembrane",
        "return_type": "dict{status, transmembrane_regions, count, topology, method}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["transmembrane regions"],
        "output_bindings": {"transmembrane regions": "transmembrane_regions"},
    },
    "predict_solubility": {
        "description": "根据GRAVY、不稳定性指数、pI和带电残基比例进行轻量级蛋白溶解度评分",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/predict_solubility",
        "return_type": "dict{status, soluble, solubility_score, parameters, method}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["solubility score"],
        "output_bindings": {"solubility score": "solubility_score"},
    },
    "check_protein_toxicity": {
        "description": "使用本地毒性蛋白数据库和Smith-Waterman局部比对筛查高度相似序列，不替代实验毒性鉴定",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": "app/tools/safety/toxicity_checker.py",
        "api_path": "/api/check_protein_toxicity",
        "return_type": "dict{status, high_similarity_to_known_toxin, similarity_percent, identity_percent, coverage_percent, matched_protein}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": ["toxin sequence similarity screening"],
        "output_bindings": {"toxin sequence similarity screening": "$"},
    },
    "protein_analysis": {
        "description": "基础一站式蛋白分析接口，返回pI/分子量、信号肽、跨膜区和溶解度结果",
        "parameters": _SEQUENCE_PARAMETER,
        "file_path": _BIOLOGY_FILE_PATH,
        "api_path": "/api/protein_analysis",
        "return_type": "dict{status, basic_info, signal_peptide, transmembrane, solubility}",
        "input_types": _SEQUENCE_INPUT_TYPES,
        "output_types": [
            "pI",
            "molecular weight",
            "signal peptide prediction",
            "transmembrane regions",
            "solubility score",
        ],
        "output_bindings": {
            "pI": "basic_info.pI",
            "molecular weight": "basic_info.molecular_weight",
            "signal peptide prediction": "signal_peptide",
            "transmembrane regions": "transmembrane.transmembrane_regions",
            "solubility score": "solubility.solubility_score",
        },
    },
    "parse_protein_fasta": {
        "description": "使用Biopython在本地解析单记录Protein FASTA，严格校验20种标准氨基酸并拒绝多记录自动选择",
        "parameters": _FASTA_PARAMETER,
        "file_path": _FASTA_FILE_PATH,
        "api_path": "/api/parse_protein_fasta",
        "return_type": "dict{status, file_name, record_id, description, sequence, sequence_length, method}",
        "input_types": ["single-record protein FASTA file"],
        "output_types": ["amino acid sequence"],
        "output_bindings": {"amino acid sequence": "sequence"},
        "repair_policy": {"mode": "deterministic_bridge", "cost": 1},
        "replan_policy": {"role": "artifact_adapter", "allow_drop": True},
    },
    "parse_multi_protein_fasta": {
        "description": "使用Biopython在本地严格解析多记录Protein FASTA，返回保留精确record_id的具名蛋白质序列集合",
        "parameters": _MULTI_FASTA_PARAMETER,
        "file_path": _FASTA_COLLECTION_FILE_PATH,
        "api_path": "/api/parse_multi_protein_fasta",
        "return_type": "dict{status, file_name, record_count, record_ids, collection, method}",
        "input_types": ["multi-record protein FASTA file"],
        "output_types": ["protein sequence collection"],
        "output_bindings": {"protein sequence collection": "collection"},
        "repair_policy": {"mode": "deterministic_bridge", "cost": 1},
        "replan_policy": {"role": "artifact_adapter", "allow_drop": True},
        "benchmark_generation": "topology-v2-lite",
    },
    "select_fasta_record": {
        "description": "从具名蛋白质序列集合中按用户明确提供的record_id精确选择一条序列；禁止模糊匹配和自动选择",
        "parameters": _FASTA_RECORD_PARAMETER,
        "file_path": _FASTA_COLLECTION_FILE_PATH,
        "api_path": "/api/select_fasta_record",
        "return_type": "dict{status, record_id, description, sequence, sequence_length, available_record_ids, selection_method}",
        "input_types": ["protein sequence collection", "FASTA record identifier"],
        "output_types": ["amino acid sequence"],
        "output_bindings": {"amino acid sequence": "sequence"},
        "repair_policy": {
            "mode": "deterministic_bridge",
            "cost": 1,
            "allow_duplicate_invalid_prune": True,
        },
        "replan_policy": {"role": "artifact_adapter", "allow_drop": True},
        "benchmark_generation": "topology-v2-lite",
    },
    "align_fasta_records": {
        "description": "对具名蛋白质序列集合中用户明确指定的两个record_id执行配置化BLOSUM全局比对",
        "parameters": _FASTA_RECORD_PAIR_PARAMETER,
        "file_path": _PROTEIN_ALIGNMENT_FILE_PATH,
        "api_path": "/api/align_fasta_records",
        "return_type": "dict{status, record_ids, alignment_length, score, alignment}",
        "input_types": ["protein sequence collection", "FASTA record identifier pair"],
        "output_types": ["pairwise protein alignment"],
        "output_bindings": {"pairwise protein alignment": "alignment"},
        "repair_policy": {"mode": "deterministic_bridge", "cost": 1},
        "benchmark_generation": "topology-v2-lite",
    },
    "summarize_pairwise_alignment": {
        "description": "从成对蛋白质比对对象计算identity、BLOSUM正相似性、gap比例与覆盖率调整identity",
        "parameters": _ALIGNMENT_PARAMETER,
        "file_path": _PROTEIN_ALIGNMENT_FILE_PATH,
        "api_path": "/api/summarize_pairwise_alignment",
        "return_type": "dict{status, record_ids, statistics}",
        "input_types": ["pairwise protein alignment"],
        "output_types": ["alignment statistics"],
        "output_bindings": {"alignment statistics": "statistics"},
        "benchmark_generation": "topology-v2-lite",
    },
    "analyze_pdb_structure": {
        "description": "使用Bio.PDB离线分析上传的PDB文件，统计链、残基、原子、配体、水、几何特征及HELIX/SHEET注释",
        "parameters": _PDB_PARAMETER,
        "file_path": _PDB_FILE_PATH,
        "api_path": "/api/analyze_pdb_structure",
        "return_type": "dict{status, chain_count, chains, protein_residue_count, protein_atom_count, radius_of_gyration_angstrom, secondary_structure_annotation}",
        "input_types": ["PDB structure file"],
        "output_types": [
            "PDB structure summary",
            "secondary structure annotation",
            "structural geometry",
        ],
        "output_bindings": {
            "PDB structure summary": "$",
            "secondary structure annotation": "secondary_structure_annotation",
            "structural geometry": "$",
        },
    },
    "extract_pdb_sequences": {
        "description": "使用Bio.PDB从上传的PDB第一模型中按链提取蛋白质氨基酸序列",
        "parameters": _PDB_PARAMETER,
        "file_path": _PDB_FILE_PATH,
        "api_path": "/api/extract_pdb_sequences",
        "return_type": "dict{status, chain_count, total_residue_count, chains{sequence, sequence_length}}",
        "input_types": ["PDB structure file"],
        "output_types": ["PDB chain sequence map"],
        "output_bindings": {"PDB chain sequence map": "chains"},
    },
    "select_pdb_chain": {
        "description": "从上传PDB中按用户明确提供的chain_id选择一条蛋白质链序列；未指定链时拒绝自动猜测",
        "parameters": _PDB_CHAIN_PARAMETER,
        "file_path": _PDB_FILE_PATH,
        "api_path": "/api/select_pdb_chain",
        "return_type": "dict{status, file_name, chain_id, sequence, sequence_length, available_chain_ids}",
        "input_types": ["PDB structure file", "PDB chain identifier"],
        "output_types": ["amino acid sequence"],
        "output_bindings": {"amino acid sequence": "sequence"},
        "repair_policy": {"mode": "deterministic_bridge", "cost": 1},
        "replan_policy": {"role": "artifact_adapter", "allow_drop": True},
    },
    "calculate_protein_anm": {
        "description": "使用NumPy对上传PDB中指定链或残基最多的链执行Cα各向异性网络模型，返回最低频非零模态特征值",
        "parameters": {
            "file_path": "str（已上传到项目目录的.pdb文件路径）",
            "n_modes": "int（1-20，默认10）",
            "cutoff": "float（4-20埃，默认12）",
            "chain_id": "str或null（可选）",
        },
        "parameter_schema": {
            "n_modes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
            "cutoff": {
                "type": "number",
                "minimum": 4.0,
                "maximum": 20.0,
                "default": 12.0,
            },
            "chain_id": {
                "type": "string",
                "nullable": True,
                "maximum_length": 8,
                "default": None,
            },
        },
        "file_path": _PDB_FILE_PATH,
        "api_path": "/api/calculate_protein_anm",
        "return_type": "dict{status, chain_id, residue_count, cutoff_angstrom, contact_count, eigenvalues}",
        "input_types": ["PDB structure file"],
        "output_types": ["ANM eigenvalues", "protein dynamic modes"],
        "output_bindings": {
            "ANM eigenvalues": "eigenvalues",
            "protein dynamic modes": "$",
        },
    },
    "send_expasy_request": {
        "description": "历史兼容的内部网络请求工具，不向Agent开放",
        "parameters": {"sequence": "str"},
        "file_path": "ToolsAgent/ToolsFuns/Biology/support/send_request.py",
        "is_internal": True,
    },
}
