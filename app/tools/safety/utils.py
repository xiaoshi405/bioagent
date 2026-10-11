#目前没有找到这个文件定义这个函数的应用点

import re
from typing import List, Tuple

def extract_smiles_or_sequences(data_str: str) -> List[Tuple[str, str]]:
    """
    Extract protein sequence from a string. (SMILES logic removed)
    """
    if not isinstance(data_str, str):
        return []

    extracted_items = []
    # 字典里只保留生物学 sequence 的正则匹配规则
    patterns = {
        'sequence': [r"protein sequence: ([^}]*)", r"Protein sequence:\s*([ARNDCQEGHILKMFPSTWYV]+)"]
    }

    for item_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, data_str)
            extracted_items.extend((match, item_type) for match in matches)

    return extracted_items

