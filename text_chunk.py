import re
from typing import List, Callable, Optional

def recursive_text_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    separators: Optional[List[str]] = None,
    length_function: Callable[[str], int] = len,
    keep_separator: bool = True
) -> List[str]:
    """
    递归地将文本分割成指定大小的块，尽量保持语义完整性
    
    参数:
        text: 要分割的文本
        chunk_size: 目标块大小
        chunk_overlap: 块之间的重叠大小
        separators: 用于分割的分隔符列表，按优先级排序
        length_function: 用于计算文本长度的函数
        keep_separator: 是否在分割后保留分隔符
    
    返回:
        分割后的文本块列表
    """
    # 默认分隔符：段落、句子、单词、字符
    if separators is None:
        separators = ["\r\n\r\n", "\n\n", "\r\n", "\n", ". ", "? ", "! ", " "]
    
    # 如果文本已经足够小，直接返回
    if length_function(text) <= chunk_size:
        return [text]
    
    # 尝试使用每个分隔符进行分割
    for i, separator in enumerate(separators):
        if separator == "":
            # 最后的手段：按字符分割
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        
        # 使用当前分隔符分割文本
        splits = _split_text_with_separator(text, separator, keep_separator)
        
        # 检查是否成功分割
        if len(splits) > 1:
            # 递归处理每个分割部分
            chunks = []
            for split in splits:
                if length_function(split) < chunk_size:
                    chunks.append(split)
                else:
                    # 递归调用，使用下一个分隔符
                    chunks.extend(
                        recursive_text_split(
                            split,
                            chunk_size,
                            chunk_overlap,
                            separators[i+1:] if i+1 < len(separators) else [""],
                            length_function,
                            keep_separator
                        )
                    )
            
            # 合并小块并处理重叠
            return _merge_splits_with_overlap(chunks, chunk_size, chunk_overlap, length_function)
    
    # 如果所有分隔符都失败，按字符分割
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

def _split_text_with_separator(text: str, separator: str, keep_separator: bool) -> List[str]:
    """使用指定分隔符分割文本"""
    if separator == "":
        return list(text)
    
    if keep_separator:
        # 保留分隔符在分割后的文本中
        splits = re.split(f"({re.escape(separator)})", text)
        # 将分隔符与前面的文本合并
        result = []
        for i in range(0, len(splits)-1, 2):
            result.append(splits[i] + splits[i+1])
        # 处理最后一个可能没有分隔符的部分
        if len(splits) % 2 == 1:
            result.append(splits[-1])
        return result
    else:
        # 不保留分隔符
        return re.split(re.escape(separator), text)

def _merge_splits_with_overlap(
    splits: List[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int]
) -> List[str]:
    """合并分割后的文本块，添加重叠部分"""
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for split in splits:
        split_length = length_function(split)
        
        # 如果当前块为空，直接添加
        if current_length == 0:
            current_chunk = split
            current_length = split_length
        # 如果添加新分割不会超过块大小
        elif current_length + split_length <= chunk_size:
            current_chunk += split
            current_length += split_length
        else:
            # 保存当前块
            chunks.append(current_chunk)
            
            # 创建有重叠的新块
            if chunk_overlap > 0:
                # 从当前块末尾提取重叠部分
                overlap_start = max(0, current_length - chunk_overlap)
                overlap_text = current_chunk[overlap_start:]
                
                # 开始新块，包含重叠文本
                current_chunk = overlap_text + split
                current_length = length_function(overlap_text) + split_length
            else:
                # 没有重叠
                current_chunk = split
                current_length = split_length
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# 示例用法
if __name__ == "__main__":
    # 示例文本
    sample_text = """'家庭常见植物养护指南知识库\r\n版本： 1.0\r\n创建日期： 2023-10-27\r\n说明： 本知识库由个人整理，旨在提供简洁实用的家庭植物养护建议。请注意，植物生长受环境差异影响，需灵活调整。\r\n\r\n【绿萝】\r\n光照： 喜明亮的散射光，耐阴性强，避免阳光直射。\r\n浇水： 喜湿润，见干见湿。表土干燥后浇透水，冬季减少浇水频率。定期喷雾增湿有益。\r\n注意事项： 盆内不可长期积水，否则易烂根。是净化空气的优良品种。\r\n\r\n【虎皮兰】\r\n光照： 喜光也耐阴，适应性强，但长期阴暗会导致叶片变暗。\r\n浇水： 非常耐旱。务必等盆土完全干透后再浇水，冬季可一个月甚至更长时间浇水一次。宁干勿湿。\r\n注意事项： 浇水过勤是其死亡的主要原因。对猫狗有毒，需注意。\r\n\r\n【多肉植物（通用）】\r\n光照： 非常喜光，需要每天至少4-6小时的充足日照，否则易徒长。\r\n浇水： 极度耐旱。遵循“干透浇透”原则，即盆土完全干燥后一次性浇透水，且盆底不能积水。夏季高温和冬季低温时严格控水甚至断水。\r\n注意事项： 最关键的是通风和透水，务必使用专用的颗粒土。\r\n\r\n【君子兰】\r\n光照： 喜明亮的散射光，忌强光直射。\r\n浇水： 保持盆土微润但不过湿。浇水时避免浇到叶芯，防止烂心。\r\n注意事项： 每年可换盆一次，喜肥沃疏松的土壤。开花前后可施磷钾肥。\r\n\r\n【发财树】\r\n光照： 喜充足散射光，也能适应较低光照环境。\r\n浇水： 耐旱，怕涝。浇水前需检查土壤干湿情况，干透后再浇，冬季保持土壤干燥。\r\n注意事项： 树干粗壮能储存水分，主要死因是浇水过多导致根部腐烂。\r\n\r\n【通用养护小贴士】\r\n施肥： “薄肥勤施”。在植物生长旺盛期（春、秋）定期施肥，休眠期（夏、冬）停止施肥。\r\n通风： 良好的通风环境可以有效预防病虫害。\r\n清洁叶片： 定期用湿布擦拭叶片上的灰尘，有助于植物进行光合作用。'"""
    
    # 使用递归分割
    chunks = recursive_text_split(
        text=sample_text,
        chunk_size=150,
        chunk_overlap=30,
        separators=["\r\n\r\n", "\n\n", "\r\n", "\n", ". ", "? ", "! ", " "]
    )
    
    # 打印结果
    for i, chunk in enumerate(chunks):
        print(f"块 {i+1} (长度: {len(chunk)}):")
        print(repr(chunk))
        print("-" * 50)