def create_masks(src, trg_input):
    src_mask = (src != 0).unsqueeze(-2)  # 假设0是填充token的索引
    trg_mask = (trg_input != 0).unsqueeze(-2)
    return src_mask, trg_mask
