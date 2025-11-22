# (*** 新 ***) 
# 建立一個「特徵白名單」，只包含論文 TABLE V 中列出的 58 個特徵
# (我們使用 `column_map.py` 中對應的「乾淨」名稱)
FEATURES_WHITELIST = [
    # Time-Realted (15)
    'flow_duration', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
    'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
    
    # Packet-Realted (13)
    'tot_fwd_pkts', 'tot_bwd_pkts', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 
    'bwd_pkts_s', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 
    'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts',
    
    # Length-Realted (21)
    'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min',
    'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min',
    'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'fwd_header_len', 'bwd_header_len',
    'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
    'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_seg_size_min',
    
    # Protocol-Related (9)
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt',
    'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio', 'fwd_psh_flags',
    
    # 標籤 (Label)
    'label'
]

# ----------------------------------------------------------------------------
# 離散型特徵 (DISCRETE FEATURES) - 共 29 個
# ----------------------------------------------------------------------------
DISCRETE_FEATURES = [
    # Packet-Realted (9) - 封包/位元組的「計數」
    'tot_fwd_pkts', 
    'tot_bwd_pkts',
    'subflow_fwd_pkts', 
    'subflow_fwd_byts', 
    'subflow_bwd_pkts', 
    'subflow_bwd_byts', 
    'init_fwd_win_byts', 
    'init_bwd_win_byts', 
    'fwd_act_data_pkts',
    
    # Length-Realted (11) - 封包/標頭的「長度」
    'totlen_fwd_pkts', 
    'totlen_bwd_pkts', 
    'fwd_pkt_len_max', 
    'fwd_pkt_len_min',
    'bwd_pkt_len_max', 
    'bwd_pkt_len_min',
    'fwd_header_len', 
    'bwd_header_len',
    'pkt_len_min', 
    'pkt_len_max', 
    'fwd_seg_size_min',
    
    # Protocol-Related (8) - 旗標的「計數」
    'fin_flag_cnt', 
    'syn_flag_cnt', 
    'rst_flag_cnt', 
    'psh_flag_cnt', 
    'ack_flag_cnt',
    'urg_flag_cnt', 
    'ece_flag_cnt', 
    'fwd_psh_flags',
    
]

# ----------------------------------------------------------------------------
# 連續型特徵 (CONTINUOUS FEATURES) - 共 30 個
# ----------------------------------------------------------------------------
CONTINUOUS_FEATURES = [
    # Time-Realted (15) - 所有的「時間測量值」
    'flow_duration', 
    'flow_iat_mean', 
    'flow_iat_std', 
    'flow_iat_max', 
    'flow_iat_min',
    'fwd_iat_tot', 
    'fwd_iat_mean', 
    'fwd_iat_std', 
    'fwd_iat_max', 
    'fwd_iat_min',
    'bwd_iat_tot', 
    'bwd_iat_mean', 
    'bwd_iat_std', 
    'bwd_iat_max', 
    'bwd_iat_min',
    
    # Packet-Realted (4) - 所有的「速率」
    'flow_byts_s', 
    'flow_pkts_s', 
    'fwd_pkts_s', 
    'bwd_pkts_s', 
    
    # Length-Realted (10) - 所有的「統計值」(平均, 標準差, 變異數)
    'fwd_pkt_len_mean', 
    'fwd_pkt_len_std', 
    'bwd_pkt_len_mean', 
    'bwd_pkt_len_std', 
    'pkt_len_mean', 
    'pkt_len_std', 
    'pkt_len_var',
    'pkt_size_avg', 
    'fwd_seg_size_avg', 
    'bwd_seg_size_avg', 
    
    # Protocol-Related (1) - 「比率」
    'down_up_ratio'
]

# (*** 新 ***)