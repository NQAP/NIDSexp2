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

CONTINUOUS_FEATURES = [
    'fwd_packet_length_max',
    'fwd_packet_length_min',
    'fwd_packet_length_std',
    'bwd_packet_length_min',
    'bwd_packet_length_std',
    'flow_bytes/s', # 注意：/s 部分通常不會被替換，因為它不是空格
    'flow_packets/s', # 注意：/s 部分通常不會被替換
    'flow_iat_mean',
    'flow_iat_std',
    'flow_iat_min',
    'fwd_iat_total',
    'fwd_iat_mean',
    'fwd_iat_std',
    'fwd_iat_min',
    'bwd_iat_total',
    'bwd_iat_mean',
    'bwd_iat_std',
    'bwd_iat_max',
    'bwd_iat_min',
    'fwd_packets/s', # 注意：/s 部分通常不會被替換
    'bwd_packets/s', # 注意：/s 部分通常不會被替換
    'packet_length_min',
    'packet_length_std',
    'packet_length_variance',
    'down/up_ratio',
    'average_packet_size',
    'fwd_segment_size_avg',
    'bwd_segment_size_avg',
    'active_mean',
    'active_std',
    'active_max',
    'active_min',
    'idle_std',
    'idle_max',
    'idle_min'
]

DISCRETE_FEATURES = [
    'fwd_header_length',
    'bwd_header_length',
    'fin_flag_count',
    'syn_flag_count',
    'psh_flag_count',
    'ack_flag_count',
    'urg_flag_count',
    'ece_flag_count',
    'subflow_fwd_bytes',
    'subflow_bwd_bytes',
    'fwd_init_win_bytes',
    'bwd_init_win_bytes',
    'fwd_act_data_pkts',
    'fwd_seg_size_min',
    'label'
]
# (*** 新 ***)