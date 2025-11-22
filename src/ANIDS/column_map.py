# column_map.py
#
# 這是一個標準化欄位名稱的對應表 (Standardization Map)。
# (新) 修正了 Subflow 欄位對應

COLUMN_MAP = {
    # --- 關鍵識別欄位 (Key Identifiers) ---
    'flow id': 'flow_id',
    'source ip': 'src_ip',
    'source port': 'src_port',
    'destination ip': 'dst_ip',
    'destination port': 'dst_port',
    'dst port': 'dst_port', # 2018
    'protocol': 'protocol',
    'timestamp': 'timestamp',
    'label': 'label',
    
    # --- 時間相關 (Time-related) ---
    'flow duration': 'flow_duration',
    'flow iat mean': 'flow_iat_mean',
    'flow iat std': 'flow_iat_std',
    'flow iat max': 'flow_iat_max',
    'flow iat min': 'flow_iat_min',
    
    'fwd iat total': 'fwd_iat_tot',
    'fwd iat tot': 'fwd_iat_tot', # 2018
    'fwd iat mean': 'fwd_iat_mean',
    'fwd iat std': 'fwd_iat_std',
    'fwd iat max': 'fwd_iat_max',
    'fwd iat min': 'fwd_iat_min',
    
    'bwd iat total': 'bwd_iat_tot',
    'bwd iat tot': 'bwd_iat_tot', # 2018
    'bwd iat mean': 'bwd_iat_mean',
    'bwd iat std': 'bwd_iat_std',
    'bwd iat max': 'bwd_iat_max',
    'bwd iat min': 'bwd_iat_min',
    
    # --- 封包計數 (Packet Count) ---
    'total fwd packets': 'tot_fwd_pkts',
    'tot fwd pkts': 'tot_fwd_pkts', # 2018
    'total backward packets': 'tot_bwd_pkts',
    'tot bwd pkts': 'tot_bwd_pkts', # 2018
    
    # --- 封包總長度 (Total Length) ---
    'total length of fwd packets': 'totlen_fwd_pkts',
    'totlen fwd pkts': 'totlen_fwd_pkts', # 2018
    'total length of bwd packets': 'totlen_bwd_pkts',
    'totlen bwd pkts': 'totlen_bwd_pkts', # 2018
    
    # --- 封包長度統計 (Packet Length Stats) ---
    'fwd packet length max': 'fwd_pkt_len_max',
    'fwd pkt len max': 'fwd_pkt_len_max', # 2018
    'fwd packet length min': 'fwd_pkt_len_min',
    'fwd pkt len min': 'fwd_pkt_len_min', # 2018
    'fwd packet length mean': 'fwd_pkt_len_mean',
    'fwd pkt len mean': 'fwd_pkt_len_mean', # 2018
    'fwd packet length std': 'fwd_pkt_len_std',
    'fwd pkt len std': 'fwd_pkt_len_std', # 2018
    
    'bwd packet length max': 'bwd_pkt_len_max',
    'bwd pkt len max': 'bwd_pkt_len_max', # 2018
    'bwd packet length min': 'bwd_pkt_len_min',
    'bwd pkt len min': 'bwd_pkt_len_min', # 2018
    'bwd packet length mean': 'bwd_pkt_len_mean',
    'bwd pkt len mean': 'bwd_pkt_len_mean', # 2018
    'bwd packet length std': 'bwd_pkt_len_std',
    'bwd pkt len std': 'bwd_pkt_len_std', # 2018
    
    'min packet length': 'pkt_len_min',
    'pkt len min': 'pkt_len_min', # 2018
    'max packet length': 'pkt_len_max',
    'pkt len max': 'pkt_len_max', # 2S018
    'packet length mean': 'pkt_len_mean',
    'pkt len mean': 'pkt_len_mean', # 2018
    'packet length std': 'pkt_len_std',
    'pkt len std': 'pkt_len_std', # 2018
    'packet length variance': 'pkt_len_var',
    'pkt len var': 'pkt_len_var', # 2018
    
    # --- 速率 (Rate) ---
    'flow bytes/s': 'flow_byts_s',
    'flow byts/s': 'flow_byts_s', # 2018
    'flow packets/s': 'flow_pkts_s',
    'flow pkts/s': 'flow_pkts_s', # 2018
    'fwd packets/s': 'fwd_pkts_s',
    'fwd pkts/s': 'fwd_pkts_s', # 2018
    'bwd packets/s': 'bwd_pkts_s',
    'bwd pkts/s': 'bwd_pkts_s', # 2018
    
    # --- 標記計數 (Flag Count) ---
    'fwd psh flags': 'fwd_psh_flags',
    'bwd psh flags': 'bwd_psh_flags',
    'fwd urg flags': 'fwd_urg_flags',
    'bwd urg flags': 'bwd_urg_flags',
    'fin flag count': 'fin_flag_cnt',
    'fin flag cnt': 'fin_flag_cnt', # 2018
    'syn flag count': 'syn_flag_cnt',
    'syn flag cnt': 'syn_flag_cnt', # 2018
    'rst flag count': 'rst_flag_cnt',
    'rst flag cnt': 'rst_flag_cnt', # 2018
    'psh flag count': 'psh_flag_cnt',
    'psh flag cnt': 'psh_flag_cnt', # 2018
    'ack flag count': 'ack_flag_cnt',
    'ack flag cnt': 'ack_flag_cnt', # 2018
    'urg flag count': 'urg_flag_cnt',
    'urg flag cnt': 'urg_flag_cnt', # 2018
    'cwe flag count': 'cwe_flag_cnt',
    'ece flag count': 'ece_flag_cnt',
    'ece flag cnt': 'ece_flag_cnt', # 2018
    
    # --- 標頭與區段 (Header / Segment) ---
    'fwd header length': 'fwd_header_len',
    'fwd header len': 'fwd_header_len', # 2018
    'bwd header length': 'bwd_header_len',
    'bwd header len': 'bwd_header_len', # 2018
    
    'down/up ratio': 'down_up_ratio',
    'average packet size': 'pkt_size_avg',
    'pkt size avg': 'pkt_size_avg', # 2018
    
    'avg fwd segment size': 'fwd_seg_size_avg',
    'fwd seg size avg': 'fwd_seg_size_avg', # 2018
    'avg bwd segment size': 'bwd_seg_size_avg',
    'bwd seg size avg': 'bwd_seg_size_avg', # 2018
        
    'min_seg_size_forward': 'fwd_seg_size_min',
    'fwd seg size min': 'fwd_seg_size_min', # 2018
    
    # --- Bulk (大量) ---
    'fwd avg bytes/bulk': 'fwd_byts_b_avg',
    'fwd byts/b avg': 'fwd_byts_b_avg', # 2018
    'fwd avg packets/bulk': 'fwd_pkts_b_avg',
    'fwd pkts/b avg': 'fwd_pkts_b_avg', # 2018
    'fwd avg bulk rate': 'fwd_blk_rate_avg',
    'fwd blk rate avg': 'fwd_blk_rate_avg', # 2018
    'bwd avg bytes/bulk': 'bwd_byts_b_avg',
    'bwd byts/b avg': 'bwd_byts_b_avg', # 2018
    'bwd avg packets/bulk': 'bwd_pkts_b_avg',
    'bwd pkts/b avg': 'bwd_pkts_b_avg', # 2018
    'bwd avg bulk rate': 'bwd_blk_rate_avg',
    'bwd blk rate avg': 'bwd_blk_rate_avg', # 2018
    
    # --- 子流 (Subflow) ---
    # (新) 修正：同時包含 2017 和 2018 的拼寫
    'subflow fwd packets': 'subflow_fwd_pkts', # 2017
    'subflow fwd pkts': 'subflow_fwd_pkts',    # 2018
    'subflow fwd bytes': 'subflow_fwd_byts',   # 2017
    'subflow fwd byts': 'subflow_fwd_byts',    # 2018
    'subflow bwd packets': 'subflow_bwd_pkts', # 2017
    'subflow bwd pkts': 'subflow_bwd_pkts',    # 2018
    'subflow bwd bytes': 'subflow_bwd_byts',   # 2017
    'subflow bwd byts': 'subflow_bwd_byts',    # 2018
    
    # --- 初始視窗 (Init Window) ---
    'init_win_bytes_forward': 'init_fwd_win_byts',
    'init fwd win byts': 'init_fwd_win_byts', # 2018
    'init_win_bytes_backward': 'init_bwd_win_byts',
    'init bwd win byts': 'init_bwd_win_byts', # 2018
    
    # --- 活躍 (Active / Idle) ---
    'act_data_pkt_fwd': 'fwd_act_data_pkts',
    'fwd act data pkts': 'fwd_act_data_pkts', # 2S018
    
    'active mean': 'active_mean',
    'active std': 'active_std',
    'active max': 'active_max',
    'active min': 'active_min',
    'idle mean': 'idle_mean',
    'idle std': 'idle_std',
    'idle max': 'idle_max',
    'idle min': 'idle_min'
}