import pandas as pd
import numpy as np

def add_network_features(
    df,
    include_ports=True,
    include_ratios=True,
    include_flags=True,
    include_signatures=True,
    include_logs=True
):
    """
    Engineers domain-grounded network traffic features.
    
    Parameters:
    - df: pandas DataFrame with raw Zeek / FlowMeter network flow features.
    - include_ports: Whether to add port category and hierarchy flags.
    - include_ratios: Whether to add packet, byte, and payload ratios.
    - include_flags: Whether to add TCP flag interaction features.
    - include_signatures: Whether to add deterministic intrusion tool fingerprints.
    - include_logs: Whether to add log1p transforms for heavy-tailed rate distributions.
    
    Returns:
    - Transformed DataFrame with newly engineered features.
    """
    data = df.copy()
    
    # 1. Port Categorizations
    if include_ports and 'id.resp_p' in data.columns and 'id.orig_p' in data.columns:
        data['resp_is_well_known'] = (data['id.resp_p'] <= 1023).astype(np.float32)
        data['resp_is_registered'] = ((data['id.resp_p'] > 1023) & (data['id.resp_p'] <= 49151)).astype(np.float32)
        data['resp_is_dynamic'] = (data['id.resp_p'] > 49151).astype(np.float32)
        
        data['orig_is_well_known'] = (data['id.orig_p'] <= 1023).astype(np.float32)
        data['orig_is_dynamic'] = (data['id.orig_p'] > 49151).astype(np.float32)
        data['is_same_port'] = (data['id.orig_p'] == data['id.resp_p']).astype(np.float32)
        
        # High-traffic specific port flags
        data['is_port_21'] = (data['id.resp_p'] == 21.0).astype(np.float32)
        data['is_port_53'] = (data['id.resp_p'] == 53.0).astype(np.float32)
        data['is_port_80'] = (data['id.resp_p'] == 80.0).astype(np.float32)
        data['is_port_443'] = (data['id.resp_p'] == 443.0).astype(np.float32)
        data['is_port_1883'] = (data['id.resp_p'] == 1883.0).astype(np.float32)
        data['is_port_5353'] = (data['id.resp_p'] == 5353.0).astype(np.float32)

    # 2. Flow Packet, Byte, and Payload Ratios
    if include_ratios:
        data['total_pkts'] = (data['fwd_pkts_tot'] + data['bwd_pkts_tot']).astype(np.float32)
        data['pkt_fwd_ratio'] = (data['fwd_pkts_tot'] / (data['total_pkts'] + 1e-5)).astype(np.float32)
        data['fwd_to_bwd_pkt_ratio'] = (data['fwd_pkts_tot'] / (data['bwd_pkts_tot'] + 1.0)).astype(np.float32)
        
        data['total_header_bytes'] = (data['fwd_header_size_tot'] + data['bwd_header_size_tot']).astype(np.float32)
        data['fwd_to_bwd_header_ratio'] = (data['fwd_header_size_tot'] / (data['bwd_header_size_tot'] + 1.0)).astype(np.float32)
        
        data['fwd_payload_ratio'] = (data['fwd_pkts_payload.tot'] / (data['flow_pkts_payload.tot'] + 1.0)).astype(np.float32)
        data['avg_bytes_per_pkt'] = (data['total_header_bytes'] / (data['total_pkts'] + 1.0)).astype(np.float32)
        data['avg_fwd_pkt_size'] = (data['fwd_pkts_payload.tot'] / (data['fwd_pkts_tot'] + 1.0)).astype(np.float32)

    # 3. TCP Flag Relationships
    if include_flags and 'flow_SYN_flag_count' in data.columns and 'flow_ACK_flag_count' in data.columns:
        data['syn_to_ack_ratio'] = (data['flow_SYN_flag_count'] / (data['flow_ACK_flag_count'] + 1.0)).astype(np.float32)
        data['rst_to_ack_ratio'] = (data['flow_RST_flag_count'] / (data['flow_ACK_flag_count'] + 1.0)).astype(np.float32)
        data['fin_to_ack_ratio'] = (data['flow_FIN_flag_count'] / (data['flow_ACK_flag_count'] + 1.0)).astype(np.float32)
        
        data['is_syn_only'] = ((data['flow_SYN_flag_count'] > 0) & (data['flow_ACK_flag_count'] == 0)).astype(np.float32)
        data['is_syn_ack'] = ((data['flow_SYN_flag_count'] > 0) & (data['flow_ACK_flag_count'] > 0)).astype(np.float32)
        data['has_rst'] = (data['flow_RST_flag_count'] > 0).astype(np.float32)

    # 4. Tool & Attack Signatures
    if include_signatures:
        # hping3 SYN flood signature: window size 64 + 1 forward packet
        data['is_hping_sig'] = (
            (data['fwd_init_window_size'] == 64.0) & (data['fwd_pkts_tot'] == 1.0)
        ).astype(np.float32)
        
        # NMAP TCP SYN scan signature: 1 forward packet, 0 backward packets, SYN flag set, window != 64
        data['is_nmap_syn_sig'] = (
            (data['fwd_pkts_tot'] == 1.0) &
            (data['bwd_pkts_tot'] == 0.0) &
            (data['flow_SYN_flag_count'] > 0) &
            (data['fwd_init_window_size'] != 64.0)
        ).astype(np.float32)
        
        # UDP single packet probe signature
        data['is_udp_probe'] = (
            (data['fwd_pkts_tot'] == 1.0) & (data['bwd_pkts_tot'] == 0.0) & (data['fwd_init_window_size'] == 0.0)
        ).astype(np.float32)

    # 5. Skew-Corrected Log Transformations
    if include_logs:
        skewed_cols = [
            'flow_duration', 'payload_bytes_per_second', 'flow_pkts_per_sec',
            'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_iat.tot', 'fwd_iat.tot'
        ]
        for col in skewed_cols:
            if col in data.columns:
                data[f'log_{col}'] = np.log1p(np.clip(data[col].values, 0, None)).astype(np.float32)

    return data

