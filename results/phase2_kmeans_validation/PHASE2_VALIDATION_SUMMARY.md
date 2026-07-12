# Phase 2 Feature List Validation Summary

## Status
RECOMMENDED PHASE 2 FEATURE CANDIDATE

## Selected feature list
optimized15_xss19

- src_ip_5_mean
- stream_60_count
- stream_jitter_60_sum
- is_dns_packet
- channel_10_count
- http_uri_special_char_count
- tcp_window_size
- inter_arrival_time
- max_p
- user_agent_length
- is_http_packet
- src_ip_mac_60_var
- l3_ip_dst_count
- http_uri_parameter_count
- highest_layer_is_tls
- http_uri_xss_token_count
- http_uri_has_xss_pattern
- http_uri_angle_bracket_count
- http_uri_percent_encoded_count

## Selected pipeline
kmeans {'k': 6}
Three-seed score ensemble: [42, 143, 244]

## Final holdout — high-F1 operating point
- Accuracy: 0.9720
- Precision: 0.4904
- Recall: 0.3891
- F1: 0.4339
- FPR: 0.0115
- ROC-AUC: 0.8854
- PR-AUC: 0.3492

## Final holdout — coverage operating point
- Accuracy: 0.9612
- Precision: 0.3571
- Recall: 0.5044
- F1: 0.4182
- FPR: 0.0258
- Macro attack recall: 0.4426
- Minimum attack recall: 0.0914
- ROC-AUC: 0.8854
- PR-AUC: 0.3492

## Weak attack types
['xss']

## Interpretation
The feature candidate was selected using development data across multiple model families and random seeds. The final holdout was evaluated only after the pipeline and thresholds were frozen.