# Headless Replay Report: replay_test_headless_trace_1_500_l

- status: `ok`
- trace: `traces/trace_test_headless_trace_1_500.jsonl`
- perf: `perf/headless_replay_perf_replay_test_headless_trace_1_500_l_11.jsonl`

## Overall

- simulation_steps: 500
- agent_count: 3
- agent_moves: 1500
- wall_time_ms: 2669511.185
- llm_total_ms: 1547381.095
- embedding_total_ms: 887692.405
- model_total_ms: 2435073.500
- agent_move_total_ms: 2549129.490
- agent_move_non_model_total_ms: 114055.990
- overhead_excluding_llm_embedding_ms: 234437.685
- steps_per_second: 0.187
- agent_moves_per_second: 0.562

## Replay Quality

- retrieval_canonicalized: 1500
- reflection_skipped_by_trace: 1494
- memory_exact: 256
- memory_filling_mismatch: 14
- memory_core_mismatch: 22
- errors: 0

## Parallelism Estimate

- sequential_agent_move_ms: 2549129.490
- agent_parallel_agent_move_ms: 2325295.084
- sequential_model_ms: 2435073.500
- agent_parallel_model_ms: 2215220.685
- sequential_llm_ms: 1547381.095
- agent_parallel_llm_ms: 1424238.006
- sequential_non_model_ms: 114055.990
- agent_parallel_non_model_ms: 110183.953
- estimated_agent_parallel_agent_move_speedup: 1.096
- estimated_agent_parallel_model_speedup: 1.099
- estimated_agent_parallel_llm_speedup: 1.086
- estimated_agent_parallel_non_model_speedup: 1.035
- avg_model_ms_per_step: 4870.147
- avg_agent_move_ms_per_step: 5098.259
- avg_non_model_ms_per_step: 228.112

## LLM By Prompt Template

- `persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt`: count=149, total_ms=330173.843, avg_ms=2215.932, p95_ms=6545.570
- `persona/prompt_template/v2/generate_event_triple_v1.txt`: count=115, total_ms=238652.128, avg_ms=2075.236, p95_ms=6984.125
- `persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt`: count=32, total_ms=226368.633, avg_ms=7074.020, p95_ms=9461.524
- `persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt`: count=32, total_ms=211486.164, avg_ms=6608.943, p95_ms=9047.511
- `persona/prompt_template/v2/new_decomp_schedule_v1.txt`: count=20, total_ms=209535.244, avg_ms=10476.762, p95_ms=11484.789
- `persona/prompt_template/v2/insight_and_evidence_v1.txt`: count=47, total_ms=187502.782, avg_ms=3989.421, p95_ms=8382.980
- `persona/prompt_template/v2/task_decomp_v3.txt`: count=4, total_ms=42484.836, avg_ms=10621.209, p95_ms=11073.716
- `persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt`: count=4, total_ms=26595.695, avg_ms=6648.924, p95_ms=8523.632
- `persona/prompt_template/v1/action_location_sector_v1.txt`: count=24, total_ms=16964.633, avg_ms=706.860, p95_ms=5153.789
- `persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt`: count=2, total_ms=12887.938, avg_ms=6443.969, p95_ms=6548.568
- `persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt`: count=69, total_ms=7512.078, avg_ms=108.871, p95_ms=157.638
- `persona/prompt_template/v2/planning_thought_on_convo_v1.txt`: count=2, total_ms=6912.284, avg_ms=3456.142, p95_ms=5741.682
- `persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt`: count=9, total_ms=6492.987, avg_ms=721.443, p95_ms=1006.291
- `persona/prompt_template/v2/generate_focal_pt_v1.txt`: count=3, total_ms=5119.725, avg_ms=1706.575, p95_ms=1832.492
- `persona/prompt_template/v1/action_object_v2.txt`: count=24, total_ms=4186.753, avg_ms=174.448, p95_ms=198.387
- `persona/prompt_template/v1/action_location_object_vMar11.txt`: count=24, total_ms=4186.619, avg_ms=174.442, p95_ms=195.261
- `persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt`: count=24, total_ms=3749.018, avg_ms=156.209, p95_ms=302.396
- `persona/prompt_template/v2/decide_to_react_v1.txt`: count=10, total_ms=2673.121, avg_ms=267.312, p95_ms=405.976
- `persona/prompt_template/v2/decide_to_talk_v2.txt`: count=10, total_ms=2495.066, avg_ms=249.507, p95_ms=362.588
- `persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt`: count=2, total_ms=1401.548, avg_ms=700.774, p95_ms=835.409

## Trace Coverage

- movement_chat: 238
- nonempty_retrieval: 110
- memory_kinds: `{"chat": 4, "event": 229, "thought": 85}`
- random_fns: `{"choice": 74, "sample": 28}`
