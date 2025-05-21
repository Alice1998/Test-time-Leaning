python run_benchmark_multiRound.py --task "TwentyQuestion" \
    --approach_name "DynamicCheatsheet_Cumulative" \
    --model_name "gpt-4o-2024-11-20" \
    --additional_flag_for_save_path "DynamicCheatsheet_Cumulative" \
    --save_directory "TEST_RESULTS" \
    --generator_prompt_path "prompts/generator_prompt_simple_TwentyQuestion.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative_hasReward_multiRound_TwentyQuestion.txt" \
    --max_n_samples 20 \
    --no_shuffle 1 \