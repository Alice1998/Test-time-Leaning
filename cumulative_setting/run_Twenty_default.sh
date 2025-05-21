python run_benchmark_multiRound.py --task "TwentyQuestion" \
    --approach_name "default" \
    --model_name "deepseek-v3-250324" \
    --additional_flag_for_save_path "default" \
    --save_directory "TEST_RESULTS" \
    --generator_prompt_path "prompts/generator_prompt_simple_TwentyQuestion.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative_hasReward_multiRound_TwentyQuestion.txt" \
    --max_n_samples 20 \
    --no_shuffle 1