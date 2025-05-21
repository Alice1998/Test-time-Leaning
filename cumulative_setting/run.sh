python run_benchmark.py --task "AIME_2025" \
    --approach_name "default" \
    --model_name "deepseek-v3-250324" \
    --additional_flag_for_save_path "default" \
    --save_directory "TEST_RESULTS" \
    --generator_prompt_path "prompts/generator_prompt_WOcode.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative_WOcode.txt" \
    --max_n_samples 50 \
    --no_shuffle 1