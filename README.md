# How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison


0. set the API key in config.env

1. For evaluation setting of limited number of experience, see ./fixed_setting.
python agent.py (WITH_HISTORY 0: without experience; 1 with raw experience; 2 with human policy; 4 with model-derived policy)

2. For cumulative setting, see ./cumulative_setting
w/o experience ``bash run_Twenty_default.sh``, w/ experience ``bash run_Twenty.sh``