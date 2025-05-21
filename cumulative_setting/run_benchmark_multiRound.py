from datetime import datetime
import json
import os
import pandas as pd
import numpy as np

from datasets import load_dataset, load_from_disk
from tap import TAP
from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import eval_TwentyQuestion, eval_sudoku,eval_for_GameOf24, eval_for_multiple_choice, eval_for_exact_matching_with_no_punctuation, eval_equation_balancer

from dotenv import load_dotenv

PREDEFINED_PROMPTS = {
	"GameOf24": f"Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24.",
	"Sudoku": "You're solving a 9×9 Sudoku puzzle. Sudoku rule: each column, each row, and each of the nine 3×3 subgrids that compose the grid contains all of the digits from 1 to 9. _ represents a cell to be filled.\n"
}

# PvP_PREDEFINED_PROMPTS = {
# 	"TwentyQuestion": "You are playing the game Twenty Question. You will be given 157 candidate words. One of them is the answer word. You can ask up to 20 yes/no questions to identify the answer word. You will get a Yes, No, or Invalid answer for each question. Note, you can NOT ask questions about the letters of the answer word. Questions can only be asked around semantics."
# }

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Arguments to pass to the program.")

	# Task name
	parser.add_argument('--task', type=str, default="GameOf24")

	# Approach name
	parser.add_argument('--approach_name', type=str, default="DynamicCheatsheet_Cumulative")

	# Model name
	parser.add_argument('--model_name', type=str, default="openai/gpt-4o-mini")

	# Paths to the prompt files
	parser.add_argument('--generator_prompt_path', type=str, default="prompts/simple_generator.txt")
	parser.add_argument('--cheatsheet_prompt_path', type=str, default=None)

	# Additional model-related arguments
	parser.add_argument('--max_tokens', type=int, default=2048)
	parser.add_argument('--temperature', type=float, default=0.0)
	parser.add_argument('--max_num_rounds', type=int, default=1)

	parser.add_argument('--execute_python_code', action='store_true')
	parser.add_argument('--initialize_cheatsheet_path', type=str, default=None)
	parser.add_argument('--retrieve_top_k', type=int, default=3)

	# Continue from the previous run
	parser.add_argument('--continue_from_last_run_path', type=str, default=None)

	# Additional save-path-related arguments
	parser.add_argument('--save_directory', type=str, default="results")
	parser.add_argument('--additional_flag_for_save_path', type=str, default="")
	parser.add_argument('--max_n_samples', type=int, default=-1)
	parser.add_argument('--no_shuffle', type=int, default = 0)

	return parser.parse_args()


# class Arguments(TAP):
#	 """
#	 Arguments to the pass to the program.
#	 """
#	 # Task name
#	 task: str = "GameOf24"
	
#	 # Approach name
#	 approach_name: str = "DynamicCheatsheet_Cumulative"

#	 # Model name
#	 model_name: str = "openai/gpt-4o-mini"

#	 # Paths to the prompt files
#	 generator_prompt_path: str = "prompts/simple_generator.txt"
#	 cheatshet_prompt_path: str = None

#	 # Additional model-related arguments
#	 max_tokens: int = 2048
#	 temperature: float = 0.0
#	 max_num_rounds: int = 1

#	 execute_python_code: bool = True
#	 initialize_cheatsheet_path: str = None
#	 retrieve_top_k: int = 3

#	 # Continue from the previous run
#	 continue_from_last_run_path: str = None

#	 # Additional save-path-related arguments
#	 save_directory: str = "results"
#	 additional_flag_for_save_path: str = ""
#	 max_n_samples: int = -1
#	 no_shuffle: bool = False


def read_file(file_path: str) -> str:
	"""
	Read the file and return the content.
	"""
	with open(file_path, "r") as file:
		return file.read()

	
def write_jsonl(file_path, data):
	"""
	Save the outputs to a file.
	"""
	dir_path = os.path.dirname(file_path)
	os.makedirs(dir_path, exist_ok=True)

	with open(file_path, "w") as file:
		for line in data:
			file.write(json.dumps(line) + "\n")


def main(args):
	"""
	Main function to run the benchmark.
	"""
	# Load the environment variables
	load_dotenv("config.env")

	# Read the prompt files
	args.generator_prompt = read_file(args.generator_prompt_path)
	if args.cheatsheet_prompt_path:
		args.cheatsheet_prompt = read_file(args.cheatsheet_prompt_path)
	else:
		args.cheatsheet_prompt = "(empty)"

	args.max_n_samples = int(args.max_n_samples)


	# Initialize the language model
	model = LanguageModel(
		model_name=args.model_name,
	)

	# Add a flag to the save path if the code execution is not allowed
	if not args.execute_python_code:
		args.additional_flag_for_save_path += "_no-code-execution"

	# Load the dataset based on the task name
	if args.task in PREDEFINED_PROMPTS and args.task not in ['Sudoku',"TwentyQuestion","P3_Test"]:
		if args.task == 'GameOf24':
			dataset = load_dataset("/home/jiayinwang/lab/dynamic-cheatsheet/data/meta_prompting")['train']
			# print('[dataset] size='s,len(dataset))
			# dataset = dataset[:10]
			# print('[dataset] leave size=10')
			# import ipdb; ipdb.set_trace()
			# dataset = dataset[args.task]
		else:
			dataset = load_dataset("turingmachine/meta-prompting")
			dataset = dataset[args.task]
	elif args.task in ["GPQA_Diamond", "AIME_2020_2024", "AIME_2024", "AIME_2025", "MMLU_Pro_Physics", "MMLU_Pro_Engineering", "MathEquationBalancer"]:
		dataset = load_from_disk(f"data/{args.task}")
	elif args.task in ['Sudoku']:
		dataset_train = pd.read_parquet(f"./data/0_{args.task}/train.parquet", engine="pyarrow").head(5)
		dataset_test =  pd.read_parquet(f"./data/0_{args.task}/test.parquet", engine="pyarrow").head(50)
		prompts = dataset_train['prompt'].tolist()
		prompts = [p[0]['content'] for p in prompts]
		prompts = [p.split('[Current Board]\n')[1] for p in prompts]
		# dataset_train['input'] = prompts
		answer_list = dataset_train['reward_model'].tolist()
		answer_list = [str(ans['ground_truth']) for ans in answer_list]
		# dataset_train['target'] = answer_list
		length = len(dataset_train)
		dataset_train = [{"input":prompts[index],"target":answer_list[index]} for index in range(length)]
		
		prompts = dataset_test['prompt'].tolist()
		prompts = [p[0]['content'] for p in prompts]
		prompts = [p.split('[Current Board]\n')[1] for p in prompts]
		# dataset_test['input'] = prompts
		answer_list = dataset_test['reward_model'].tolist()
		answer_list = [str(ans['ground_truth']) for ans in answer_list]
		# dataset_test['target'] = answer_list
		length = len(dataset_test)
		dataset_test = [{"input":prompts[index],"target":answer_list[index]} for index in range(length)]
	
	elif args.task in ['TwentyQuestion']:
		# file_path = "data/twentyQuestion/word=10_count=600.jsonl"
		file_path = "data/twentyQuestion/word=157_count=100.jsonl"
		dataset = []
		with open(file_path, 'r') as f:
			for line in f:
				item = json.loads(line.strip())  # 去掉末尾换行符，然后解析 JSON
				dataset.append(item)
		dataset_train = dataset[:100]
		dataset_test = dataset[100:]
		candidate_word_list = ['Airplane', 'Apple', 'Banana', 'Baseball', 'Baseball bat', 'Basketball', 'Battery', 'Bear', 'Bed', 'Belt', 'Blender', 'Boat', 'Bookcase', 'Boots', 'Bowl', 'Bracelet', 'Broccoli', 'Brooch', 'Bus', 'Bush', 'Cactus', 'Calculator', 'Calendar', 'Camera', 'Cantaloupe', 'Canvas', 'Car', 'Carrot', 'Cat', 'Celery', 'Chair', 'Chopstick', 'Clarinet', 'Computer', 'Computer keyboard', 'Cooking pot', 'Corn', 'Couch', 'Cow', 'Cucumber', 'Cup', 'Desk', 'Diary', 'Dog', 'Doll', 'Dress', 'Dresser', 'Drill', 'Drum', 'Earring', 'Elephant', 'Eraser', 'Flute', 'Football', 'Forest', 'Fork', 'Gloves', 'Glue', 'Golf ball', 'Grape', 'Guitar', 'Hairclip', 'Hammer', 'Harp', 'Hat', 'Headphone', 'Helicopter', 'Helmet', 'Horse', 'Jacket', 'Key', 'Kite', 'Knife', 'Lake', 'Lawn mower', 'Lego', 'Lion', 'Locket', 'Mango', 'Marker', 'Mattress', 'Meteorite', 'Microwave', 'Monitor', 'Motorcycle', 'Mountain', 'Necklace', 'Nightstand', 'Ocean', 'Onion', 'Orange', 'Paintbrush', 'Painting', 'Pan', 'Pants', 'Paper', 'Peach', 'Peas', 'Pen', 'Pencil', 'Pendant', 'Piano', 'Pillow', 'Pineapple', 'Plate', 'Pliers', 'Potato', 'Printer', 'Puzzle', 'Rabbit', 'Rake', 'Refrigerator', 'Ring', 'River', 'Rock', 'Saw', 'Saxophone', 'Scarf', 'Scissors', 'Scooter', 'Screwdriver', 'Sculpture', 'Sea', 'Sharpie', 'Sheep', 'Ship', 'Shirt', 'Shoes', 'Shovel', 'Skirt', 'Smartphone', 'Soccer ball', 'Socks', 'Spinach', 'Spoon', 'Stapler', 'Strawberry', 'Table', 'Television', 'Tennis ball', 'Tennis racket', 'Tiger', 'Tomato', 'Toothbrush', 'Toothpaste', 'Train', 'Tree', 'Trombone', 'Truck', 'Trumpet', 'Violin', 'Volleyball', 'Watch', 'Watering can', 'Watermelon', 'Whisk', 'Wrench']
	else:
		raise ValueError(f"Task {args.task} is not recognized. Please make sure the task name is correct.")
	
	# If the previous run parameter is provided, make sure that the provided arguments are consistent with those found in the previous run
	if args.continue_from_last_run_path:
		if not os.path.exists(args.continue_from_last_run_path):
			raise ValueError(f"The provided path {args.continue_from_last_run_path} does not exist.")
		
		# Read the previous run parameters from the previous run file and compare them with the provided arguments
		previous_run_param_path = args.continue_from_last_run_path.replace(".jsonl", "_params.json")
		# Read the previous run parameters
		with open(previous_run_param_path, "r") as file:
			previous_run_params = json.load(file)

		# Compare the provided arguments with the previous run parameters
		args_keys = ["generator_prompt_path", "cheatsheet_prompt_path", "temperature", "execute_python_code", "task", "model_name", "approach_name", "max_num_rounds"]

		# Compare the provided arguments with the previous run parameters
		for key in args_keys:
			if getattr(args, key) != previous_run_params[key]:
				raise ValueError(f"Warning: The provided argument {key} is inconsistent with the previous run. The previous run value is {previous_run_params[key]}.")
		
		# Create a new save path name based on the previous run path
		args.save_path_name = args.continue_from_last_run_path.replace(".jsonl", "_continued.jsonl")
	else:
		# Create a new save path name based on the current time stamp
		time_stamp = datetime.today().strftime('%Y-%m-%d-%H-%M')
		args.save_path_name = f"{args.save_directory}/{args.task}/{args.model_name}_{args.approach_name}_{time_stamp}_{args.additional_flag_for_save_path}.jsonl"
		
		# Create the directory if it does not exist
		dir_path = os.path.dirname(args.save_path_name)
		os.makedirs(dir_path, exist_ok=True)

	save_param_path = args.save_path_name.replace(".jsonl", "_params.json")
	dir_path = os.path.dirname(save_param_path)
	os.makedirs(dir_path, exist_ok=True)
	
	# Save the arguments to a file
	with open(save_param_path, "w") as file:
		# json.dump(args.as_dict(), file, indent=4)
		json.dump(vars(args), file, indent=4)

	# Initialize the cheatsheet
	cheatsheet = "(empty)"
	if args.initialize_cheatsheet_path is not None:
		with open(args.initialize_cheatsheet_path, "r") as file:
			cheatsheet = file.read()
	
	# Initialize the outputs and the generator outputs so far
	outputs = []
	generator_outputs_so_far = []
	if args.continue_from_last_run_path:
		# Load the previous run
		with open(args.continue_from_last_run_path, "r") as file:
			outputs = [json.loads(line) for line in file.readlines()]

		# Load the previous cheatsheet from the last output
		# cheatsheet = outputs[-1]["final_cheatsheet"]
		# import ipdb;ipdb.set_trace()
		cheatsheet = outputs[-1]["reflector"]["cheatsheet"]
		
		# generator_outputs_so_far = [output["final_output"] for output in outputs]
		# generator_outputs_so_far = [output["outsput"] for output in outputs]
		generator_outputs_so_far = [output['game_context'] for output in outputs]
		# Print the details
		print(f"Continuing from the previous run at {args.continue_from_last_run_path}.")
		print(f"Loaded {len(outputs)} examples from the previous run.")
		print(f"Most recent cheatsheet: {cheatsheet}")
		print("-" * 50)

	# Split the dataset by taking the first n samples
	# dataset = dataset.select(range(args.max_n_samples))

	# Shuffle the dataset if the no_shuffle flag is not set
	if not args.no_shuffle:
		dataset = dataset.shuffle(seed=10)

	# Initialize the questions and the embeddings
	questions = None
	embeddings = None
	if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"]:
		df = pd.read_csv(f"embeddings/{args.task}.csv")
		questions = df["input"].tolist()
		embeddings = df["embedding"]
		embeddings = embeddings.apply(eval)
		embeddings = np.array(embeddings.tolist()) # (N, 1536)

		# Re-order the embeddings based on the order of the dataset inputs
		# import ipdb; ipdb.set_trace()
		dataset_inputs = [example["input"] for example in dataset]
		indices = [questions.index(input) for input in dataset_inputs]
		embeddings = embeddings[indices]
		questions = dataset_inputs
	elif args.approach_name != 'DynamicCheatsheet_Cumulative_hasReward':
		dataset = [example["input"] for example in dataset]

	# previous_inputs = []
	# for phase, dataset in [('train',dataset_train), ('test', dataset_test)]:
	for phase, dataset in [('train',dataset_train)]:
		questions = [example["input"] for example in dataset]
		# questions = dataset['input'].tolist()
		start_idx = len(outputs)
		correct_so_far = 0
		total_so_far = 0
		hit_position_list = []
		# previous_inputs = []

		# Iterate over the dataset
		for idx, example in enumerate(dataset):
			# original_input = dataset[idx]["input"]
			original_target = dataset[idx]["target"]
			orig_input = example["input"]
			if args.task in PREDEFINED_PROMPTS:
				input = f"{PREDEFINED_PROMPTS[args.task]}\n\nQuestion #{idx+1}:\n{orig_input}"
			# elif args.task in PvP_PREDEFINED_PROMPTS:
			# 	input = f"{PvP_PREDEFINED_PROMPTS[args.task]}\n\nGame #{idx+1}:\n{orig_input}"
			# 	if args.task == 'TwentyQuestion':
			# 		candidate_word_list = example['candidate_word_list']
			elif args.task == "TwentyQuestion":
				input = f"Game #{idx+1}:\n"
			else:
				input = f"Question #{idx+1}:\n{orig_input}"

			# previous_inputs.append(input)

			if args.task == "AIME_2020_2024" or args.task == "AIME_2024" or args.task == "AIME_2025":
				# Add a specific format to the input for the AIME tasks
				input = f"{input} (Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; make sure to pay attention to the desired format of the final answer though.)"
			elif args.task == "MathEquationBalancer":
				# Add a specific format to the input for the MathEquationBalancer task
				input = f"Below is an equation with missing operators. Your task is to fill in the blanks with the correct mathematical operators: +, -, *, or /. Ensure that the equation is correct once the operators are added. The operators should be placed in the sequence they appear from left to right. Include the full equation with the operators filled in. For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6.\n\nEquation: {input}"

			# Skip the examples that have been already seen in the previous run
			if idx < start_idx:
				continue

			# Print the details
			print(f"### Example {idx+1} ###")
			finish_round = -1
			game_context = input
			game_info_list = []
			for round_idx in range(20):
				print(f"### Example {idx+1} ### Round {round_idx}")
				# Generate the output from the language model using the DynamicCheatsheet approach or other approaches
				if args.task == 'TwentyQuestion':
					input = game_context + f"\nNow, it is your turn to ask the #{round_idx+1} question. Provide your question in the format:FINAL ANSWER:\n<answer>\nYOUR QUESTION HERE\n</answer>"
				# import ipdb; ipdb.set_trace()
				output_dict = model.generate_multi_actor(
					approach_name=args.approach_name,
					input_txt=input,
					cheatsheet=cheatsheet,
					generator_template=args.generator_prompt,
					cheatsheet_template=args.cheatsheet_prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					max_num_rounds=args.max_num_rounds,
					allow_code_execution=args.execute_python_code,
					code_execution_flag="EXECUTE CODE!",
					original_input_corpus=questions[:idx+1],
					original_input_embeddings=embeddings[:idx+1] if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"] else None,
					generator_outputs_so_far=[], #generator_outputs_so_far,
					retrieve_top_k=args.retrieve_top_k,
					target_answer = original_target,
				)
				final_answer = output_dict["final_answer"]
				if args.task == 'TwentyQuestion':
					round_result, env_feedback = eval_TwentyQuestion(final_answer, original_target, candidate_word_list, model)
					game_context += f"\n{final_answer} {env_feedback}"
					if round_result:
						finish_round = round_idx
						round_info_dict = {
							'round_idx':round_idx,
							'input':input,
							'output':output_dict['final_output'],
							"output_extract_answer":output_dict['final_answer'],
							'output_feedback':env_feedback,
							'hit':finish_round,
						}
						game_info_list.append(round_info_dict)
						break
				round_info_dict = {
					'round_idx':round_idx,
					'input_question':input,
					'input':output_dict['input_instruction'],
					'output':output_dict['final_output'],
					"output_extract":output_dict['final_answer'],
					'feedback':env_feedback,
					'hit':finish_round
				}
				game_info_list.append(round_info_dict)
			if round_result:
				feedback = f"You get the right answer word in the {finish_round+1} question!"
			else:
				feedback = f"You did NOT get the right answer word in 20 questions! The answer word is {original_target}."
			# if phase == 'train':
			# 	generator_outputs_so_far.append(output_dict["final_output"])

			# outputs.append({
			# 		"input": input,
			# 		"target": original_target,
			# 		"raw_input": original_input,
			# 		**output_dict,
			# 	})
			
			reflection = model.generate_multi_reflection(
					approach_name=args.approach_name,
					game_content = game_context,
					env_feedback = feedback,
					cheatsheet=cheatsheet,
					generator_template=args.generator_prompt,
					cheatsheet_template=args.cheatsheet_prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					max_num_rounds=args.max_num_rounds,
					allow_code_execution=args.execute_python_code,
					code_execution_flag="EXECUTE CODE!",
					original_input_corpus=questions[:idx+1],
					original_input_embeddings=embeddings[:idx+1] if args.approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis", "FullHistoryAppending"] else None,
					generator_outputs_so_far=[], #generator_outputs_so_far,
					retrieve_top_k=args.retrieve_top_k,
					target_answer = original_target,
			)

			outputs.append({
					'game_NO':idx,
					"actor":game_info_list,
					'game_context':game_context,
					'feedback':finish_round,
					'reward':round_result,
					'reflector':reflection
			})
			
			if phase == 'train':
				# cheatsheet = output_dict["final_cheatsheet"]
				cheatsheet = reflection['cheatsheet']
			# current_sheet = output_dict["final_cheatsheet"]
			# final_answer = output_dict["final_answer"]
			current_sheet = reflection['cheatsheet']

			## FOR DEBUGGING PURPOSES
			# import pdb; pdb.set_trace()
			print(f"@ CHEATSHEET:\n{current_sheet}")
			print('- ' * 50)
			# print(f"Input: {input}")		
			# print(f"Final answer: {final_answer}")
			print("Game Content\n",game_context)
			print(f"Target: {original_target}")
			print("**" * 50)

			# if args.task == "GameOf24":
			# 	result = eval_for_GameOf24(original_input, final_answer)
			# elif args.task in ["AIME_2025", "AIME_2024", "AIME_2020_2024"]:
			# 	result = eval_for_exact_matching_with_no_punctuation(final_answer.lower(), original_target.lower())
			# elif args.task in ["GPQA_Diamond", "MMLU_Pro_Engineering", "MMLU_Pro_Physics"]:
			# 	result = result = eval_for_multiple_choice(input, final_answer, original_target)
			# elif args.task == "MathEquationBalancer":
			# 	result = eval_equation_balancer(None, final_answer, original_target)
			# elif args.task == 'Sudoku':
			# 	result = eval_sudoku(final_answer,original_target)
			# else:
			# 	raise ValueError(f"Task {args.task} not supported.")
			
			if round_result: # result
				correct_so_far += 1
			total_so_far += 1
			hit_position_list.append(finish_round)
			print("[Reward] hit position=",finish_round)

			print(f"---- Correct so far: {correct_so_far}/{total_so_far}")
			print(f"---- Hit Position: ",hit_position_list)
			print("###" * 50)

			# Temporarily save the outputs to a file after each example
			write_jsonl(args.save_path_name, outputs)

			if args.max_n_samples > 0 and idx == args.max_n_samples - 1:
				break
		
	# Save the entire outputs to a file
	write_jsonl(args.save_path_name, outputs)

		
if __name__ == "__main__":
	# args = Arguments().parse_args()
	args = parse_args()
	main(args)