import pandas as pd
import numpy as np
import random
import json

DEFAULT_OBJECT_DICT = {
    "Sports": ["Basketball", "Football", "Baseball", "Soccer ball", "Golf ball", "Tennis ball", "Volleyball", "Tennis racket", "Baseball bat", "Helmet"],
    "Animals": ["Cat", "Dog", "Horse", "Cow", "Sheep", "Rabbit", "Lion", "Tiger", "Bear", "Elephant"],
    "Fruits": ["Apple", "Banana", "Orange", "Strawberry", "Grape", "Watermelon", "Pineapple", "Mango", "Cantaloupe", "Peach"],
    "Vehicles": ["Car", "Truck", "Motorcycle", "Boat", "Airplane;Plane", "Train", "Bus", "Helicopter", "Scooter", "Ship"],
    "Clothes": ["Shirt", "Pants;Pant;Pair of pants", "Jacket", "Dress", "Skirt", "Belt", "Shoes;Shoe;Pair of shoes", "Boots;Boot;Pair of boots", "Socks;Sock;Pair of socks", "Hat", "Scarf"],
    "Electronics": ["Computer", "Smartphone", "Television;TV", "Headphone;Headphones;Pair of headphones", "Monitor;Computer monitor", "Camera", "Microwave;Microwave oven", "Refrigerator", "Blender", "Computer keyboard;Keyboard"],
    "Musical Instruments": ["Piano", "Guitar", "Drum;Drums", "Violin", "Saxophone", "Flute", "Trumpet", "Clarinet", "Harp", "Trombone"],
    "Furniture": ["Chair", "Table", "Bed", "Desk", "Couch", "Dresser", "Bookcase", "Nightstand", "Mattress", "Pillow"],
    "Office Supplies": ["Pen", "Paper;Piece of paper", "Stapler", "Printer", "Calculator", "Battery;Battery pack;Pack of batteries", "Toothbrush", "Toothpaste", "Pencil", "Sharpie", "Scissors;Pair of scissors", "Key", "Diary", "Calendar"],
    "Vegetables": ["Carrot", "Potato", "Broccoli", "Tomato", "Onion", "Spinach", "Corn", "Peas;Pea", "Celery", "Cucumber"],
    "Art": ["Painting;Canvas painting;Oil painting;Watercolor painting", "Paintbrush", "Canvas;Painting canvas", "Eraser;Pencil eraser", "Marker", "Glue;Glue stick;Bottle of glue", "Sculpture"],
    "Kitchen Tools": ["Knife", "Spoon", "Fork", "Plate", "Bowl", "Cooking pot;Pot", "Pan;Saucepan;Frying pan", "Cup", "Chopstick;Chopsticks;Pair of chopsticks", "Whisk"],
    "Nature": ["Rock", "Tree", "Bush", "Mountain", "Forest", "Ocean", "Sea", "Lake", "River", "Meteorite", "Cactus"],
    "Toys": ["Lego;Lego set", "Doll;Toy doll;Plush doll", "Kite", "Puzzle;Jigsaw puzzle"],
    "Jewelry": ["Earring;Earrings;Pair of earrings", "Necklace", "Bracelet", "Ring", "Brooch", "Hairclip", "Pendant", "Watch", "Locket"],
    "Garden Supplies": ["Gloves;Glove;Pair of gloves", "Shovel", "Rake", "Watering can", "Lawn mower"],
    "Tools": ["Hammer", "Screwdriver", "Wrench", "Saw", "Pliers;plier;Pair of pliers", "Drill"]
}


if __name__ == "__main__":
	word_list = []
	SELECT_NUMBER = 10
	QUESTION_NUMBER = 5
	for type_name in DEFAULT_OBJECT_DICT:
		for word in DEFAULT_OBJECT_DICT[type_name]:
			single_w = word.split(";")[0]
			word_list.append(single_w)
		# word_list += DEFAULT_OBJECT_DICT[type_name]
	
	print(len(word_list))

	action_prompt = "Now, it is your turn to ask question [[TURN]]"
	result_list = []
	for i in range(100):
		# select_word_list = random.sample(word_list, SELECT_NUMBER)
		select_word_list = word_list
		random.shuffle(select_word_list)
		secret_word = random.choice(select_word_list)
		word_list_str = ', '.join(select_word_list)
		# prompt = f"You are playing the game Five Questions. There are {SELECT_NUMBER} possible words: {word_list_str}. You need to guessed the only secret words among the {SELECT_NUMBER} candidiates. You can ask {QUESTION_NUMBER} Yes/No questions to find the word. You will get response in Yes, No, or Invalid for invalid questions."
		prompt =  f"You are playing the game Five Question. You will be given {SELECT_NUMBER} candidate words. One of them is the answer word. You can ask up to {QUESTION_NUMBER} yes/no questions to identify the answer word. You will get a Yes, No, or Invalid answer for each question."
		question = f"The candidate words are {word_list_str}."
		question_dict = {"input":question,
						# 'word_list': word_list_str,
						"target":secret_word,
						'candidate_word_list': select_word_list}
		result_list.append(question_dict)
		# import ipdb; ipdb.set_trace()

	save_file_path = "./word=157_count=100.jsonl"
	with open(save_file_path,'w') as f:
		for item in result_list:
			f.write(json.dumps(item, ensure_ascii=False) + "\n")

		


