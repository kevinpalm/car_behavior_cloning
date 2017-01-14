import pandas as pd
import numpy as np


class ExampleGenerator():
	
	def __init__(self):
		
		# Load in the driving log
		self.df = pd.read_csv("data/driving_log.csv",
		header=None,
		names=["Center Image", "Left Image", "Right Image",
		"Steering Angle", "Throttle", "Break", "Speed"])
	
	def verify_df(self):
		
		# Check to make sure something was loaded
		print(self.df.head())
		
	def epoch_reset(self):
		
		# Get a freshly shuffled index list
		self.shuffled = np.array(self.df.index)
		np.random.shuffle(self.shuffled)
		print(self.shuffled)


gen = ExampleGenerator()
gen.epoch_reset()
