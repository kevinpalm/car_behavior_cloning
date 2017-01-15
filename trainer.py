import pandas as pd
import numpy as np
from scipy.misc import imread

class ExampleGenerator():
	
	
	def __init__(self):
		
		# Load in the driving log
		self.df = pd.read_csv("data/driving_log.csv",
		header=None,
		names=["Center Image", "Left Image", "Right Image",
		"Steering Angle", "Throttle", "Break", "Speed"])
		
		# Shuffle indexes in prep for the first epoch
		self.shuffs = np.array(self.df.index)
		np.random.shuffle(self.shuffs)
		
		# Start a count for n epochs
		self.ecount = 0
		
	
	def verify_df(self):
		
		# Check to make sure something was loaded
		print(self.df.head())
		
		
	def epoch_reset(self):
		
		# Get a freshly shuffled index list
		self.shuffs = np.array(self.df.index)
		np.random.shuffle(self.shuffs)
		print(self.shuffs)
		
		# Add a count
		self.ecount += 1
		
	
	def read_image(self, fileloc):
		
		# Read in the image
		image = imread(fileloc)
		
		return image
		
	def n_examples(self, n):
		
		# Make sure we have enough examples left in this epoch
		if len(self.shuffs) < n:
			self.epoch_reset()
		
		# Pull the slice list from the shuffled indexes
		islice = self.shuffs[:n]
		self.shuffs = self.shuffs[n:]
		
		# Format the outputs
		angles = self.df.loc[islice, "Steering Angle"].values
		locations = self.df.loc[islice, "Center Image"].tolist()
		images = np.array([self.read_image(x) for x in locations])
		
		return images, angles
		


gen = ExampleGenerator()
images, angles = gen.n_examples(2)
print(images.shape)
print(angles)
