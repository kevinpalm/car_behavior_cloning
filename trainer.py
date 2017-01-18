import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from sklearn.model_selection import train_test_split

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
		image = imresize(image, 0.1)
		
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
	
	def all_examples(self):
		
		# Store all the angles and read images
		angles = self.df["Steering Angle"].values
		images = np.array([self.read_image(x) for x in
		self.df["Center Image"]])
		
		return images, angles
		
		

class NNet():
	
	def model_def(self):
		
		# Init model
		model = Sequential()
		
		# First layer
		model.add(Flatten(input_shape=(16, 32, 3)))
		model.add(Dense(922, init='normal', activation='relu'))
		
		# Output layer
		model.add(Dense(1, init='normal'))
		
		# Compile model
		model.compile(loss='mean_squared_error', optimizer='adam',
		metrics=["mean_absolute_error"])
		
		return model
		
	def full_train(self, train_x, test_x, train_y, test_y):
		self.model = self.model_def()
		self.model.fit(train_x, train_y,
		batch_size=128, nb_epoch=10,
        verbose=1, validation_data=(test_x, test_y))
		score = self.model.evaluate(test_x, test_y, verbose=0)
		print('Test score:', score[0])
		print('Test mean absolute error:', score[1])

gen = ExampleGenerator()
images, angles = gen.all_examples()
train_x, test_x, train_y, test_y = train_test_split(images, angles)

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

model = NNet()
model.full_train(train_x, test_x, train_y, test_y)
