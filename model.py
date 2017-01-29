import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Flatten, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

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
		if len(self.shuffs) > n:
		
			# Pull the slice list from the shuffled indexes
			islice = self.shuffs[:n]
			self.shuffs = self.shuffs[n:]
			
			# Format the outputs
			angles = self.df.loc[islice, "Steering Angle"].values
			locations = self.df.loc[islice, "Center Image"].tolist()
			images = np.array([self.read_image(x) for x in locations])
			
			return images, angles
		
		else:
			return False, False
	
	def all_examples(self):
		
		# Store all the angles and read images
		print("Formatting all examples...")
		angles = self.df["Steering Angle"].values
		images = np.array([self.read_image(x) for x in
		self.df["Center Image"]])
		
		return images, angles
		
		

class NNet():
	
	def model_def(self, fresh_start=True):
		
		if fresh_start == False:

			# If not having a fresh start, load the model
			json_file = open('model.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_model_json)
			model.load_weights("model.h5")
			print("Loaded model")
		
		else:
			
			# 2x2 pool
			model = Sequential()
			model.add(AveragePooling2D(pool_size=(2, 2),
			input_shape=(160, 320, 3)))
			
			# 7x7 32 /2 conv
			model.add(Convolution2D(32, 7, 7, border_mode="same",
			subsample=(2, 2)))
			model.add(BatchNormalization())
			model.add(Activation("relu"))
			model.add(Dropout(0.8))
			
			# 2x2 pool
			model.add(AveragePooling2D(pool_size=(2, 2)))
			
			# 3x3 64 conv
			model.add(Convolution2D(64, 3, 3, border_mode="same"))
			model.add(BatchNormalization())
			model.add(Activation("relu"))
			model.add(Dropout(0.8))
			
			# 2x2 pool
			model.add(AveragePooling2D(pool_size=(2, 2)))
			
			# 3x3 128 conv
			model.add(Convolution2D(128, 3, 3, border_mode="same"))
			model.add(BatchNormalization())
			model.add(Activation("relu"))
			model.add(Dropout(0.8))
			
			# 2x2 pool
			model.add(AveragePooling2D(pool_size=(2, 2)))
			
			# 3x3 256 conv
			model.add(Convolution2D(256, 3, 3, border_mode="same"))
			model.add(BatchNormalization())
			model.add(Activation("relu"))
			model.add(Dropout(0.8))
			
			# Define fully connected layer
			model.add(Flatten())
			model.add(Dense(7680, init="normal"))
			model.add(BatchNormalization())
			model.add(Activation("relu"))
			model.add(Dropout(0.5))
			
			# Output layer
			model.add(Dense(1))

		
		# Compile model
		model.compile(loss='mean_absolute_error', optimizer='adam')
		
		return model
		
	def full_train(self, train_x, test_x, train_y, test_y, fs):
		
		# Read the model structure
		self.model = self.model_def(fresh_start=fs)
		
		# Save the model structure
		with open("model.json", "w") as outjson:
			outjson.write(self.model.to_json())
			
		# Define image generator and the callbacks
		imgen = ImageDataGenerator(channel_shift_range=0.1)
		callbacks = [EarlyStopping(patience=10),
		ModelCheckpoint("model.h5", save_best_only=True,
		save_weights_only=True, monitor='val_loss', mode="min")]
		
		# Fit the model, saving checkpoints as we go
		self.model.fit_generator(imgen.flow(train_x, train_y,
		batch_size=128), samples_per_epoch=len(train_y), nb_epoch=1000,
        verbose=1, validation_data=(test_x,
        test_y), callbacks=callbacks)
		score = self.model.evaluate(test_x,
		test_y, verbose=0)
		
		# Final summary
		print('Test score:', score)
		
	def save_model(self):
		with open("model.json", "w") as outjson:
			outjson.write(self.model.to_json())
		self.model.save_weights("model.h5")
		
def fresh_train():

	# Define the generator and the model
	gen = ExampleGenerator()
	model = NNet()
	
	# Train on the first examples, forcing a fresh start
	images, angles = gen.n_examples(4080)
	train_x, test_x, train_y, test_y = train_test_split(images, angles)
	model.full_train(train_x, test_x, train_y, test_y, True)
	
	# Train on the next examples without fresh starts
	images, angles = gen.n_examples(4080)
	while images is not None:
		train_x, test_x, train_y, test_y = train_test_split(images,
		angles)
		model.full_train(train_x, test_x, train_y, test_y, False)
		images, angles = gen.n_examples(4080)
	
	# Output some example predictions for dummy checking if needed
	output = pd.DataFrame()
	output["True"] = angles
	output["Predicted"] = model.model.predict(images)
	output.to_csv("predicts.csv")
	
def reload_train():

	# Define the generator and the model
	gen = ExampleGenerator()
	model = NNet()
	
	# Train on batches of examples without any fresh starts
	images, angles = gen.n_examples(4080)
	while images is not None:
		train_x, test_x, train_y, test_y = train_test_split(images,
		angles)
		model.full_train(train_x, test_x, train_y, test_y, False)
		images, angles = gen.n_examples(4080)

	# Output some example predictions for dummy checking if needed
	output = pd.DataFrame()
	output["True"] = angles
	output["Predicted"] = model.model.predict(images)
	output.to_csv("predicts.csv")
	
reload_train()


