# Behavioral Cloning Project
My work on the third Udacity self-driving car project - behavioral cloning in a driving simulation. Running the model requires
the Udacity Self-drving Car Nanodegree simulator.

## Architecture
My final architecture ended up as:

* 2x2 average pool
* 7x7 32 conv (w/ batch normalization, relu activation, and 0.8 dropout)
* 2x2 average pool
* 3x3 64 conv (w/ batch normalization, relu activation, and 0.8 dropout)
* 2x2 average pool
* 3x3 128 conv (w/ batch normalization, relu activation, and 0.8 dropout)
* 2x2 average pool
* 3x3 256 conv (w/ batch normalization, relu activation, and 0.8 dropout)
* Fully connected layer 7680 (w/ batch normalization, relu activation, and 0.5 dropout)
* Output layer 1

## Training
I used an adam optimizer and chose mean absolute error for loss. I used the built in Keras image generator to augment my data with
0.1 channel shift as it was fed in batch sizes of 128, in sets of 4080. I chose to keep each set of training data small because I 
wanted to keep all my training images in memory for speed, and because I found it super interesting to check my model's progress
frequently. One quarter of each training set was reserved for validation, and throughout training I only saved model weights which
improved each validation loss. Each training set was allowed up to 1,000 epochs, but would terminate early after 25 epochs without
improvement to validation loss.

I started out with training data only from course 1 and of "normal driving" in the sense that I just hit record and did a few laps.
Once I settled on an architecture, I switched to a "data as needed" approach - I specifically drove to bolster where my model was
having problems like correcting lane position, tight turns, and crossing the bridge. I also started training on course 2 at this
point. Finally, I did some more normal driving to smoothen the model out again. In total, I used about 50,000 recorded images.

## Reflection
What an awesome project! I spent a lot of time playing with architectures, and feel like I'm starting to gain an appreciation for
the level of effort that must have gone into designing those top-notch competition and production networks. It's also sinking in
how much time it takes to train a network from scratch.

Unfortunately, my model is not polished at this point. It can navigate the whole course 1, and does okay on course 2, but it's not
smooth. Mostly what is lacking is good data. I definitely intend to revisit the project and polish things up. I've learned so much,
and as soon as I can afford to tie up my computer with a day or two of training I will!
