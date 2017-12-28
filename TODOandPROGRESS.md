# TODO
##### General method / technique implementations:
- Cross validation
- two layer bidirectional
- attention
- ensembling
- We need to set up a way (google docs spread cdsheet?) to keep the record on each of the architectures/ methods we tried
   and the effect it had on the performance to report it.
- Save word embeddings, and save a final version we will use, so we do not create it every time
- 

##### Fine Tunings:

- Tune hyperparameters : dropout rate, batch size, learning rate, maximum gradient for clipping
- Architecture tunings: More neurons in dense layer, more dense layers, add a convolutional layer,
more RNN layers
- Play with data dimentionality: Try 50D - 300D glove + 300D and reduced with PCA to different values


# Things tried/ implemented to talk about

- Better tokenisation : Made it so that we get the least amount of OOVs in our word embeddings
- Better pipeline : use lowercased versions of tokens if the pretrained word embeddings do not create
- Two methods for getting word embeddings, all of which assign OOVs to the average embedding of non-used glove words and PADs to 0
    - Using glove to get a vocabulary and a  embeddings and then filter our dataset through this vocab
    - Using the training data to get the vocab, then filter the vocab through glove and randomly allocate words not in glove
- After getting word embeddings , dimentionality has been varied using these different methods:
    - Using the Glove embeddings with glove6B50D
    - Using PCA to reduce the 300D glove embeddings to 50D
- For sentence embeddings we used (ordered by performance):
    - Single direction RNNS 1 layer
    - BI direction RNNS 1 layer
    - Single direction RNNS 2 layers
- After the sentence embedding, we used MLPs for training on the embeddings with the following architectures
    - Architecture 1 : 2 hidden layers of 512 and 256 neurons and an output layer of 25 neurons - classification
    - Architecture 2 : 2 hidden layers of 512 and 256 neurons and an output layer of 5 neurons - regression
- For the output and training, and classification, we used the following:
    - An output layer of 25 neurons - doing classification. We are predicting the probability of
    each sentence being at each position, and then classify each sentence to a position using a
    probability maximisation algorithm
        - Losses used : cross-entropy
    - An output layer of 5 neurons - doing regression. We are predicting the position of each sentence
    directly.
        - Losses used : MSE
- Probability maximisation algorithms for classification:
    - Greedy (initially)
    - Hungarian method
- Optimisation algorithms tried:
    - Adam
- Dropout configurations tried:
    - 0.5 dropout layer after each layer in the architecture.
- Overfitting prevention techniques used:
    - Early stopping
- Other deep learning techniques
    - Batch normalization between the feedforward layers - didn't improve
    - Gradient clipping
        - By value
        - By norm - this works better
- Loading training data
    - we create new points in the dataset by mixing up the order of the sentences
    - Random order of data in batches each epoch


# Open Questions
- Is dropout dropped automatically in feed-forward in TF?

