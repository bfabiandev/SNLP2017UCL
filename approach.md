### Overview
In this task of finding the sentence order in a paragraph, our approach was broken down into the following tasks:

+ Tokenising, making a vocabulary  and creating word embeddings.
+ Creating Sentence embeddings using the word embeddings.
+ Making sentence order predictions based on the sentence embeddings.

In the following sections, we break down how we tackled each of these components in detail.

### Word embeddings and Tokenisation
The initial model provided had a fairly simple approach to data preprocessing and tokenisation. The steps followed where (1) Tokenise the sentences into words by splitting them on white-spaces, (2) making a vocabulary (that maps tokens to unique integers) out of the tokens found in the training set, (3) add OOV and PAD tokens to account for unseen words and make all the sentences the same length and (4) assign to all tokens a random vector embedding. 

Or approach was fairly different and was centered on the pre-trained Glove embeddings from Stanford University. In fact our approach consisted of (1) Get the N most popular (in terms of frequency in the English language) words according to Glove and define our vocab and word embeddings. (2) Set the PAD tokens to 0 and the OOV tokens to the average of the unused embeddings on the Glove set and (3) tokenise the sentences and map the tokens to the embeddings obtained in the previous step.  

In order for our procedure to work, we required a tokenisation procedure that would at best match the Glove tokens, so that we minimize the amount of OOVs present our training and test sets. The initial tokenising function was way too simplistic, and some of the important steps we took to achieve this where the following:

 - separate punctuation from words
 - separate grammatical endings (e.g. "'s", "n't") from the base word
 - separate currency symbols and numbers
 - lowercase all words

One thing we noticed when using Glove was that it provided 50, 100, 200 and 300 dimensional embeddings. In order to provide us with more flexibility, and in an attempt to perhaps keep more of the information richness of higher dimensional embeddings while avoiding overfitting, we have implemented the option of using PCA on our embeddings. This allowed us to experiment with any arbitrary dimensionality in training our models while attempting to as much of the information present in the higher dimensional embeddings as possible.

Finally, it is interesting to note that in our approach we also attempted to use the training set to create a vocabulary (for consistency the initial approach given) and then create word embeddings using this vocabulary and Glove, but we noticed a smaller performance of our models.

### Sentence embeddings
In terms of sentence embeddings, the initial model summed all the word embeddings in the sentence to obtain its sentence embeddings. This simple approach does not rely on the training set. Although this approach eliminates the risk of overfitting, it has both an extremely high bias and a lack of any conceptual backing, which makes it less than ideal.

In contrast to this, we chose to use representation learning with Recurrent Neural networks in order to learn the sentence embeddings from our data. 

+ single direction RRNs
+ bidirection RNNs
+ Multilayer RNNs
+ attention
+ droupout in order to avoid overfitting

### Sentence Ordering

##### Sentence ordering via Classification
+ MLP with 2 hiddenl layers of x neurons and an output layer of 25 neurons
+ interpretaion 
+ cross-entrpy loss
+ Hungarian method

##### Sentence ordering via Regression
+ MLP with 2 hidden layers of x neurons and an output layer of 5 neurons
+ MSE
+ interpretaion


+ Early stopping in order to avoid overfitting


### Model Selection and Tunning
+ How we chose the architecture (splitting training/ dev and choosing by performance on dev set for now, maybe we can do something better)
 
+ How we chose all the model hyperparameters : early stopping, dropout, optimiser

+ What are the results of all these different  methods gave and what is our final model and performance

### Error Analysis 
We finally used our selected model and performed error analysis on it. Our purpose was to figure out if there are any particular types of errors that it still makes in an attempt to find out how we could address them.

Our approach started with creating the Confusion Matrix (CM) of our model on the dev set, which is shown as the output of the Fig. 1 cell below. This CM shows the correct classification for each sentence in the dev set vs the classification that was predicted by our model, the color coding showing how many sentences fall into each configuration. Diagonal values then represent correctly classified sentences. As we can see from Fig. 1, our model does a good job at classifying sentences that should go in positions 0 and 4, but drops for the middle positions.
 
In order to attempt to gain further insight on the mechanisms at play, we proceed by answering a series of questions:

##### Are OOVs or PADs the cause of error?
In Figs. 2-3 we have the distributions of the number of OOVs in misclassified sentences and all the sentences respectively and in Figs. 4-5 we have the distributions of the number of PAD tokens in misclassified sentences and all the sentences respectively. From these Figures we see that the distributions are very similar which invites the conclusion that the number of OOVs or PAD tokens in a sentence does not play a significant role in the likelihood of it being misclassified.

##### What words appear frequently in misclassified vs in all phrases?
In Fig. 6 are shown the 10 most frequent words in misclassified sentences, in Fig. 7 are shown the 10 most frequent words all the sentences, and in Fig. 8 are shown the 10 words that appear the most in misclassified sentences relative their total appearance. So for instance we can see that  ~75% of the phrases the word "trip" appears in are misclassified. We can see here that words that often appear in misclassified examples tend to either be very common such as "the" or have more than one meaning such as "trip".


##### For each sentence position in the initial paragraph, what is the likelihood of it being misclassified?
+ we dont have something for that yet

### Conclusion and Further Work
+ summary
+ what else we would have liked to try but run out of time