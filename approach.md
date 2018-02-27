### Overview
In this task of finding the sentence order in a paragraph, our approach was broken down into the following tasks:

+ Tokenising, making a vocabulary  and creating word embeddings.
+ Creating sentence embeddings using the word embeddings.
+ Making sentence order predictions based on the sentence embeddings.

In the following sections, we break down how we tackled each of these components in detail.

### Word embeddings and Tokenisation
The initial model provided had a fairly simple approach to data preprocessing and tokenisation. The steps followed where (1) the sentences are tokenised into words by splitting them on whitespaces, (2) make a vocabulary (that maps tokens to unique integers) out of the tokens found in the training set, (3) add OOV and PAD tokens to account for unseen words and make all the sentences the same length and (4) assign to all tokens a random vector embedding.

Our approach was fairly different and was centered on the pre-trained Glove embeddings. In fact our approach consisted of (1) get the N most popular (in terms of frequency in the English language) words according to Glove and define our vocab and word embeddings, (2) set the PAD tokens to 0 and the OOV tokens to the average of the unused embeddings on the Glove set and (3) tokenise the sentences and map the tokens to the embeddings obtained in the previous step.

We required a tokenisation procedure that would at best match the Glove tokens, so that we minimize the amount of OOVs present in our training and test sets. The initial tokenising function was too simplistic, and some of the important steps we took to achieve this were the following:

 - separate punctuation from words
 - separate grammatical endings (e.g. "'s", "n't") from the base word
 - separate currency symbols and numbers and common abbreviations (such as 'e.g.', 'Mr.')
 - lowercase all words

Glove provides 50, 100, 200 and 300 dimensional embeddings. In order to provide us with more flexibility, and in an attempt to perhaps keep more of the information richness of higher dimensional embeddings while avoiding overfitting, we have implemented the option of using PCA on our embeddings. This allowed us to experiment with any arbitrary dimensionality in training our models while attempting to retain as much of the information present in the higher dimensional embeddings as possible.

Finally, it is interesting to note that in our approach we also attempted to use the training set to create a vocabulary (for consistency the initial approach given) and then create word embeddings using this vocabulary and Glove, but we noticed a smaller performance of our models.

### Sentence embeddings
In terms of sentence embeddings, the initial model summed all the word embeddings in the sentence to obtain its sentence embeddings. Although this approach eliminates the risk of overfitting, it has both an extremely high bias and a lack of any conceptual backing.

In contrast to this, we chose to use Recurrent Neural Networks (RNN) to take into account the order of the sentences. More precisely, we have tried the following basic architectures, all of which which take as an input the word-embedded sentences and output the sentence embeddings:

+ A Simple LSTM.
+ A Bi-directional LSTM.
+ A two layer LSTM.
+ A two layer Bi-directional LSTM.

On top of these architectures, we have also implemented the following variations in an attempt to maximise our performance:
+ Self Attention (word by word attention based on [Zhouhan Lin et al's paper : A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf) )
+ 1D convolutional layers on the word embeddings

Finally, in order to combat overfitting, we have implemented the following techniques:

+ Randomizing the order of the batches
+ Dropout
+ Early Stopping
+ Data Augmentation i.e. shuffling the sentences in the training set before each epoch, so it is highly unlikely that the model sees any training data point twice
+ Regularisation (L2)

### Sentence Ordering
Once the sentences are embedded we need a way to predict a sentence order. It is possible to do this in two ways: Regression and Classification. In Classification, the model will output a (5,5) matrix where the rows are the sentences and the columns are the probability of being in a position. In Regression the model will output the positions of the sentences directly.

While the initial solution just treated Classification through a linear regressor, we are usign non-linear Multi-layer perceptrons (MLP) and attempting both regression and classification. Details of our approach for these two techniques are described below:

##### Sentence ordering via Classification
For classificaiton we implemented an MLP with 2 hidden layers with Relu activations and a 25-neuron softmax output layer. To train the model, we used cross-entropy loss and the RMS-prop optimiser (which we found to be better than other options such as Adam).

The last challenge we had to face is how to obtain the final ordering from our (5,5) output. The initial model used a greedy approach but this resulted in the same position being assigned a sentence more than once. We wanted to avoid this while maximising the probability of the whole paragraph. We identified this as an instance of the Maximum Weight Assignment problem (considering a bipartite graph between the sentences and the positions) and hence used the Hungarian method in order to solve it.

##### Sentence ordering via Regression
For regression, we again used an MLP with 2 hidden layers, but we now have a 5-neuron linear output layer instead. In order to train it, we used the Mean Squarred Error (MSE) loss.

### Model Selection and Tunning
In order to select our final model we had to compromise between  getting a model that achieves good generalisation performance and catering to the limited computational resources we had our disposal. We decided to use the dev set for validation for our models, and proceed in three times : (1) Select the dimentionality of the word embeddings (2) Select whether to use Classification or Regression, (3) Select a base architecture form the list above and (4) Tune this base architecture to get the highest possible score.

After establishing that using 200D data was the best setup that kept us from running into memory issues, we experimented  with the regression and classification styles. It became clear that  using the classification method gave better performance, hence we proceeded with this technique. Then, keeping all hyperparameters equal, we trained all the base architectures listed, which gave the results listed below:

| Model         | Accuracy on the validation set |
| ------------- |:------------------:|
| LSTM |   57.3276    |
| Bi-directional LSTM    |57.7552      |
| Two layer LSTM   |  57.6269         |
| Two layer Bi-directional LSTM | 57.9583          |

As we can see, the model that gave the best performance was the the two-layer bi-directional LSTM, so we chose it to proceed.

We tuned this architecture by experimenting with (1) the number of hidden layer neurons, (2) the LSTM hindden neurons, (3) adding self-attention, (4) adding a 1D CNN layer.

In the conclusion of our tuning, and although the inclusion of attention and an additional convolutional layer looked promising, we finally proceeded with the simple Two layer Bi-directional LSTM due to computational resource limitations.


### Error Analysis
We finally used our selected model and performed error analysis on it. Our purpose was to figure out if there are any particular types of errors that it still makes in an attempt to find out how we could address them.

Our approach started with creating the Confusion Matrix (CM) of our model on the dev set, which is shown as the output of the Fig. 1 cell below. This CM shows the correct classification for each sentence in the dev set vs the classification that was predicted by our model, the color coding showing how many sentences fall into each configuration. Diagonal values then represent correctly classified sentences. As we can see from Fig. 1, our model does a good job at classifying sentences that should go in positions 0 and 4, but drops for the middle positions.

In order to attempt to gain further insight on the mechanisms at play, we proceed by answering a series of questions using the dev set:
##### Are OOVs or PADs the cause of error?
In Figs. 2-3 we have the distributions of the number of OOVs in misclassified sentences and all the sentences respectively and in Figs. 4-5 we have the distributions of the number of PAD tokens in misclassified sentences and all the sentences respectively. From these Figures we see that the distributions are very similar which invites the conclusion that the number of OOVs or PAD tokens in a sentence does not play a significant role in the likelihood of it being misclassified.

##### What words appear frequently in misclassified vs in all phrases?
In Fig. 6 are shown the 10 most frequent words in misclassified sentences, in Fig. 7 are shown the 10 most frequent words all the sentences, and in Fig. 8 are shown the 10 words that appear the most in misclassified sentences relative their total appearance. So for instance we can see that  ~70% of the phrases the word "trip" appears in are misclassified. We can see here that words that often appear in misclassified examples tend to either be very common such as "the" or have more than one meaning such as "trip".


##### For each sentence position in the initial paragraph, what is the likelihood of it being misclassified?
In Figs. 9 to 13 are shown the confusion matrices for sentences 1 to 5 respectively. We can see that the mistakes of our model varies depending on which sentence it tries to classify. Nonetheless, a very apparent trend is that for all sentences, whenever their true position is at the start of the paragraph, the model overwhelmingly predicts this correctly.

### Conclusions
In our approach, we experimented with various architectures of neural networks in order to solve the sentence ordering problem. Our best performing model used 200D Glove embeddings, two-layer bi-directional LSTMs and an MLP performing the classification task. We believe that given more resources our models would have achieved classifications comfortably above 60%, and hence are satisfied with the results of our project.