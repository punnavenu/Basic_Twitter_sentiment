#!/usr/bin/env python
# coding: utf-8
import nltk
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')



import numpy as np
from nltk.corpus import twitter_samples
from Utilities_preprocesing import process_tweet, build_freqs

# ### Preparing the data
# * The `twitter_samples` contains subsets of 5k positive tweets, 5,000 negative tweets, and the full set of 10,000 tweets.
# Hence we will be splitting the data for train_set = 8k (4k pos and 4k neg) and test_set = 2k(1k pos and 1k neg)


# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# * Train test split: 20% will be in the test set, and 80% in the training set.


# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# * Create the numpy array of positive labels and negative labels.

# In[29]:


# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# In[30]:


# Print the shape train and test sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))




# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# In[32]:


# test the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

def sigmoid(z):
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    return h


# Testing your function
if (sigmoid(0) == 0.5):
    print('SUCCESS!')
else:
    print('Oops!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('CORRECT!')
else:
    print('Oops again!')

-1 * (1 - 0) * np.log(1 - 0.9999)  # loss is about 9.2

-1 * np.log(0.0001)  # loss is about 9.2

def gradientDescent(x, y, theta, alpha, num_iters):

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = len(x)

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        # z = np.dot(x,theta)
        z = x.dot(theta)
        print(z)

        # get the sigmoid of z
        h = sigmoid(z)
        print(h)

        # calculate the cost function
        # J = -1/m*(np.sum((np.dot(np.transpose(y),np.log(h)),(np.dot(1-np.transpose(y),np.log(1-h)))))
        J = (-1) / m * (y.transpose().dot(np.log(h)) + (1 - y).transpose().dot(np.log(1 - h)))

        # update the weights theta
        theta = theta - alpha / m * (x.transpose().dot(h - y))
        # theta -= alpha/m*(x.transpose().dot(h-y))

    ### END CODE HERE ###
    J = float(J)
    return J, theta

np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")



# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):

    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_l:
        if (word, 1) in freqs.keys():
            # increment the word count for the positive label 1
            x[0, 1] += freqs[(word, 1)]
        if (word, 0) in freqs.keys():
            # increment the word count for the negative label 0
            x[0, 2] += freqs[(word, 0)]
    assert (x.shape == (1, 3))
    return x

# test on training data
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

# check for when the words are not in the freqs dictionary
tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)

# In[80]:


# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(x.dot(theta))


    return y_pred

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great',
              'great great great', 'great great great great']:
    print('%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))

my_tweet = 'I am learning :) :( ;( ;( :( '
predict_tweet(my_tweet, freqs, theta)

def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = sum(sum(np.array(y_hat) == np.array(test_y.transpose()))) / len(y_hat)

    ### END CODE HERE ###

    return accuracy


# In[99]:


tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

print('Label Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

my_tweet = 'This is a flop movie. The plot was amazing but the ending was sad!'
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')




