# HCR
Handwritten Character Recognition using Deep Learning
### DATASET
- [IAM Dataset](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

## Roadmap

- Data: First we will import the dataset from the given link and navigate to the “words” folder as we are only focusing on training words only. Along with the words folder we will also have a words.txt file, which will contain information regarding folder structure. This text fill will also have labels associated along with the path of the picture. There are a total of 96,456 words that are written and their respective labels are in words.txt file. For our model we will be splitting 96,456 into train dataset, test dataset and validation dataset. For training 90% 86,810, 5% for validation and test each 4,823.
- Preprocessing: We begin by preparing the image paths for our data input pipeline. Then we concat the image path with its corresponding label. We achieve this by the image path which we have in our words folder and use that path to find its corresponding label in words.txt file. This preparation of image path and respective labels is our preliminary processing part. After cleaning the images (essentially the path) and corresponding labels are fed to our model.
- String lookup layer is also used as we know we cannot feed character to our model, we are required to convert these characters to numbers. Vice versa is also required because the outcome will also be a number and to understand that we need to convert this numeric output to character.
- As an initial preprocessing task we convert all images to the same size such that their content is intact and aspect ratio is also not disturbed. This resizing of image without any significant distortion is carried out by padding on image. First we calculate the total required padding and do that on both ends of the image. This is different from simply resizing the image.
- After putting everything together we have an image size of 128x32 and a batch size of 64. In the processing part we read the file and decode it unit8 tensor. After that we vectorize the labels character to numeric with input encoding as UTF-8. Next we prepare the final dataset with image paths and corresponding cleaned labels send these paths and their labels to the model. As for the last layer, we use CTC loss.
- Model and its parameters:After the input layer we have 2 convolutional and 2 maxpool layers each having pool size and stride of 2. With that, the overall size is decreased by a factor of 4. Following we reshape the image and add a dense layer with 64 nodes and ReLu activation function (where negative input is converted to 0 and positive input remains as it is) and dropout of 0.2.
Subsequently we add 2 bidirectional LSTM each with a dropout of 0.25. As for the end layer we have CTC loss which will calculate loss after each step/epoch. This whole model uses Adam optimizer, we can also use different optimizers (like mini batch gradient descent, RMSProp ) and check the results. We used ADAM because of its fast convergence and it can solve the problem of vanishing learning rate. As for evaluation purposes we use Edit distance.
After setting the layers and their parameters, the model is then trained on different epochs and with each epoch we calculate the mean edit distance. Loss for training and validation data is also computed.


## Documentation

- [Regular Expression (RegEx)](https://docs.python.org/3/library/re.html)

- [Tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)

- [StopWords](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)

- [Lemmatization](https://pythonprogramming.net/lemmatizing-nltk-tutorial/)

- [Bag of words/Count vectorizer](https://www.mygreatlearning.com/blog/bag-of-words/)

- [TF-iDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

- [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

- [Stemming](https://www.nltk.org/howto/stem.html)

- [One Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

- [Padding](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)

- [Sequential model](https://keras.io/guides/sequential_model/)

- [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)


## Feedback

If you have any feedback, please reach out to me at puranshu@iitg.ac.in.

## License

[MIT](https://choosealicense.com/licenses/mit/)

  
