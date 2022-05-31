# HCR
Handwritten Character Recognition using Deep Learning
### DATASET
- [Link](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)#
### Here 2 files are submitted: 
- One is Fake News Detection by [NLP](https://github.com/Puranshu/FND_project/blob/main/FND_NLP_code.ipynb) techniques where we use StopWords, Tokenisation, Lemmatization etc. ([Course Project](https://drive.google.com/drive/folders/1ZBO2sUA6tarcFLq58T95YHoVmMZrnnUm?usp=sharing) with Akshit).

- In other one we used [Long-Short-Term-Memory](https://github.com/Puranshu/FND_project/blob/main/FND_LSTM_code.ipynb) (Neural Network) to achieve same agenda.

## Roadmap
### Natural Language Processing

- First we import and explore the dataset.

- After that we Preprocess the data (using RegEx, Tokenization, StopWords, Lemmatization).

- Then we convert words into matrix using BOW/Count Vectorizer and TF-iDF Vectorizer.

- After that we predict the accuracy of our model using Logistic Regression.

- At last we create and save a pipeline for future prediction.

### Long Short Term Memory 

- First we import the dataset.

- Then we remove StopWords and use Porterstemmer (to lower the words in root form).

- After that we apply onehot encoding technique to convert rooted words in numbers.

- Now we create the Sequential model with adam optimizer.

- At last we train our model and compute accuracy. 

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

## Acknowledgements

 - [Utsav Aggarwal](https://www.youtube.com/watch?v=xyq-zYr1cnI&t=2117s)
 - [Krish Naik](https://www.youtube.com/watch?v=MXPh_lMRwAI&t=1103s)

## Feedback

If you have any feedback, please reach out to me at puranshu@iitg.ac.in.

## License

[MIT](https://choosealicense.com/licenses/mit/)

  
