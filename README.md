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
- Model and its parameters: After the input layer we have 2 convolutional and 2 maxpool layers each having pool size and stride of 2. With that, the overall size is decreased by a factor of 4. Following we reshape the image and add a dense layer with 64 nodes and ReLu activation function (where negative input is converted to 0 and positive input remains as it is) and dropout of 0.2.
Subsequently we add 2 bidirectional LSTM each with a dropout of 0.25. As for the end layer we have CTC loss which will calculate loss after each step/epoch. This whole model uses Adam optimizer, we can also use different optimizers (like mini batch gradient descent, RMSProp ) and check the results. We used ADAM because of its fast convergence and it can solve the problem of vanishing learning rate. As for evaluation purposes we use Edit distance.
After setting the layers and their parameters, the model is then trained on different epochs and with each epoch we calculate the mean edit distance. Loss for training and validation data is also computed.

## Conclusion
- We discussed a RNN based deep learning model for Words recognition from the IAM dataset. To carry out this, we used bidirectional LSTM. With a resized image of 128x32, and labelled as well we try to train our model on different epochs and as epochs are increased we can see significant changes in the performance. Here we witnessed that as we increase epochs from 10 to 50, we can see slight reduction in MED (17.8 to 17.35), however loss on training and validation are reduced significantly from 3.6 to 1.67 and 2.9 to 1.77 respectively. This process can further be optimised by use of more nodes in layers and by use of different activation functions. It is generally observed that if we use more nodes in layers, we get better results. However, while doing so we should also avoid overfitting.

## Documentation

- [Bi-LSTM layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional)

- [CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

- [Edit Distance](https://edit-distance.readthedocs.io/en/latest/)

- [ADAM optimizer](https://keras.io/api/optimizers/adam/)


## Feedback

If you have any feedback, please reach out to me at puranshu@iitg.ac.in.

## License

[MIT](https://choosealicense.com/licenses/mit/)

  
