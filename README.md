# Puppy Party
### Kyle Fang, Charisse Hung, Adhvaith Vijay, Britney Zhao

For our PIC 16B project, we created a webapp to satisfy your dog breed needs. There are two main components to the app, a dog breed predictor, and a dog breed recommender.

**Link to our [webapp!](https://pic16b-dog-detector.herokuapp.com/)**

## Using the Dog Breed Predictor

Navigate to the **Predict Dog Breed** page. Upload any dog image from your computer or phone. The file type must be png, jpg, jpeg, or heic. Within seconds, the model will predict what breed the dog is. The top three breeds will be displayed.

After using the predictor, please fill out the feedback form that will appear so we can understand where our model is succeeding are where it needs improvement.

To learn more about the model, navigate to the **Model Architecture** page to see the structure of the convolutional neural network.

## Using the Dog Breed Recommender

Going to the **Find Your Perfect Dog** page brings you to our breed recommender page. Here, you can input a variety of characteristics you want in a dog using the sliders on the page. Then, after clicking submit, the model will display the top three dog breeds that match your wanted characteristics. If you want to try a different set of character traits, simply move the sliders and click submit again, as the model will automatically reset when the sliders are moved. 

## Group Contribution

**Adhvaith:** Built initial model with tensorflow and created/hosted web app to display everyone's work.

**Kyle:** Helped fine-tune model by introducing new breeds to classify. Additionally, web scraped for more data/dog photos to create a database of images we can query for in real-time.

**Britney:** Cleaning the dog attribute dataset, and used KDtree to predict the best dog breed based on user preferences!

**Charisse:** Worked on breed recommender with Britney. Focused on transforming the json file into a cleaned dataframe, feature selection, and a breed recommendation based on cosine similarity.

## Future steps
One suggestion to further improve our project is to include more dog breeds. Currently, the breed prediction model is trained on 121 breeds and the breed recommender includes 199 breeds. There are many more dog breeds that could be included in these two aspects of our project. Furthermore, our breed prediction model is trained on images of purebred dogs, and thus it does not perform as well on mixed breed dogs. If we could obtain or create a database of mixed breed dogs that included the list of breeds that each dog is, we could further train the model to predict the multiple breeds of a dog. This does pose many challenges since the amount of breed combinations is very great, however, it would allow our project to be more inclusive of dogs.
