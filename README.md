# Puppy Party
### Kyle Fang, Charisse Hung, Adhvaith Vijay, Britney Zhao

For our PIC 16B project, we created a webapp to satisfy your dog breed needs. There are two main components to the app, a dog breed predictor, and a dog breed recommender.

Link to our [webapp!](https://pic16b-dog-detector.herokuapp.com/)

## Using the Dog Breed Predictor

Navigate to the **Predict Dog Breed** page. Upload any dog image from your computer or phone. The file type must be png, jpg, jpeg, or heic. Within seconds, the model will predict what breed the dog is. The top three breeds will be displayed.

After using the predictor, please fill out the feedback form that will appear so we can understand where our model is succeeding are where it needs improvement.

To learn more about the model, navigate to the **Model Architecture** page to see the structure of the convolutional neural network.

## Using the Dog Breed Recommender

Going to the **Find Your Perfect Dog** page brings you to our breed recommender page. Here, you can input a variety of characteristics you want in a dog using the sliders on the page. Then, after clicking submit, the model will display the top three dog breeds that match your wanted characteristics. If you want to try a different set of character traits, simply move the sliders and click submit again, as the model will automatically reset when the sliders are moved. 

## Group Contribution

**Adhvaith:** Built initial model with tensorflow and created/hosted web app to display everyone's work.

**Kyle:** Helped fine-tune model by introducing new breeds to classify. Additionally, web scraped for more data/dog photos to create a database of images we can query for in real-time.

**Britney:** cleaning the dog attribute dataset, and used KDtree to predict the best dog breed based on user preferences!

**Charisse:** Worked on breed recommender with Britney. Focused on transforming the json file into a cleaned dataframe, feature selection, and a breed recommendation based on cosine similarity.
