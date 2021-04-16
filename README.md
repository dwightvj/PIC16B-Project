# PIC16B-Project

**Planned Deliverables**

As stated previously, we would have a fully functional web app. The user would be able to upload an image to the web app, our model would process the image, and the web app would provide top matches and information about those matches.
Full success would include the image recognition model, a program that can scrape information about the animal and provide links, and the fully functional web app.
Partial success would include just the image recognition model. We will start with that, and if things go to plan, we will have time to work on the web scraping and the webapp after.

**Resources Required**

- [Animals with Attributes](https://cvml.ist.ac.at/AwA2/). This contains  37322 images of 50 animals classes. We can use this database to learn how to create a model that predicts the animal from an image.
- [Stanford Dog’s Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). This dataset contains images of 120 dog breeds and at least 150 images per breed for a total of 20,580 images. Once we have an understanding of how to predict animals from images, we can use this dataset to train a more specific model that can predict dog breeds.
- [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset contains images of 25 dog breeds and 12 cat breeds with roughly 200 images per breed for a total of 7349 images. The dog images could be used in conjunction with the Stanford Dog’s Dataset to increase the training data or test a working model.

There are additional datasets online of animal images and some are very broad containing many animals whereas others are more specific with many breeds of the same animal. Therefore, we do not think we will have trouble finding additional images for our project if necessary. 

**Tools/Skills Required**

- We will need to use computer vision since we will be dealing with images, so *OpenCV* and *Scikit-image* will be useful. 
- To create a machine learning model to identify the animals, we can use *Tensorflow* and *Keras*. For data manipulation we can use *Pandas* and *Numpy* and to create plots to evaluate our model we can use *Matplotlib*. 
- To scrape information about animals off the internet we can use *Scrapy*.
- We will use a combination of *Dash* and *Heroku* in order to deploy our work to the cloud. We intend to create an interactive web application that is capable of handling user data (i.e. jpg/png) and generating a result (likely in the form of the ‘top x’ predicted classes)

**Risks**

1. Poor accuracy
    * Image recognition is a field we are new to so there may be a learning curve that could result in a less-than-ideal solution
    * For this reason we are trying to predict the ‘top x’  predicted classes rather than a cut-and-dry class that an animal might fall under
2. Deployment
    * Deploying our work on a web app comes with a few nuances, and we may run into problems concerning storage
3. Learning Curves
    * We will likely need to delve into PyTorch for the purposes of image classification

Our biggest concern is bad accuracy with our image recognition. Particularly if we were to focus on dog breeds, many dogs look familiar, which could result in false readings. As mentioned earlier, it is impossible to get 100% accuracy with a task like this. To counteract this issue, we would give more possibilities for the animal in the photo. In this case, we would give the top 5 possibilities so that the user can receive more information in case the first choice animal is not correct.

Another risk with our project is that we have not found a dataset that contains every possible dog breed or animal type. Specifically with dog breeds, there is an infinite amount of mixed-breed dogs. The datasets we have found focus mainly on pure-bred dogs, which would also skew the analysis of the image recognition. There are also millions of animal species that we would not be able to account for. Therefore, not only can we not guarantee full accuracy, we cannot tailor the app to all possibilities. To address this risk, we would give multiple options for animals and information about each option so that we can at least guide the user towards the correct animal. In this way, the user would not receive the correct answer each time our web app is used, but rather our web app would teach the user about animals that look similar so they can get closer to the correct species or breed.

**Ethics**

Because we have not found a dataset encompassing all possible dog breeds or animal species, there would be an inherent bias against the less common breeds or species. Since these data sets focus on what would most likely be seen, they tend to focus on more popular dog breeds/animal species. Therefore, since our project would use data sets that do not include more rare types of animals, these species would not be accounted for. However, since our plan is to show a list of possible species, we would hope that this would influence the user to seek more information about similar animals in physical appearance.



**Tentative Timeline**

By week 6, we hope to have the image recognition model working. By week 8, we hope to have the web scraping and facts/resource/link collection program working. By week 10, we hope to put everything together onto the web app and add final features.

