# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models

# Project Overview & Purpose
The objective of this project was to use machine learning and neural networks to analyze the features provided in the dataset to create a binary classifier that is 
of predicting whether applicants will be successful if funded by Alphabet Soup. In addition, the neural and deep learning models implemented using TensorFlow were  optimized to improve the accuracy of the predictions. 


## Scope
* Implement neural network models using TensorFlow.
* Explain how different neural network structures change algorithm performance.
* Preprocess and construct datasets for neural network models.
* Compare the differences between neural network models and deep neural networks.
* Implement deep neural network models using TensorFlow.
* Save trained TensorFlow models for later use.


## Technologies 
* TensorFlow 2.x
* Google Colab
* Python and Pandas


# Results
The initial model produced an accuracy of 72%. After tuning the model by performing additional preprocessing of the data and tuning the hyper parameters of the deep learning model, the performance of the model improved to 78.8%. 

The specific dimensions considered in the analysis and tuning:

### Data Preprocessing
* The target variable(s) considered for your model:  The Is_Successful variable was considered the target for the model.

  target_variable.png![target_variable](https://user-images.githubusercontent.com/80140082/126168854-aa9c92ef-6916-4a2e-b799-f289ae8b4354.png)

* The variable(s) considered to be the features for your model: 

  features_variables.png![features_variables](https://user-images.githubusercontent.com/80140082/126168816-f84abd3c-8570-4b7b-b8d2-edb1cd062f56.png)
  
  features.png![features](https://user-images.githubusercontent.com/80140082/126170219-ed87b5fc-de28-4310-aa8b-e64682978c1a.png)

* Variable(s) that were neither targets nor features that were removed from the input data:  "EIN", "STATUS", "SPECIAL_CONSIDERATIONS"

  nonbeneficial_columns_dropped.png![nonbeneficial_columns_dropped](https://user-images.githubusercontent.com/80140082/126169290-b6ff3393-3bf4-43b9-801e-ef9517c90528.png)


### Compiling, Training, and Evaluating the Model
* Three hidden layers and two activation functions were selected for the neural network. 

  optimizing_model.png![optimizing_model](https://user-images.githubusercontent.com/80140082/126171731-ae377539-a0af-4903-8866-5cff3d63af39.png)

* The target performance was to be > 72%. With the optimization adjustments, the target model achieved a 78.8% accuracy. Were you able to achieve the target model performance?

  model_performance_results.png![model_performance_results](https://user-images.githubusercontent.com/80140082/126172075-b1b280fe-1140-4614-9332-383abfac28b3.png)


* The followinng steps were taken to try and increase model performance. The optimization entailed adjusting data to ensure that there are no variables or outliers that are causing confusion in the model, such as:

    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
    * Adding more neurons to a hidden layer.
    * Adding more hidden layers.
    * Using different activation functions for the hidden layers.
    * Adding or reducing the number of epochs to the training regimen.

# Summary
Overall, the model improved by dropping non-beneficial variables, binning the unique values and tuning multiple hyper parameters. 

## Recommendation
Additional tuning of the nodel could be performed by:

  * Utilizing different classifiers such as Random Forest and Multi Linear Regression.
  * Utilizingn the Keras tuner.
