# %% [markdown]
# # Building a model to predict readability scores using the linear regression with and without cross-validation
#  
# In earlier weeks, we discussed how to process text data and generate 768 numerical features for a given text using a pre-trained NLP model ([see Lecture 2b](https://edld654-fall22.netlify.app/lecture-2b.html) and [Kaggle Notebook 2b](https://www.kaggle.com/code/uocoeeds/lecture-2b-data-preprocessing-ii/notebook). These features were exported as a separate dataset in [Kaggle Notebook 2b](https://www.kaggle.com/code/uocoeeds/lecture-2b-data-preprocessing-ii/notebook), and imported to this notebook.
# 
# First, we will import this previously created matrix of numerical features.
# 
# This dataset has 2834 rows and 769 columns. Each row represents a reading passage. The first 768 columns are the sentence embeddings previously generated for each reading passage using the [Longformer model](https://huggingface.co/allenai/longformer-base-4096) as potential predictor variables.The last column is the target readability score, the outcome variable to predict (`target`).

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:11.952108Z","iopub.execute_input":"2022-10-14T19:09:11.954119Z","iopub.status.idle":"2022-10-14T19:09:23.985821Z"}}

readability <- read.csv('../input/lecture-2b-data-preprocessing-ii/readability_features.csv',
                        header=TRUE)

dim(readability)

colnames(readability)

# %% [markdown]
# # 1. Do It Yourself: Fitting linear regression without cross-validation
# 
# In this section, we will fit the linear regression model without cross validation using the `lm` function.

# %% [markdown]
# ## 1.1. Initial Data Preparation
# 
# We will first do some initial exploration of the variables. First, we can look at the percentage of missing values. Particularly, We can look for any feature with more than 80% of values are missing. Then, we can remove those features from the data.
# 
# *Note. For this dataset, it is just a formality because we know there is no missing data as the pre-trained NLP model generated 768 numerical features for any text.*
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T21:40:38.589455Z","iopub.execute_input":"2022-10-06T21:40:38.591136Z","iopub.status.idle":"2022-10-06T21:40:47.8008Z"}}
require(finalfit)

missing_ <- ff_glimpse(readability)$Continuous

head(missing_)

# %% [markdown]
# Because there are 768, it is not practical to print them all in this notebook. In case of data with missing values, Iyou filter the ones with missing data, and then print those. Finally, you can remove the flagged variables with more than 80% missingness, or impute them.

# %% [code]
#flag_na <- which(as.numeric(missing_$missing_percent) > 80)
#flag_na

#readability <- readability[,-flag_na]

# %% [markdown]
# ## 1.2. Train/Test Split
# 
# In order to obtain a realistic measure of model performance, we will split the data into two subsamples: training and test datasets. Due to the relatively small sample size, I will use a 90-10 split (typically a 80-20 or 70-30 split is used). The smaller test dataset will be used as a final hold-out set, and training dataset will be used to build the model.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:31.549606Z","iopub.execute_input":"2022-10-14T19:09:31.58108Z","iopub.status.idle":"2022-10-14T19:09:31.815748Z"}}
set.seed(10152021)  # for reproducibility
  
loc      <- sample(1:nrow(readability), round(nrow(readability) * 0.9))
read_tr  <- readability[loc, ]
read_te  <- readability[-loc, ]

dim(read_tr)

dim(read_te)

# %% [markdown]
# ## 1.3. Preparing the blueprint and processing the variables
#   
# We will use the recipes package to create a recipe to process the variables in the dataset. Note that all my features are numeric, and the last column is outcome variable while every other column is a predictor variable. This recipe will 
# 
# - assign the last column (target) as outcome and everything else as predictors,
# - remove any variable with zero variance or near-zero variance,
# - and standardize all variables.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:35.016161Z","iopub.execute_input":"2022-10-14T19:09:35.01762Z","iopub.status.idle":"2022-10-14T19:09:37.341702Z"}}
require(recipes)

blueprint <- recipe(x     = readability,
                    vars  = colnames(readability),
                    roles = c(rep('predictor',768),'outcome')) %>%
             step_zv(all_numeric()) %>%
             step_nzv(all_numeric()) %>%
             step_normalize(all_numeric_predictors())

# %% [markdown]
# We will first train the blueprint using the training dataset, and then bake it for both training and test datasets.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:41.11381Z","iopub.execute_input":"2022-10-14T19:09:41.115748Z","iopub.status.idle":"2022-10-14T19:09:51.868831Z"}}
prepare <- prep(blueprint, 
                training = read_tr)
prepare

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:51.872315Z","iopub.execute_input":"2022-10-14T19:09:51.87398Z","iopub.status.idle":"2022-10-14T19:09:52.216599Z"}}
baked_tr <- bake(prepare, new_data = read_tr)

baked_te <- bake(prepare, new_data = read_te)

dim(baked_tr)

dim(baked_te)

# %% [markdown]
# ## 1.4. Fit the linear regression model (no cross-validation)
# 
# First, we will fit the model to the training dataset using all predictors in the dataset without any cross validation. Note that we will very likely overfit with 768 predictors and relatively small sample size. After fitting the model using the `lm()` function, I will extract the R-squared as a performance metrix.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:09:58.969126Z","iopub.execute_input":"2022-10-14T19:09:58.970717Z","iopub.status.idle":"2022-10-14T19:10:00.229915Z"}}
mod <- lm(target ~ .,data=baked_tr)

summary(mod)$r.squared

# %% [markdown]
# In the training dataset, the model explains about 84% of the total variance in the outcome variable (WOW!). We can also calculate the MAE, MSE, and RMSE for the model predictions in the training dataset.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:10:03.261903Z","iopub.execute_input":"2022-10-14T19:10:03.263343Z","iopub.status.idle":"2022-10-14T19:10:03.404562Z"}}
predicted_tr <- predict(mod)

rsq_tr <- cor(baked_tr$target,predicted_tr)^2
rsq_tr

mae_tr <- mean(abs(baked_tr$target - predicted_tr))
mae_tr

rmse_tr <- sqrt(mean((baked_tr$target - predicted_tr)^2))
rmse_tr

# %% [markdown]
# You can check the high correlation between observed values and predicted values.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:10:06.714675Z","iopub.execute_input":"2022-10-14T19:10:06.716122Z","iopub.status.idle":"2022-10-14T19:10:07.732932Z"}}
require(ggplot2)

ggplot()+
  geom_point(aes(y=baked_tr$target,x=predicted_tr))+
  xlab('Model Predictions')+
  ylab('Observed Readability Scores')+
  theme_bw()+
  ggtitle('Model Performance in the Training Dataset')

# %% [markdown]
# It is too good to be true! As we suspected, the model predictions are unusually good in the training data because we are fitting a super complex model, and we are overfitting. This is why you should never judge how well a model is by looking at the performance of the model on the dataset it is trained. 
# 
# This is like giving 15 questions to a number of students for studying and then giving them an exam with the exact same 15 questions. It would be no surprising that students would do extremely well in that exam because they have seen and studied the questions before.
# 
# When you fit a model using a dataset, and then use the same dataset to measure the model performance, the same thing happens. We will overestimate the model's predictive power on unseen datasets.
# 
# Let's check how well this model does on the test data which we didn't use in the estimation.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:10:11.56592Z","iopub.execute_input":"2022-10-14T19:10:11.567784Z","iopub.status.idle":"2022-10-14T19:10:11.695019Z"}}
# first obtain the predictions according to the model for the observations
# in the test dataset

predicted_te <- predict(mod,newdata=baked_te)

# Calculate the outcome metrics

rsq_te <- cor(baked_te$target,predicted_te)^2
rsq_te

mae_te <- mean(abs(baked_te$target - predicted_te))
mae_te

mse_te <- mean((baked_te$target - predicted_te)^2)
mse_te

rmse_te <- sqrt(mean((baked_te$target - predicted_te)^2))
rmse_te

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:10:16.473583Z","iopub.execute_input":"2022-10-14T19:10:16.475105Z","iopub.status.idle":"2022-10-14T19:10:16.760257Z"}}
ggplot()+
  geom_point(aes(y=baked_te$target,x=predicted_te))+
  xlab('Model Predictions')+
  ylab('Observed Readability Scores')+
  theme_bw()+
  ggtitle('Model Performance in the Test Dataset')

# %% [markdown]
# Compare the model's performance on the training set and test set. Where do you see the difference?
# 
# The model performance significantly dropped in the testing dataset. This is a classic example of model variance (overfitting). We have a very complex model that does a great job in the training dataset but does not perform at the same level in a different dataset. In other words, the specific noise in the training set that has been used as information when estimating the model parameters does not generalize to a different dataset. If we are planning to use this model for any future prediction, it is much better to consider the performance on the test data as it will be a more realistic picture of model performance. 

# %% [markdown]
# # 2. Do It Yourself: Fitting linear regression with 10-fold cross-validation
# 
# One way of obtaining realistic performance values while we train the dataset is to use k-fold cross validation. The code below first creates 10 folds for the training dataset. Then, it fits the model using the nine folds while it evaluates the performance on the tenth fold.
# 
# We will do this first by coding everything ourselves with a `for` loop. In the next section, we will do it using the `caret` package for a more user-friendly experience.

# %% [markdown]
# ## 2.1. Create vectors of indices for each fold
# 
# We will create a vector of indices to assign each observation to a fold. Here is the basic idea. Suppose you have a dataset with 20 observations, and you want to create 10 folds. Then, you can create a vector of indices as the following:
# 
#     {1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10}
# 
# This indicates that the first two observations are assigned to Fold 1, the third and fourth observations are assigned to Fold 2, and the fifth and sixth observation is assigned to Fold 3, and so on.
# 
# It is always a good idea to shuffle the row prior to assigning observations to each fold in order to introduce some randomness (in case there is an already existing pattern in terms of how rows are ordered).
# 

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:10:44.259927Z","iopub.execute_input":"2022-10-14T19:10:44.262388Z","iopub.status.idle":"2022-10-14T19:10:44.318044Z"}}
set.seed(10152021)  # for reproducibility of shuffling

# Randomly shuffle the data

    baked_tr = baked_tr[sample(nrow(baked_tr)),]

# Create the indices to assign each observation to 10 folds with (approximately) equal size

folds = cut(seq(1,nrow(baked_tr)),breaks=10,labels=FALSE)

folds

table(folds)

# %% [markdown]
# ## 2.2. Fitting the models to k-folds
# 
# In this section, we will implement the k-fold cross validation. This is how the algorithm should work:
# 
# - 1. Exclude Fold 1 from the dataset (using the indices we created before) and assign it as the temporary test set
# 
# - 2. Keep the data from the remaining 9 folds (Fold 2, Fold 3, ..., Fold 10) and assign it as the temporary training set 
# 
# - 3. Fit the model to the temporary training set
# 
# - 4. Predict the outcome on the temporary test set using the model, compute the performance metrics, store them for future evaluation.
# 
# - 5. Repeat Step 1 - 4 for all folds by replacing Fold 1 with another fold (Fold 2, Fold 3, ... Fold 10).
# 
# The code below implements this procedure.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:20:00.897646Z","iopub.execute_input":"2022-10-14T19:20:00.899597Z","iopub.status.idle":"2022-10-14T19:20:09.078146Z"}}
# Create empty vectors for performance measures

    rsq  <- c()
    mae  <- c()
    mse  <- c()
    rmse <- c()

# Fit the model by excluding one of the folds, and then evaluate the performance
# on the excluded fold

    for(i in 1:10){

      data_tr <- baked_tr[which(folds!=i),] # observation from the remaining 9 folds after excluding Fold i
                                            # temporary training set

      data_te <- baked_tr[which(folds==i),] # observation from Fold i, temporary training set  

      mod  <- lm(target ~ .,data=data_tr)   # Fit the model to the temporary training set

      pred <- predict(mod,newdata=data_te)  # Predict the outcome for the temporary test set

      rsq[i]  <- cor(data_te$target,pred)^2             # Compute R-square and save the value
      mae[i]  <- mean(abs(data_te$target - pred))       # Compute the Mean Absolute Error and save the value
      rmse[i] <- sqrt(mean((data_te$target - pred)^2))  # Compute the Root Mean Squared Error and save the value

      #cat(paste0('Fold ',i,' is completed.'),'\n')
    }

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:20:48.54879Z","iopub.execute_input":"2022-10-14T19:20:48.554091Z","iopub.status.idle":"2022-10-14T19:20:48.598615Z"}}
# R-squared obtained across 10 folds

    round(rsq,3)

# the average R-squared across folds

    round(mean(rsq),3)

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:21:14.299859Z","iopub.execute_input":"2022-10-14T19:21:14.304723Z","iopub.status.idle":"2022-10-14T19:21:14.326204Z"}}
# RMSE obtained across 10 folds

    round(rmse,3)

# the average RMSE across folds

    round(mean(rmse),3)

# %% [markdown]
# The performance evaluations we obtain from k-fold cross validation is more similar to the one we get from the test data, so they provide a more realistic picture of model performance. We will frequently use k-fold cross-validation for tuning the hyperparameters for several models in later classes. 
# 
# # 3. Model Fitting Using the `caret` package
# 
# It is not always the most pleasant experience to write your code to conduct k-fold cross-validation. Packages like `caret` provide built-in functions for conducting cross-validation and also bring several user-friendly experiences in modeling. `caret` provides a standardized user experience for fitting many different models beyond linear regression. So, one doesn't have to learn the nuances of all different types of functions to fit different types of models. Packages like `caret` provide a more consistent workflow while working with different types of models. On the other hand, this also brings less flexibility. During this class, I will try to demonstrate both how to work with direct functions and how to work with `caret` for fitting different types of models.
# 
# Below is how one could implement the whole process using the `caret` package.

# %% [markdown]
# ## 3.1. Train/Test Split

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:24:48.910876Z","iopub.execute_input":"2022-10-14T19:24:48.912451Z","iopub.status.idle":"2022-10-14T19:24:49.005321Z"}}
# Initial data preparation

require(caret)
require(recipes)

set.seed(10152021)  # for reproducibility

# Train/Test Split
  
loc      <- sample(1:nrow(readability), round(nrow(readability) * 0.9))
read_tr  <- readability[loc, ]
read_te  <- readability[-loc, ]

# %% [markdown]
# ## 3.2. Blueprint to process variables using the recipes package

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:24:53.707999Z","iopub.execute_input":"2022-10-14T19:24:53.709549Z","iopub.status.idle":"2022-10-14T19:24:53.745786Z"}}
# Blueprint

blueprint <- recipe(x     = readability,
                    vars  = colnames(readability),
                    roles = c(rep('predictor',768),'outcome')) %>%
             step_zv(all_numeric()) %>%
             step_nzv(all_numeric()) %>%
             step_normalize(all_numeric_predictors())

# %% [markdown]
# ## 3.3. Cross-validation Settings
# 
# We will create the index values for 10-folds to provide to the `trainControl` function. This way, you can reproduce the results in the future and use the same folds across different models.
# 
# While there are so many different ways of creating folds for cross-validation, here is my approach (similar to how we did it in Section 2).
# 
# - Randomly shuffle the training data
# 
# - Assign numbers from 1 to 10 in order to the observations
# 
# - Create a list object to store the indices for each fold
# 
# - Create an object using the `trainControl` function to store cross-validation settings
#    

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:24:58.539231Z","iopub.execute_input":"2022-10-14T19:24:58.540751Z","iopub.status.idle":"2022-10-14T19:24:58.610113Z"}}
# Randomly shuffle the data

    read_tr = read_tr[sample(nrow(read_tr)),]

# Create 10 folds with equal size

    folds = cut(seq(1,nrow(read_tr)),breaks=10,labels=FALSE)
  
# Create the list for each fold 
      
    my.indices <- vector('list',10)

    for(i in 1:10){
        my.indices[[i]] <- which(folds!=i)
    }
      
cv <- trainControl(method = "cv",
                   index  = my.indices)

# %% [markdown]
# ## 3.4. Train the model
# 
# Note that I only provide the blueprint and original unprocessed training dataset as input. The `caret::train()` will internally process the variables according to the blueprint before fitting the model.
# 
# This is roughly how the algorithm works when running this code:
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 1 from dataset, and then test the model performance on Fold 1 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 2 from dataset, and then test the model performance on Fold 2 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 3 from dataset, and then test the model performance on Fold 3 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 4 from dataset, and then test the model performance on Fold 4 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 5 from dataset, and then test the model performance on Fold 5 using a performance metric (e.g., RMSE)
#    
#     - Fit the regression model to remaining 9 folds after excluding Fold 6 from dataset, and then test the model performance on Fold 6 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 7 from dataset, and then test the model performance on Fold 7 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 8 from dataset, and then test the model performance on Fold 8 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 9 from dataset, and then test the model performance on Fold 9 using a performance metric (e.g., RMSE)
#     
#     - Fit the regression model to remaining 9 folds after excluding Fold 10 from dataset, and then test the model performance on Fold 10 using a performance metric (e.g., RMSE)
#     
#     - Compute the performance metrics from all 10 folds

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:25:02.731362Z","iopub.execute_input":"2022-10-14T19:25:02.732902Z","iopub.status.idle":"2022-10-14T19:27:19.872754Z"}}
caret_mod <- caret::train(blueprint, 
                          data      = read_tr, 
                          method    = "lm", 
                          trControl = cv)

                        # For available methods in the train function

                          # ?names(getModelInfo())

                          # ?getModelInfo()$lm

caret_mod

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:28:04.63179Z","iopub.execute_input":"2022-10-14T19:28:04.63587Z","iopub.status.idle":"2022-10-14T19:28:04.786145Z"}}
caret_mod$results

# %% [markdown]
# ## 3.5. Check the model performance on the hold-out test set
# 
# Although the cross-validated model performance should provide a realistic picture of model performance, it is always useful to see how well the model does on a hold-out test set the model has never seen.
# 
# First, calculate the model predictions for the hold-out test dataset (read_te). Remember that the hold-out test dataset has 283 observations, so this will return a vector of 283 numbers.
# 
# Also, you only need to provide the test dataset in raw form without any data process. When we train the model using the caret::train() function, we provided the blueprint, and the model object `caret_mod` stores that blueprint. The function already knows how to process the variables in any new dataset according to the same blueprint, and then apply the model weights to calculate the predictions.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:31:54.506608Z","iopub.execute_input":"2022-10-14T19:31:54.50822Z","iopub.status.idle":"2022-10-14T19:31:54.767615Z"}}
predicted_te <- predict(caret_mod, read_te)

# %% [markdown]
# Now, compute RMSE, R-square, and MAE for the test dataset. These are very close to the performance values obtained from the 10-fold cross validation.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-14T19:35:09.341725Z","iopub.execute_input":"2022-10-14T19:35:09.34337Z","iopub.status.idle":"2022-10-14T19:35:09.377293Z"}}
rsq_te <- cor(read_te$target,predicted_te)^2
rsq_te

mae_te <- mean(abs(read_te$target - predicted_te))
mae_te

rmse_te <- sqrt(mean((read_te$target - predicted_te)^2))
rmse_te

# %% [markdown]
# <table>
#   <tr>
#     <th> </th>
#     <th>R-Square</th>
#     <th>MAE</th>
#     <th>RMSE</th>   
#   </tr>
#   <tr>
#     <th>Performance on 10-fold cross validation</th>
#     <th>0.679</th>
#     <th>0.477</th>
#     <th>0.602</th>
#   </tr>
#     <tr>
#     <th>Performance on Test Set</th>
#     <th>0.658</th>
#     <th>0.499</th>
#     <th>0.620</th>
#   </tr>  
#    <tr>
#     <th>Performance on Training Set</th>
#     <th>0.842</th>
#     <th>0.169</th>
#     <th>0.411</th>
#   </tr>
# </table>

# %% [markdown]
# 
# # 4. Using the Prediction Model for a New Text
# 
# We now have a model to predict the readability scores using 887 features. We also have a rough idea of how well it works. It is not a great model (it wouldn't win any prize in the Kaggle competition), but good enough to satisfy your advisor or boss. Now, how do we use this model to predict a readability score for a new text?
# 
# Suppose I have the following passage:
# 
#      Mittens sits in the grass. He is all alone. He is looking for some fun. Mittens hits his old ball. 
#      Smack! He smells a worm. Sniff! Mittens flips his tail back and forth, back and forth. 
#      Then he hears, Scratch! Scratch! What's that, Mittens? What's scratching behind the fence? 
#      Mittens runs to the fence. He scratches in the dirt. Scratch! Scratch! Ruff! Ruff! What's that, 
#      Mittens? What's barking behind the fence? Mittens meows by the fence. Meow! Meow!
# 
# What would be the predicted readability score for this reading passage?
# Moving forward, you need the R object (`caret_mod`) you created to save all the information from the fitted model using the `caret::train()` function.
# 
# First, let's do a cleanup. I will remove everything but the model object from my environment.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T22:15:19.076588Z","iopub.execute_input":"2022-10-06T22:15:19.079064Z","iopub.status.idle":"2022-10-06T22:15:19.103603Z"}}
# This is pretty old school, but works!

rm(list= ls()[!(ls() %in% c('caret_mod'))])

ls()

# %% [markdown]
# Now, we have to remember how we processed the text data and constructed the numerical the features before for the data we used to build the model. We should apply the exact same procedure to a new text and generate the same features for the new text.

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T22:16:07.728524Z","iopub.execute_input":"2022-10-06T22:16:07.730191Z","iopub.status.idle":"2022-10-06T22:16:07.752288Z"}}
new.text <- "Mittens sits in the grass. He is all alone. He is looking for some fun. Mittens hits his old ball. Smack! He smells a worm. Sniff! Mittens flips his tail back and forth, back and forth. Then he hears, Scratch! Scratch! What's that, Mittens? What's scratching behind the fence? Mittens runs to the fence. He scratches in the dirt. Scratch! Scratch! Ruff! Ruff! What's that, Mittens? What's barking behind the fence? Mittens meows by the fence. Meow! Meow!"

new.text  

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T22:17:47.462715Z","iopub.execute_input":"2022-10-06T22:17:47.464702Z","iopub.status.idle":"2022-10-06T22:19:55.400767Z"}}
require(reticulate)

use_condaenv('r-reticulate')

conda_install(envname  = 'r-reticulate',
              packages = 'sentence_transformers',
              pip      = TRUE)

st <- import('sentence_transformers')

model.name <- 'allenai/longformer-base-4096'

longformer      <- st$models$Transformer(model.name)
pooling_model   <- st$models$Pooling(longformer$get_word_embedding_dimension())
LFmodel         <- st$SentenceTransformer(modules = list(longformer,pooling_model))

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T22:23:01.908943Z","iopub.execute_input":"2022-10-06T22:23:01.911316Z","iopub.status.idle":"2022-10-06T22:23:06.015469Z"}}
new.embeddings <- LFmodel$encode(new.text)

new.embeddings.df <- as.data.frame(matrix(new.embeddings, 1,768))

new.embeddings.df

# %% [code] {"execution":{"iopub.status.busy":"2022-10-06T22:23:11.755113Z","iopub.execute_input":"2022-10-06T22:23:11.756673Z","iopub.status.idle":"2022-10-06T22:23:11.969509Z"}}
predict(caret_mod, new.embeddings.df)