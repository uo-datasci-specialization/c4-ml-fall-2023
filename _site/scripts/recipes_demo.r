# # The `recipes` package demo
# 
# We will first import the demo datasets to implement several data 
# processing steps discussed in the class. The purpose of this demo 
# is to understand the mechanics of working with the recipes package.
# 
# These datasets has a subset of the original recidivism dataset. 
# Both have the same seven variables:
# 
# - ID
# 
# - Gang_Affiliated (integer, binary, 0 and 1)
# 
# - Education_Level (character, categorical with three levels, nominal)
# 
# - Dependents (integer, discrete with four levels, ordinal)
# 
# - Avg_Days_per_DrugTest (double, continuous)
# 
# - DrugTests_Cocaine_Positive (double, bounded continuous between 0 and 1, 
#                               proportion)
# 
# - Recidivism_Arrest_Year2 (integer, binary, 0 and 1)
# 

d <- read.csv('./data/recipes_demo.csv', header=TRUE)

View(d)

new_d <- read.csv('./data/new_data.csv', header=TRUE)

View(new_d)

# When we work with the recipes package, there are three steps to follow:
# 
# - **Prepare a recipe**: This step includes declaring variables of interests 
#     to be used in modeling, assigning different roles to variables 
#     (e.g., predictor, id, outcome), and processes to be applied.
# 
# - **Mix the ingredients**: This step estimates certain parameters for 
#     data processing from a training data as needed. For instance, 
#     if a variable is standardized, we would need to compute a mean and 
#     standard deviation for this process. In this step, a mean and standard 
#     deviation is calculated and saved for future use based on some training 
#     data. Or, if we would impute missing values using an imputation model, 
#     then the parameters of the imputation model would be estimated and 
#     stored in this step from some training data for future use.
# 
# - **Bake it**: The processes initialized during the training data is 
#     applied to a dataset and a new transformed dataset is 
#     produced. The same dataset from the first two steps can be used for 
#     the third step. Or, the processes initialized can be applied to a 
#     new dataset with the same structure. 

################################################################################
# ## Example 1: No data processing
# 
# In this example, we will prepare a simple blueprint by declaring the variable 
# of interests in the dataset and assigning roles (ID, predictor, outcome), 
# and nothing else. This will not change anything in the dataset, but you will
# get familiar with the three steps.
# 
# ### Step 1: Prepare a recipe
# 
# First we prepare the recipe. In this example, we only declare the variables 
# to be used in the analysis. These variables must exist in the data object, 
# and you provide the column names. There are three roles we assign to each 
# variable we declare:
# 
# - **ID**: 'ID'
#
# - **predictor**: 'Gang_Affiliated','Education_Level','Dependents',
#                   'Avg_Days_per_DrugTest','DrugTests_Cocaine_Positive'
#
# - **outcome**: 'Recidivism_Arrest_Year2'
# 
# 
# 
# Note that you don't have to declare all variables in a dataset. 
# You only list the ones you are going to use as predictors and outcome. 
# It may also be useful to have and declare a unique identification variable 
# in the dataset for further analysis. For instance, your input data may 
# have 100 variables, but you can only declare the selected 10 variables 
# to be used in the analysis.

require(recipes)

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) 

blueprint












# ### Step 2: Train
# 
# We are going to mix the ingredients in the recipe by using the `prep` 
# function. In this case, this is just a formality because we didn't ask 
# to conduct any process on any of the variables. 
# The output object only print the number of variables for each assigned role.

prepare   <- prep(blueprint, 
                  training = d)
prepare

# ### Step 3: Bake
# 
# Now, we apply the prepared recipe to a dataset. This can be the same dataset
# we used to prepare the recipe or a new dataset that has the same data 
# structure. In either case, since there was no process integrated in our 
# recipe, it will return the exact same input data.



baked_d <- bake(prepare, 
                new_data = d)

baked_d


baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################
# ## Example 2: One-hot encoding
# 
# In this example, we will add a process to our recipe from Example 1. 
# We will ask to create dummy variables for group membership of the 
# Education_Level variable.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE)


blueprint













# In the training case, `prep` function will learn about the different 
# levels of the Education_Level variable and will create the coding 
# scheme to create dummy variables and will store that information.
# 
# |                       | Dummy Variable 1 | Dummy Variable 2 | Dummy Variable 3 |
# |-----------------------|:----------------:|:----------------:|:----------------:|
# | At least some college |     1            |       0          |        0         |
# | High School Diploma   |     0            |       1          |        0         |
# | Less than HS diploma  |     0            |       0          |        1         | 






prepare   <- prep(blueprint, 
                  training = d)
prepare












# Now, we can apply the prepared recipe to any dataset. Examine the output, 
# and note the changes in the returned data object. 



baked_d <- bake(prepare, 
                new_data = d)

baked_d





baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################
# ## Example 3: Missing Data Indicators
# 
# In this example, we will add a new process to our recipe from Example 2. 
# We will ask to create dummy variables for missingness. `all_predictors()` is something we can use to apply the process to all predictors defined earlier instead of typing all the variable names one by one. 
# 
# Examine the returned data object and note the changes.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors())





prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d


baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################
# ## Example 4: Remove zero variance variables
# 
# Now, we will add a process to remove any variable with zero variance. 
# If there is any predictor with zero variance, we don't need them in our 
# model anyway as they don't have any information. This will also remove 
# the unnecessary missing indicator variables created in the previous step 
# for those variables with no missing data.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors())











prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d



baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################
# ## Example 5: Impute missing values with mean
# 
# Now, we will add a process to impute the missing values with the mean for 
# those variables with missing values.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) 




prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d


# What numbers were used to impute the missing values for the variables 
# 'Gang_Affiliated', 'Avg_Days_per_DrugTest', and 'DrugTests_Cocaine_Positive'? 
# Does it make sense?

mean(d$Gang_Affiliated,na.rm=TRUE)
mean(d$Avg_Days_per_DrugTest,na.rm=TRUE)
mean(d$DrugTests_Cocaine_Positive,na.rm=TRUE)


baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new

# What numbers were used to impute the missing values for the variables 
# 'Gang_Affiliated', 'Avg_Days_per_DrugTest', and 'DrugTests_Cocaine_Positive' 
# in the new dataset? Does it make sense?

mean(new_d$Gang_Affiliated,na.rm=TRUE)
mean(new_d$Avg_Days_per_DrugTest,na.rm=TRUE)
mean(new_d$DrugTests_Cocaine_Positive,na.rm=TRUE)


# For categorical variables, you can use the `step_impute_mode()` function to 
# impute the missing values with the most frequently observed category. 
# Also, there are other functions to use more complex imputation models:
# 
# - `step_impute_bag()`, `
# - `step_impute_knn()`, 
# - `step_impute_linear()`.


################################################################################
# ## Example 6: Polynomial basis functions
# 
# We will add the polynomial basis functions up to the third term for the 
# variable `Dependents`.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors())  %>%
  step_poly('Dependents',degree=3)



prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d


baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################
# ## Example 7: Box-Cox transformation
# 
# We will apply the Box-Cox transformation for the Avg_Days_per_DrugTest variable.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors())  %>%
  step_poly('Dependents',degree=3) %>%
  step_BoxCox('Avg_Days_per_DrugTest')


prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d


baked_d_new <- bake(prepare, 
                    new_data = new_d)

baked_d_new


################################################################################

# ## Example 8: Standardization
# 
# We will standardize the Avg_Days_per_DrugTest variable just after we 
# implement the Box-Cox transformation.

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors())  %>%
  step_poly('Dependents',degree=3) %>%
  step_BoxCox('Avg_Days_per_DrugTest') %>%
  step_normalize('Avg_Days_per_DrugTest')


prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d


################################################################################
# ## Example 9: Logit Transformation
# 
# We will apply the logit transformation for the DrugTests_Cocaine_Positive 
# variable as it represents a proportion.
# 
# step_logit(all_of(props)) %>%
#   

blueprint <- recipe(x     = d,
                    vars  = c('ID','Gang_Affiliated','Education_Level',
                              'Dependents','Avg_Days_per_DrugTest',
                              'DrugTests_Cocaine_Positive',
                              'Recidivism_Arrest_Year2'),
                    roles = c('ID',rep('predictor',5),'outcome')) %>% 
  step_dummy('Education_Level',one_hot=TRUE) %>%
  step_indicate_na(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors())  %>%
  step_poly('Dependents',degree=3) %>%
  step_BoxCox('Avg_Days_per_DrugTest') %>%
  step_normalize('Avg_Days_per_DrugTest') %>%
  step_logit('DrugTests_Cocaine_Positive',offset = 0.001)


prepare   <- prep(blueprint, 
                  training = d)

baked_d   <- bake(prepare, 
                  new_data = d)

baked_d

# %% [markdown]
# ## All `step_` functions available in the recipe package for further 
# exploration
# 
# Check this page.
# 
# https://recipes.tidymodels.org/reference/index.html