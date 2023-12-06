---
  #title: "Recommendation_project"
  #git: https://github.com/chrislcp/Project.git

  # 1- Introduction 
  
#For this first Project in Capstone we have the goal to develop an recommendation system considering the dataset Movielens, more precisely an subset with 10M.It is a problem with large dataset.The dataset was extracted from EDX and the data wrangling was initiated for us. 
#Define the variables
#Identification of Movie and User: MovieID and UserId 
#rating - rating of the movies 
#timestamp -> a number of the date tht the user have rated the film
#Data -> Subsets:edx (train) and final_holdout_test (test)

## 1.1-Loading Libraries

library(tidyverse)
library(caret)
library(stringr)
library(recommenderlab)

## 1.2-Data Preprocessing
### Data Loading 
#Code that was provided from EDX to download the dataset and start the preprocessing, as split the dataset and organize the data with joins. 
#The principal dataset is called movielens and it wil be separated in edx (training set) and final_holdout_test (test set without some rows that do not exist in edx)

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

#if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%mutate(userId = as.integer(userId),movieId = as.integer(movieId),rating = as.numeric(rating),timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

## 1.3- QUIZ from EDX
#This entire chunk was developed to answer the QUIZ for this Capstone. 
zero <- edx%>%filter(rating=="0")
three <- edx%>%filter(rating=="3")
ids <- edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#How many users?
n_distinct(edx$userId)
#Q4- How many different movies are in the edx dataset?
n_distinct(edx$movieId)

# Q5 - str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {sum(str_detect(edx$genres, g))})
#Q6 - Q8
edx %>% group_by(movieId,title) %>% summarize(count = n()) %>%arrange(desc(count)) 

edx%>%group_by(rating)%>%summarize(count=n())%>%top_n(10)%>%arrange(desc(count))
#2-Analysis
##2.1 Preprocessing -Handling Missing Data - remove the entire rows that contain any NA and Edit columns: timestamp as date and then as Year

## Missing Data
edx <- edx %>% drop_na() 
## Create  the Year Column for edx and final_holdout_test
edx <- edx%>%mutate(date= as_datetime(timestamp))%>%mutate(Year = year(date))
final_holdout_test <- final_holdout_test%>%mutate(date= as_datetime(timestamp))%>%mutate(Year = year(date))
## 2.2 Dataviz - Exploratory Data Analysis (EDA)
head(edx)
hist(edx$movieId)
hist(edx$rating)
hist(edx$userId)
###2.2.1Boxplot -rating
boxplot(edx$rating, data=movielens)
## 2.3 - Method: Evaluate the YEAR influence and try the  KNN(method: k-Nearest Neighbors) 
#One idea was to see if the year was an variable with some importance and the idea was to test a very small subset.
#EXAMPLE of a year ->CHOOSE A YEAR with => 2009 and try a different method as knn
hist(edx$Year)

edx %>% group_by(Year) %>%	summarize(n = n(), avg_rating = mean(rating), se_rating = sd(rating)/sqrt(n()))

RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))}

edx_2009 <- edx%>%filter(Year ==2009)
fit_knn2009 <- knn3(rating ~ movieId, data= edx_2009)
final_holdout_test_2009 <- final_holdout_test%>%mutate(date= as_datetime(timestamp)) %>%mutate(Year = year(date))%>%filter(Year ==2009)
y_hat2009 <- predict(fit_knn2009, final_holdout_test_2009)
RMSE_2009 <- RMSE(final_holdout_test_2009$rating,y_hat2009)
RMSE_2009
# The value was really high ==3.506498 
# When I tried to choose a year like 2008, the same code was not able to run.Too many rows.

#2.4 - Basic tests (Reference:as EDX explained in textbook) 

#The idea was to exploring solutions considering the edx solutions presented at Chapter 33 and then add two different variables (year and week)

#There were applied 5 models to this dataset:
#simple mean
# consider only the movieID variable, with a simplified calculation for the model fit <- lm(rating ~ as.factor(movieId), data = edx)
#Add userId-consider only the movieID variable and userID
#NEW
# Add Year (it was not a good idea) 
# Add Week (it was a possible variable)

##2.4.1-EDX BASIC -> THE SIMPLE MEAN
mu_hat<- mean(edx$rating, rm.NA=TRUE)
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))}

mu_simple <- RMSE(final_holdout_test$rating,mu_hat)
# The mu_simple was 1.061202
##2.4.2-EDX BASIC -> as.factor(movieId)
mu <- mean(edx$rating) 
movie_avgs <- edx %>%group_by(movieId) %>% summarize(b_i = mean(rating - mu))

predicted_ratings1 <- mu + final_holdout_test %>% left_join(movie_avgs, by='movieId') %>% pull(b_i)
RMSE(predicted_ratings1, final_holdout_test$rating)
# The RMSE was 0.9439087
##2.4.3-EDX BASIC -> as.factor(movieId) and as.factor(userId)
user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%summarize(b_u = mean(rating - mu - b_i))

predicted_ratings2 <- final_holdout_test%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
RMSE(predicted_ratings2, final_holdout_test$rating)
#The RMSE was 0.8653488. It is better than the two others.
#3- Results

##3.1- Model1-Add YEAR
#Is the YEAR an important variable? as.factor(movieId),as.factor(userId) and as.factor(Year)
year_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% group_by(Year) %>%summarize(bu2 = mean(rating - mu - b_i))

predicted_ratings3 <- final_holdout_test%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId')%>% left_join(year_avgs, by='Year') %>% mutate(pred = mu + b_i + b_u+ bu2) %>% pull(pred)

RMSE(predicted_ratings3, final_holdout_test$rating)
#The RMSE was 0.8677787. As it was a little higher then the method with just the two variable movieId and userId, we will not choose this method. 
##3.2-Model2 -Add WEEK as variable. Is the WEEK of the avaliation an important variable? as.factor(movieId),as.factor(userId) and as.factor(week)
#Include the week column
edx <- edx %>% mutate(week= week(date))
final_holdout_test <- final_holdout_test%>% mutate(week= week(date))

# Try the model with 
week_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% group_by(week) %>%summarize(bu_week = mean(rating - mu - b_i))

predicted_ratings3 <- final_holdout_test%>% left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId')%>% left_join(week_avgs, by='week') %>% mutate(pred = mu + b_i + b_u+ bu_week) %>% pull(pred)

RMSE(predicted_ratings3, final_holdout_test$rating)
# The RMSE was 0.8656555 a little lower than with just movieID and userId. In this case, the week can be introduce in this model in order to help the loss reduction. 
#3.3- Recommerderlab- Comparing with new methodologies 

#In order to begin a study and try to understand some new approaches, after have understood the basic ideas of modeling a recommendation problem. 
#These two references below explained how to start with the package recommederlab. The first orientation in this kind of problem was to use the: User-based collaborative filtering (UBCF).

#References: 
  #1) RDocumentation- R package recommenderlab - Lab for Developing and Testing Recommender Algorithms(https://www.rdocumentation.org/packages/recommenderlab/versions/1.0.6)
#2) Package ‘recommenderlab’ - Title Lab for Developing and Testing Recommender Algorithms - (https://cran.r-project.org/web/packages/recommenderlab/recommenderlab.pdf)
library(recommenderlab)
edx_Matrix <- as(edx,"realRatingMatrix")
model <- Recommender(edx_Matrix,method="UBCF")
final_holdout_test_Matrix <- as(final_holdout_test,"realRatingMatrix")


#4-Conclusion
#This Capstone was helpful in trying to solve a problem with large datasets. That showed us that the approaches are not simple and we can not use all the modeling that we have learned. I've tried some other modeling approaches, but it was really difficult and nothing worked.
#Finally, in this work , I've tried to use the EDX methodologies presented in Chapter 33 and added other two hypotheses: Does the date of the rating help the prediction? To answer this I have tried to insert Year and week in the models and the compared the result in the loss.
#The week variable helps to reduce the loss (RMSE).A future work in this case would be study some Regularization issues to opitimaze the model. 
#Also, for future works, I have started to study "recommederlab" package, and it was easy to develop the model.
#But I haven't managed yet to predict or evaluate the results. I send the functions here, cause maybe someone would like to finish the challenge and find my error.

## 4.1- Future Works

# The 3 commands listed below are not working. But I will try to understand and I think this kind of package will help this problems in new. 
#evaluate(model, final_holdout_test_Matrix)
#predictions <- predict(model,final_holdout_test_Matrix, type="rating")
## predictions for the test data 
#predict(model, getData(e, "known"), type="topNList", n = 10)