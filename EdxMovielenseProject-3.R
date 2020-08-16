##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# MovieLens Project EDX
##########################################################

# Add additional library

install.packages("kableExtra")
library(kableExtra)

### DATA PARTITION 

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean up memory by deleting unsused objects and performing a garbage collection
rm(dl, ratings, movies, test_index, temp, movielens, removed)
gc()



### DATA ANALYSIS

dim(edx)

head(edx)

# Exploring movies

edx_movies <- edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  arrange(desc(count))


# let's see the movies tendency

summary(edx_movies$count)


# ratings Distribution

ggplot(data = edx, aes(x = rating)) +
  geom_bar() + 
  labs(title = "Ratings Distribution", x = "Rating", y = "Number of ratings")


# Movie Ratings distribution  per year 

movies_year <- edx %>%
  transform(timestamp = format(as.POSIXlt(timestamp, origin = "1970-01-01"), "%Y")) %>%
  select(timestamp, movieId) %>%
  group_by(timestamp) %>%
  summarise(count = n_distinct(movieId))

ggplot(data = movies_year, aes(x = timestamp, y = count)) +
  geom_bar(stat = "identity") + 
  labs(title = "Movies Ratings distribution per year", x = "Year", y = "Number of ratings")


#### Finding the model

# Here is the formula of RMSE

RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# calculate the overall average rating on the training dataset

u <- mean(edx$rating)

# Calculate RMSE using validation ratings

RMSE(validation$rating, u) # 1.06 as RMSE is not good enough


# let's calculate the RSME including the movie effect b_i

# caluclate b_i for each movie and let's compare it  with the overall average u on training dataset

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - u))


#  Lets add b_i into the validaation set and lets predict all unknown ratings with u and b_i

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = u + b_i) %>%
  pull(pred)


#  calculate RMSE of movie ranking effect
RMSE(validation$rating, predicted_ratings) # 0.94 still not good enough


# plot the distribution of b_i's. 
qplot(b_i, data = b_i, bins = 15, color = I("black")) # the distrbution is normal


# lets train the model with movie effect (b_i) and users effect (b_u)

# lest find b_u

b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - u - b_i))


# predict new ratings with movie and user bias

predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = u + b_i + b_u) %>%
  pull(pred)


# calculate RMSE of movie ranking effect
RMSE(predicted_ratings, validation$rating)


edx %>% 
  group_by(userId) %>% 
  summarize (b_u = mean(rating - u - b_i)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black") # is a normal curve


  # lets optimized movie and user effect method with the best regularization factor (lamba)
  
  # let's determine the best lambda from a sequence
  
  lambdas <- seq(from=0, to=10, by=0.25 )
  
  # output RMSE of each lambda, repeat earlier steps (with regularization)
  
  rmses <- sapply (lambdas, function(l) {
    
    # calculate average rating across training data
    u <- mean(edx$rating)
    
    # compute regularized movie bias term
    b_i <- edx %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - u)/(n()+l))
    
    # compute regularize user bias term
    b_u <- edx %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - u)/(n()+l))
    
    # compute predictions on validation set based on these above terms
    predicted_ratings <- validation %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = u + b_i + b_u) %>%
      pull(pred)

    # output RMSE of these predictions
    return(RMSE(predicted_ratings, validation$rating))
  })

  # quick plot of RMSE vs lambdas
  qplot(lambdas,rmses)
  # print minimum RMSE 
  min(rmses)

#### Final model with the best fitted lambda 


  # The final linear model with the minimun lambda to optimise the model
  
  lam <- lambdas[which.min(rmses)]
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - u)/(n()+lam))
  # compute regularize user bias term
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - u)/(n()+lam))
  
  # compute predictions on validation set based on these above terms
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = u + b_i + b_u) %>%
    pull(pred)
  
  # Let's find the RMSE based on the above terms
  RMSE(predicted_ratings, validation$rating)
  
  