set.seed(1)
library(data.table)
library(keras)
library(ggplot2)

#hyperparameters
n.folds <- 5
epochs <- 20

#credit to this website https://acadgild.com/blog/55690-2 for giving me the getMode function
getMode <- function(x)
{
  uniq <- unique(x)
  uniq[which.max(tabulate(match(x,uniq)))]
}

###############################################################################
###############################################################################
####################### Data Upload and Organization ##########################
###############################################################################
###############################################################################


if(!file.exists("zip.train.gz"))
{
  download.file("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz","zip.train.gz")
}

zip.train <- data.table::fread("zip.train.gz")
n.obs <- nrow(zip.train) #7291  

data.matrix <- matrix(unlist(zip.train), nrow(zip.train), ncol(zip.train))
X <- data.matrix[,2:257]
Y <- data.matrix[,1]


fold_vec <- sample(rep(1:n.folds, ceiling(n.obs/n.folds)), n.obs)
is.test <- sample(rep(c(TRUE, FALSE), times = c(1458, 5833)), n.obs) #hard coded values

X.train.list <- list()
Y.train.list <- list()
X.test.list <- list()
Y.test.list <- list()

for( curr.fold in 1:n.folds )
{
  X.train.list[[curr.fold]] <- X[(fold_vec == curr.fold) & !is.test,]
  Y.train.list[[curr.fold]] <- Y[(fold_vec == curr.fold) & !is.test]
  X.test.list[[curr.fold]] <- X[(fold_vec == curr.fold) & is.test,]
  Y.test.list[[curr.fold]] <- Y[(fold_vec == curr.fold) & is.test]
  
  X.train.list[[curr.fold]] <- array(X.train.list[[curr.fold]], c(nrow(X.train.list[[curr.fold]]),16,16,1))
  X.test.list[[curr.fold]] <- array(X.test.list[[curr.fold]], c(nrow(X.test.list[[curr.fold]]),16,16,1))
}


###############################################################################
###############################################################################
############################### Train Models ##################################
###############################################################################
###############################################################################


#data preparation
batch_size <- 128
num_classes <- 10

#Input image dimensions
img_rows <- 16
img_cols <- 16
input_shape <- c(img_rows, img_cols, 1)

#initialize model list
conv.model.list <- list()
dense.model.list <- list()
conv.result.list <- list()
dense.result.list <- list()

for( curr.fold in 1:n.folds)
{

  #############################################################################
  ######################## Train Convolutional Models #########################
  #############################################################################

  #define convolutional model
  conv.model.list[[curr.fold]] <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = input_shape) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = num_classes, activation = 'softmax')
  
  #compile convolutional model
  conv.model.list[[curr.fold]] %>% compile(
    loss = loss_categorical_crossentropy,#for multi-class classification
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy'))
    
  #store fitted model in results
  conv.result.list[[curr.fold]] <- conv.model.list[[curr.fold]] %>% fit(
    X.train.list[[curr.fold]], to_categorical(Y.train.list[[curr.fold]], num_classes),
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2)
  
  #############################################################################
  ########################### Train Dense Models ##############################
  #############################################################################
  
  #define dense model
  dense.model.list[[curr.fold]] <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>% 
    layer_dense(units = 404, activation = 'relu') %>% 
    layer_dense(units = 404, activation = 'relu') %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dense(units = num_classes, activation = 'softmax')
  
  #compile dense model
  dense.model.list[[curr.fold]] %>% compile(
    loss = loss_categorical_crossentropy,#for multi-class classification
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy'))
  
  #store fitted model in results
  dense.result.list[[curr.fold]] <- dense.model.list[[curr.fold]] %>% fit(
    X.train.list[[curr.fold]], to_categorical(Y.train.list[[curr.fold]], num_classes),
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2)
}


###############################################################################
###############################################################################
######################### HyperParameter Tuning ###############################
###############################################################################
###############################################################################


# best_epochs
#I put this comment here in case you use control f to find variable names.
#I felt like best_epochs wasn't a good variable name for how I implemented
#the structure of this project, so I decided to create two other best_epochs
#variables 


#initialize best.epoch variables to store the best epoch for each
#respecitve model
conv.best.epochs <- numeric(n.folds)
dense.best.epochs <- numeric(n.folds)


for(curr.fold in 1:n.folds )
{
  conv.best.epochs[curr.fold] <- which.min(conv.result.list[[curr.fold]]$metrics$val_loss)
  dense.best.epochs[curr.fold] <- which.min(dense.result.list[[curr.fold]]$metrics$val_loss)
}


###############################################################################
###############################################################################
############################## Retrain Models #################################
###############################################################################
###############################################################################


#initialize model lists
conv.model.list <- list()
dense.model.list <- list()
conv.result.list <- list()
dense.result.list <- list()

#initialize evaluation lists
conv.evaluation.list <- list()
dense.evaluation.list <- list()

#initialize accuracy vectors
conv.accuracy <- numeric(n.folds)
dense.accuracy <- numeric(n.folds)
baseline.accuracy <- numeric(n.folds)

for( curr.fold in 1:n.folds)
{
  
  #############################################################################
  ######################## Train Convolutional Models #########################
  #############################################################################
  
  #define convolutional model
  conv.model.list[[curr.fold]] <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = input_shape) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = num_classes, activation = 'softmax')
  
  #compile convolutional model
  conv.model.list[[curr.fold]] %>% compile(
    loss = loss_categorical_crossentropy,#for multi-class classification
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy'))
  
  #store fitted model in results
  conv.result.list[[curr.fold]] <- conv.model.list[[curr.fold]] %>% fit(
    X.train.list[[curr.fold]], to_categorical(Y.train.list[[curr.fold]], num_classes),
    batch_size = batch_size,
    epochs = conv.best.epochs[curr.fold],
    validation_split = 0)
  
  #############################################################################
  #################### Evaluate Convolutional Models ##########################
  #############################################################################
  
  #store convolution evaluation in a list
  conv.evaluation.list[[curr.fold]] <- conv.model.list[[curr.fold]] %>%
    evaluate( X.test.list[[curr.fold]], 
              to_categorical(Y.test.list[[curr.fold]], num_classes),
              verbose = 0)
  
  #store convolutional accuracy in a vector
  conv.accuracy[curr.fold] <- conv.evaluation.list[[curr.fold]]$accuracy
  
  #############################################################################
  ########################### Train Dense Models ##############################
  #############################################################################
  
  #define dense model
  dense.model.list[[curr.fold]] <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>% 
    layer_dense(units = 404, activation = 'relu') %>% 
    layer_dense(units = 404, activation = 'relu') %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dense(units = num_classes, activation = 'softmax')
  
  #compile dense model
  dense.model.list[[curr.fold]] %>% compile(
    loss = loss_categorical_crossentropy,#for multi-class classification
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy'))
  
  #store fitted model in results
  dense.result.list[[curr.fold]] <- dense.model.list[[curr.fold]] %>% fit(
    X.train.list[[curr.fold]], to_categorical(Y.train.list[[curr.fold]], num_classes),
    batch_size = batch_size,
    epochs = dense.best.epochs[curr.fold],
    validation_split = 0)
  
  #############################################################################
  ######################## Evaluate Dense Models ##############################
  #############################################################################
  
  #store dense evaluation in a list
  dense.evaluation.list[[curr.fold]] <- dense.model.list[[curr.fold]] %>%
    evaluate( X.test.list[[curr.fold]], 
              to_categorical(Y.test.list[[curr.fold]], num_classes),
              verbose = 0)
  
  #store dense accuracy in vector
  dense.accuracy[curr.fold] <- dense.evaluation.list[[curr.fold]]$accuracy
  
  #############################################################################
  ####################### Evaluate Baseline Models ############################
  #############################################################################
  
  
  baseline.accuracy[curr.fold] <- sum(Y.test.list[[curr.fold]] == getMode(Y.test.list[[curr.fold]])) / length(Y.test.list[[curr.fold]])
}

###############################################################################
###############################################################################
################################# Plotting ####################################
###############################################################################
###############################################################################

#make the accuracy data structure
accuracy.dt <- data.table( 
  Accuracy = c( conv.accuracy, dense.accuracy, baseline.accuracy ),
  GraphType = rep(c("Convolutional", "Dense", "Baseline"), each = 5))

#make plot
p <- ggplot(accuracy.dt, aes(x = GraphType, y = Accuracy)) +
  geom_dotplot(binaxis = 'y', stackdir = 'center') + 
  coord_flip()

#print plot
print(p)


