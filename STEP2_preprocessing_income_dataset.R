incomeDataSet <- read.csv("path/to/income_prediction/datasets/realTrainData.csv", sep=",",dec = ",")
trainDataset$income_.50K<- as.factor(trainDataset$income_.50K) #convert target feature to a factor

##### PREPROCESSING #####

# Z-score calculation functions 
# Function for centering a vector
center <- function(v) { v-mean(v) }
# Function for scaling a vector
scale <- function(v) { v/sd(v) }

# Normalization with Z-Score to numeric values in the dataset 
incomeDataSet[, c(1 ,3, 8, 9, 10)] <- incomeDataSet[, c(1 ,3, 8, 9, 10)] %>%
  apply(MARGIN = 2, FUN = center) %>%
  apply(MARGIN = 2, FUN = scale)

# Under-Sampling
smaller_values <- subset(incomeDataSet , incomeDataSet$income_.50K == 0)
greater_values <- subset(incomeDataSet , incomeDataSet$income_.50K == 1)
smaller_ind <- sample(2, nrow(smaller_values), replace = T, prob = c(0.53, 0.47)) # 2 to 1 ratio

trainSet <- rbind(greater_values, smaller_values[smaller_ind == 1, ])
set.seed(42)
rows <- sample(nrow(trainSet))
shuffleTrain <- trainSet[rows,]

write.csv(shuffleTrain,"path/to/income_prediction/datasets/underSampledData.csv", row.names = FALSE)
