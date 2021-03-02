#libraries
library(ggplot2)
library(ggcorrplot)
library(visdat) #helps to plot NA values)

dataFrame <- read.csv("path/to/income_prediction/datasets/train.csv", sep=",",dec = ",")

dataFrame[dataFrame==""] <- NA #replace all empty cells with NA's

# Determine NA values in the dataset with plot
vis_dat(dataFrame)

# Delete NA values
df <- data.frame(na.omit(dataFrame)) #omit all NA rows
df$income_.50K<- as.factor(df$income_.50K) #convert target feature to a factor
DF1 <- as.data.frame(df)

#### DATA VISUALIZATION #####

# Target feature levels distribution
barplot(table(DF1[,"income_.50K"]), xlab = "Target Feature Type")
legend("topright",
       legend = c("0 - <=$50K" , "1 - >$50K "))

# Correlation Matrix
nums2 <- unlist(lapply(DF1,is.integer))# takes integer values
corr_mat=cor(DF1[,nums2],method="s") #spearman

ggcorrplot(corr_mat, hc.order = TRUE, outline.col = "white", lab=TRUE, lab_size=5, type = "upper")

# Descriptive feature plot
ggplot(DF1) + aes(x=as.numeric(age), group=income_.50K, fill=income_.50K,
                       color=income_.50K) +
       geom_histogram(binwidth=1, fill = "white")+
       labs(x="Age",y="Count",title = "Income vs Age")

ggplot(DF1) + aes(x=as.numeric(educational.num), group=income_.50K,
                       fill=income_.50K, color=income_.50K) +
       geom_histogram(binwidth=1, fill = "white")+
       labs(x="Educational Num",y="Count",title = "Income vs Educational Num")

ggplot(DF1, aes(x=as.numeric(hours.per.week), group=income_.50K,
                     fill=income_.50K, color=income_.50K)) +
       geom_histogram(fill="white", binwidth = 5)+
       labs(x="Hours per week",y="Number of Samples",title = "Income vs Hours per week")

# Income vs. fnlwgt plot
ggplot(DF1, aes(x=as.numeric(fnlwgt), group=income_.50K, fill=income_.50K,
                     color=income_.50K)) +
       geom_histogram(fill="white", binwidth = 1000)+
       labs(x="Final Weight",y="Number of Samples",title = "Income vs Final Weight")

# Descriptive features plot (capital gain-capital loss)
ggplot(DF1) + aes(x=as.numeric(capital.gain), group=income_.50K,
                                fill=income_.50K, color=income_.50K) +
       geom_histogram(binwidth=1000, fill = "white")+
       labs(x="Capital Gain",y="Count",title = "Income vs Capital Gain")

ggplot(DF1) + aes(x=as.numeric(capital.loss), group=income_.50K,
                                fill=income_.50K, color=income_.50K) +
       geom_histogram(binwidth=1000, fill = "white")+
       labs(x="Capital Loss",y="Count",title = "Income vs Capital Loss")

# Delete unused column
DF1 <- subset(DF1, select=-native.country)
DF1 <- subset(DF1, select=-race)
DF1 <- subset(DF1, select=-fnlwgt)
DF1 <- subset(DF1, select=-education)

write.csv(DF1,"path/to/income_prediction/datasets/realTrainData.csv", row.names = FALSE)
