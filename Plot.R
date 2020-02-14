
# read in CSV of information
csvData <- read.csv2("/home/jacob/Documents/Cs_499/GradientDescent/SpamLogLoss.csv", header=TRUE, sep=",", dec=".")

# break up CSV for easier readability
csvData <- csvData[-57,]
numOfObservations <- length(csvData$train.loss)
inSeq <- seq(from=1,to=numOfObservations,by=1)
trainLoss <- csvData$train.loss
valLoss <- csvData$validation.loss
graphSizeX <- c(1, 1499)
graphSizeY <- c(13, 14)
  
# plot validation vs train
tittle <- paste("Spam data, Training set Vs. Validation set, loss function over number of observations, N = ", numOfObservations)
plot(x=graphSizeX, y=graphSizeY, type="n", main=tittle, ylab="loss function", xlab="number of observations")
lines(x=inSeq,y=trainLoss)
lines(x=inSeq,y=valLoss)