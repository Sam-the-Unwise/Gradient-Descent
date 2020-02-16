
# read in CSV of information
csvData <- read.csv2("C:\\github\\GradientDescent\\SAheartLogLoss.csv", header=TRUE, sep=",", dec=".")

# break up CSV for easier readability
numOfObservations <- length(csvData$train.loss)
inSeq <- seq(from=numOfObservations,to=1,by=-1)
trainLoss <- csvData$train.loss
valLoss <- csvData$validation.loss

min <- min(valLoss)
minIndex <- match(min,valLoss)

# plot validation vs train
tittle <- paste("Spam data, Training set Vs. Validation set,\n loss function over number of observations,\n N = ", numOfObservations)
plot(x=inSeq, y=trainLoss, type="l", main=tittle, ylab="loss function", xlab="number of observations", col="red")
lines(x=inSeq,y=valLoss, col="blue")
legend(x=1, y=5, legend=c("Train", "Validation", "Minimum"),col=c("red","blue", "black"),lty=1)
points(x=inSeq[minIndex],y=min)

