library(WeightedROC)

csvFile <- read.csv(file="zipROC.csv", header = TRUE)
baseline=seq(from=0,to=1)
tittle <- "ROC curve of linear regression Vs Baseline"
plot(x=csvFile$TPR,y=csvFile$FPR, type="l",col="red", main=tittle,xlab="FPR",ylab="TPR")
lines(x=baseline,y=baseline, col="blue")
legend(x=0,y=1,legend=c("Regression", "BaseLine"), col=c("red","blue"), lty=1)
