#Load dataset
finalDataSetForModelling <- read.csv("finalDataSetForModelling.csv")

#Check it out
fix(finalDataSetForModelling)

pdf("Pair plots for predictors and Y.pdf")
pairs(finalDataSetForModelling[,1:11])
dev.off() 

pairs(finalDataSetForModelling[,1:3])

install.packages("rmarkdown", dep = TRUE)