data_log_reg <- read.csv("/Users/brandonvidro/Downloads/LassoForwardSelectionDataSetForModelingLogReg.csv")

head(data_log_reg)

log_reg = glm(AboveAverageLifeExpectancyByYear ~ ., data = data_log_reg, family = "binomial")
summary(log_reg)