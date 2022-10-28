data_log_reg <- read.csv("/Users/brandonvidro/Downloads/LassoForwardSelectionDataSetForModelingLogReg.csv")

head(data_log_reg)

log_reg = glm(AboveAverageLifeExpectancyByYear ~ ., data = data_log_reg, family = "binomial")
summary(log_reg)


log_reg.probs = predict(log_reg, type = "response")
log_reg.pred = rep("FALSE", length(log_reg.probs))
log_reg.pred[log_reg.probs > 0.5] = "TRUE"
table(log_reg.pred, data_log_reg$AboveAverageLifeExpectancyByYear)

((2148+3281) / (2148 + 398 + 432 + 3281)) * 100