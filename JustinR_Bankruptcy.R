bank <- data.frame(bankruptcy)
str(bank)

#Correlations

bank.select <- select(bank, 2, 4:27)
bank.cor <- cor(bank.select)
corrplot(bank.cor, method = "color", addCoef.col = "black", number.cex = 0.3)

#R9
ggplot(data=bank, aes(x=D, y=R9, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="green")

#R17
ggplot(data=bank, aes(x=D, y=R17, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="red")

#R23
ggplot(data=bank, aes(x=D, y=R23, color=D))+ 
  geom_point()+
  geom_smooth(method = "lm", color="yellow")

#Boxplots
ggplot(aes(x=D, y=R9, fill=D), data = bank) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R9")+
  ggtitle("R9 Boxplot (CURASS/CURDEBT)")

ggplot(aes(x=D, y=R17, fill=D), data = bank) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R17")+
  ggtitle("R17 Boxplot (INCDEP/ASSETS)")

ggplot(aes(x=D, y=R23, fill=D), data = bank) +
  geom_jitter(alpha=.25) +
  geom_boxplot(alpha=.25, aes(group=D))+
  labs(x= "Bankrupt vs. Not Bankrupt", y= "R23")+
  ggtitle("R23 Boxplot (WCFO/ASSETS)")

#Regression
bank.reg <- lm(D~R1+R2+R3+R4+R5+R6+R7+R8+R9+R10+R11+R12+R13+R14+
               R15+R16+R17+R18+R19+R20+R21+R22+R23+R24, data=bank)

summary(bank.reg)

train.index <- sample(c(1:dim(bank)[1]), dim(bank)[1]*0.6)  
train.df <- bank[train.index,]
valid.df <- bank[-train.index,]

#Stepwise Regression Week 3

bank.back <- step(bank.reg, direction = "backward")
summary(bank.back)
bank.back.pred <- predict(bank.back, valid.df)
accuracy(bank.back.pred, valid.df$D)

bank.null <- lm(D~1, data = train.df)
bank.forward <- step(bank.null, 
                     scope=list(lower=bank.null, upper=bank.reg), 
                     direction = "forward")
summary(bank.forward)
bank.forward.pred <- predict(bank.forward, valid.df)
accuracy(bank.forward.pred, valid.df$D)

bank.both <- step(bank.reg, direction = "both")
summary(bank.both)
bank.both.pred <- predict(bank.both, valid.df)
accuracy(bank.both.pred, valid.df$D)

comparison <- data.frame(
  Backward=c(accuracy(bank.back.pred, valid.df$D)),
  Forward= c(accuracy(bank.forward.pred, valid.df$D)),
  Both=c(accuracy(bank.both.pred, valid.df$D))
)
comparison
rownames(comparison) <-c("ME", "RMSE", "MAE", "MPE", "MAPE")
comparison

SWR.CM <- confusionMatrix(factor(ifelse(bank.both.pred > 0.5, 1, 0)), 
                          factor(valid.df$D))
SWR.CM

#KNN
bankless <- bank[,-c(1,3)]
ktrain.index <- sample(c(1:dim(bankless)[1]), dim(bankless)[1]*0.6)  
ktrain.df <- bankless[train.index,]
kvalid.df <- bankless[-train.index,]

norm.values <- preProcess(ktrain.df, method = c("center", "scale") )
train.norm.df <- predict(norm.values, ktrain.df)
valid.norm.df <- predict(norm.values, kvalid.df)
bank.norm.df <- predict(norm.values, bank)

train.nn <- knn(train = train.norm.df, test = valid.norm.df,
                cl = train.norm.df[, "D"], k = 1)
row.names(ktrain.df)[attr(train.nn, "nn.index")]
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

for(i in 1:14){
  knn.pred <- knn(train.norm.df, valid.norm.df,
                  cl = train.norm.df[, "D"], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, 
                                       factor(valid.norm.df[, "D"]))$overall[1]
}
accuracy.df
plot(accuracy.df)

knn.pred <- knn(train.norm.df, valid.norm.df,
                cl = train.norm.df[, "D"], k = 4)
knn.cm <- confusionMatrix(knn.pred, factor(valid.norm.df[, "D"]))
knn.cm

#Stepwise Regression = .9057 / 90.57% 
#KNN Accuracy = .9245 / 92.45%
