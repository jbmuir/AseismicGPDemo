library("SAPP")
data = read.table("ridgecrest_data.txt")
data <- data[920:1,]
data$V1 <- data$V1-data$V1[1]
data<-data[152:nrow(data),]
data<-data[2:nrow(data),]
data$V1 <- data$V1-data$V1[1]
etasoutput <- etasap(data$V1,data$V2,threshold=3.0,reference=7.1,parami=c(0,0.5,0.1,0.2,1.05), tstart=1e-5,zte=8.81)
etasoutput$ngmle
etasoutput$aic2
etasoutput$param$mu
etasoutput$param$K
etasoutput$param$c
etasoutput$param$alpha
etasoutput$param$p
