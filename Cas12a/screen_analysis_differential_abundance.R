library("ggplot2")
library("RUVSeq")
library("edgeR")
library("plyr")
library(scales)
library(stringr)
library(reshape2)
library(RColorBrewer)

#Load the crRNA count file
tab <- read.table("counts_KLEB.csv", sep="\t", header=T, stringsAsFactors = F, quote="\"")
rownames(tab) <- tab$target
colnames(tab)<-c(colnames(tab)[1:14],'MFDpir-1', 'MFDpir-2', 'T=0 NTUH-1', 'T=0 NTUH-2', 'T=0 KppR1-1', 'T=0 KppR1-2', '3h_NTUH-1', '3h_NTUH-2', '3h_KppR1-1', '3h_KppR1-2','ON_NTUH-1', 'ON_NTUH-2', 'ON_KppR1-1', 'ON_KppR1-2')

###check the correlation between samples
pdf(file = "sample_check.pdf")
cormat <- cor(log(tab[,15:length(tab[1,])] + 1))
m <- cormat[hclust(dist(cormat))$order ,hclust(dist(t(cormat)))$order]
rows <- dim(m)[1]
cols <- dim(m)[2]
melt.m <- cbind(rowInd=rep(1:rows, times=cols), colInd=rep(1:cols, each=rows),
                reshape2::melt(m))
g <- ggplot2::ggplot(data=melt.m)

g <- g + ggplot2::geom_rect(ggplot2::aes(xmin=colInd-1,xmax=colInd,
                                         ymin=rowInd-1,ymax=rowInd, fill=value)
                            ,colour='white')

g <- g + ggplot2::scale_x_continuous(breaks=(1:cols)-0.5, labels=colnames(m))
g <- g + ggplot2::scale_y_continuous(breaks=(1:rows)-0.5, labels=rownames(m))

g <- g + ggplot2::theme(panel.grid.minor=ggplot2::element_line(colour=NA),
                        panel.grid.major=ggplot2::element_line(colour=NA),
                        panel.background=ggplot2::element_rect(fill=NA,
                                                               colour=NA),axis.text.x = element_text(angle = 90))

print(g)
dev.off()

###Barplot the logFC of different gene groups
plot_logFC <- function(guide_names, logFC, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=logFC, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("targeting", "NT"))
    s1t1$X2 <- as.numeric(logFC)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=X2, colour=X3, fill=X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(), axis.title.x=element_blank(),
                                                                                                                                         axis.text.x=element_blank(), 
                                                                                                                                         axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue"))+ ylab("logFC")+ggtitle(title)                      
    print(r)
}
###Plot the FDR distribution of differential abundance
plot_FDR <- function(guide_names, FDR, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=FDR, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("targeting", "NT"))
    s1t1$X2 <- as.numeric(FDR)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=-log(X2, base=10), colour=X3, fill=X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(), axis.title.x=element_blank(),
                                                                                                                                                        axis.text.x=element_blank(), 
                                                                                                                                                        axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue")) + xlab(length(guide_names))+ ylab("-log10 FDR") + ggtitle(title)                        
    print(r)
}
# timepoint_bar <- function(value,title){
#     comparison<-"targeting - nt"
#     classes<-rep(c("essential","non-essential","NC"),1)
#     value <-value
#     data <- data.frame(comparison,classes,value)
#     r <- ggplot(data,aes(fill=classes,y=value,x=comparison))+ geom_bar(position="dodge", stat="identity")+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+ylab("logFC")
#     print(r)
# }
lcpm_group <- function(y,title) {
    nt <- cpm(y[y$genes$type=="NT",], log=TRUE, normalized.lib.sizes=TRUE)
    ess<- cpm(y[y$genes$type=="targeting",], log=TRUE, normalized.lib.sizes=TRUE)
    # noness <- cpm(y[y$genes$type=="non-essential",], log=TRUE, normalized.lib.sizes=TRUE)
    lcpm.m <- rbind(melt(nt),melt(ess))
    classes<-c(rep("NT",dim(nt)[1]*18),rep("targeting",dim(ess)[1]*18))
    value <-lcpm.m$value
    samples <- lcpm.m$Var2
    data <- data.frame(samples,classes,value)
    r <- ggplot(data,aes(fill=samples,y=value,x=classes))+ geom_boxplot()+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("lcpm")
    print(r)
}

#build count matrix
count_mat <-  tab[ , -which(colnames(tab) %in% c("MFDpir-2","ON_KppR1-1"))] ##remove two samples with low reads
groups <- c('MFDpir',  'T0_NTUH', 'T0_NTUH', 'T0_KppR1', 'T0_KppR1', 'T3h_NTUH', 'T3h_NTUH', 'T3h_KppR1', 'T3h_KppR1', 'ON_NTUH', 'ON_NTUH', 'ON_KppR1')
y <- DGEList(count_mat[,15:dim(count_mat)[2]], group=groups, genes=count_mat[,1])
y$genes$type<-count_mat[,14]

#filter by CPM
dim(y)
keep <- rowSums(cpm(y$counts) > 1) >= 2
y <- y[keep, , keep.lib.sizes=FALSE]
dim(y)

# normalize on non-targeting crRNAs
y_n <- y[,,keep.lib.sizes=TRUE]
nt <- y[y$genes$type=="NT",,keep.lib.sizes=TRUE]
# keep <- rowSums(cpm(nt$counts) > 1) >= 18
# nt <- nt[keep, , keep.lib.sizes=TRUE]
dim(nt)
nt <- calcNormFactors(nt, method="TMM",keep.lib.sizes=TRUE)

y_n <- nt
y_n$samples$norm.factors <- nt$samples$norm.factors


pdf(file = "cpm_plots.pdf")
#plots
# unnormalized barplot
lcpm <- cpm(y, log=TRUE, normalized.lib.sizes=TRUE)
boxplot(lcpm[,], las=2, main="",pars=list(par(mar=c(8,4,4,2))))
title(main="Unnormalised data", ylab="Log-cpm")
#normalized bar plot
sink("norm.factors.txt",append = FALSE)
print(y_n$samples)
sink()
lcpm <- cpm(y_n, log=TRUE,normalized.lib.sizes=TRUE)
boxplot(lcpm[,], las=2, main="",pars=list(par(mar=c(8,4,4,2))))
title(main="Normalised with NT", ylab="Log-cpm")
lcpm <- cbind(lcpm,y_n$genes$type)
colnames(lcpm) <- c(colnames(y_n$counts),'type')
write.table(lcpm,file="lcpm_normalized_with_NT.csv",sep = "\t",col.names=NA)

lcpm_group(y,"unnormalized")
lcpm_group(y_n,"normalized")
dev.off()

# library(ggfortify)
pdf(file = "./dispersion.pdf")
colors <- brewer.pal(10,'Paired')


#MDS libraries
plotMDS(y_n,col=colors[as.factor(groups)], cex=0.7)
#design matrix
groups <- factor(groups)
design <- model.matrix(~0+groups)
colnames(design) <- sub("groups","",colnames(design))
# colnames(design) <- make.names(colnames(design))
# dispersions
y_n <- estimateDisp(y_n, design, robust=TRUE)
plotBCV(y_n)
fit <- glmQLFit(y_n, design, robust=TRUE)
plotQLDisp(fit)
dev.off()

### differential abundance analysis
# mean_value<-vector()
# median_value<-vector()
values_targeting <- vector()
values_nc <- vector()
pdf(file = "DE.pdf")
constrasts <- c("(ON_KppR1-T0_KppR1)-(ON_NTUH-T0_NTUH)")
names <- c("KppR1-NTUH")
for (i in c("T0_KppR1-MFDpir","T0_NTUH-MFDpir","T3h_KppR1-T0_KppR1","T3h_NTUH-T0_NTUH","ON_KppR1-T0_KppR1","ON_NTUH-T0_NTUH")){
  str <- i
  cont <- makeContrasts(str, levels=design)
  res <- glmQLFTest(fit, contrast=cont)
  tt <- topTags(res, n=Inf)
  tt_mod <- cbind(guides = tt$table[,1], type=tt$table[,2],tt$table[,3:7],stringsAsFactors=FALSE)
  # write.table(tt_mod,file = paste(names[i],"_QLFTest.csv",sep=""),sep = "\t",row.names = FALSE)
  
  
  ###scatter plot for logFC of crRNAs targeting different genes
  hist(tt_mod$PValue,breaks=100,main = paste("p value distribution (",str,")",sep = ""),xlab = "p value")
  plot(tt_mod[which(tt_mod$type=="targeting"),"logCPM"], tt_mod[which(tt_mod$type=="targeting"),"logFC"], pch=20,main=str, xlab="logCPM", ylab="logFC",col=alpha("darkblue", 0.4))
  points(tt_mod[which(tt_mod$type=="NT"),"logCPM"], tt_mod[which(tt_mod$type=="NT"),"logFC"], pch=20,  xlab="logCPM", ylab="logFC", col="grey")
  # points(tt_mod[grep(",",tt_mod$guides),"logCPM"], tt_mod[grep(",",tt_mod$guides),"logFC"], pch=20, col=alpha("yellow", 0.4))
  legend("bottomright", legend=c("targeting","NT"),col=c("darkblue","grey"), pch=c(20),cex=0.8)
  plot_logFC(tt_mod$guides,tt_mod$logFC,tt_mod$type,str)###Barplot the logFC of different gene groups
  
  # for (j in c("targeting","NT")){
  #   print(dim(tt_mod[tt_mod$type==j,]))
  #   print(mean(tt_mod[tt_mod$type==j,]$logFC))
  #   print(median(tt_mod[tt_mod$type==j,]$logFC))
  #   mean_value<-append(mean_value,mean(tt_mod[tt_mod$type==j,]$logFC),after = length(mean_value))
  #   median_value<-append(median_value,median(tt_mod[tt_mod$type==j,]$logFC),after = length(median_value))
  # }
  # 
  values_targeting <- cbind(values_targeting,tt_mod[which(tt_mod$type=="targeting"),"logFC"])
  colnames(values_targeting)[dim(values_targeting)[2]] <- str
  values_nc <- cbind(values_nc,tt_mod[which(tt_mod$type=="NT"),"logFC"])
  colnames(values_nc)[dim(values_nc)[2]] <- str
  
}

tt_logFC <- rbind(melt(values_targeting),melt(values_nc))
classes<-c(rep("targeting",dim(values_targeting)[1]*dim(values_targeting)[2]),rep("NT",dim(values_nc)[1]*dim(values_nc)[2]))
value <-tt_logFC$value
data <- data.frame(tt_logFC$Var2,classes,value)
data$classes <- as.character(data$classes)
data$classes <- factor(data$classes, levels=c("targeting","NT"))
r <- ggplot(data,aes(fill=tt_logFC$Var2,y=value,x=classes))+ geom_boxplot()+ggtitle("logFC between timepoints")+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5),axis.text.x = element_text(angle = 0))+xlab("")+ylab("logFC")+ylim(-10,10)
print(r)
dev.off()



    
    



