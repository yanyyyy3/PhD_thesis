#setwd("PATH_TO_WORKON")
library("ggplot2")
library("RUVSeq")
library("edgeR")
library("plyr")
library(scales)
library(stringr)
library(reshape2)
library(RColorBrewer)

#Load the crRNA count file
tab <- read.table("sample_counts.csv", sep="\t", header=T, stringsAsFactors = F, quote="\"")
rownames(tab) <- tab$guide_name
colnames(tab)<-c(colnames(tab)[1:3],"nt_1","nt_2","targeting_1","targeting_2")
gene <- vapply(strsplit(unlist(strsplit(tab$guide_name,",")),"_"),'[',1,FUN.VALUE=character(1))
gene <- str_replace(gene,"random.*","NC")
gene <- unique(gene)
length(gene)

###check the correlation between samples
pdf(file = "sample_check.pdf")
cormat <- cor(log(tab[,4:length(tab[1,])] + 1))
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
                                                               colour=NA))

print(g)
###Check the counts of crRNAs targeting different gene groups
plot_input <- function(guide_names, counts, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=counts, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("non-essential", "essential", "NC"))
    s1t1$X2 <- as.numeric(counts)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=log(X2+1, base=10), colour=s1t1$X3, fill=s1t1$X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(),axis.title.x=element_blank(),
                                                                                                                                                                     axis.text.x=element_blank(),
                                                                                                                                                                     axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue"))+ ylab("log10 counts") +ggtitle(title)
    print(r)
}
plot_input(tab$guide_name, tab$nt_1,tab$character,"nt_1")
plot_input(tab$guide_name, tab$nt_2,tab$character,"nt_2")
plot_input(tab$guide_name, tab$targeting_1,tab$character,"targeting_1")
plot_input(tab$guide_name, tab$targeting_2,tab$character,"targeting_2")
dev.off()

###Barplot the logFC of different gene groups
plot_logFC <- function(guide_names, logFC, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=logFC, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("non-essential", "essential", "NC"))
    s1t1$X2 <- as.numeric(logFC)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=X2, colour=s1t1$X3, fill=s1t1$X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(), axis.title.x=element_blank(),
                                                                                                                                         axis.text.x=element_blank(), 
                                                                                                                                         axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue"))+ ylab("logFC")+ggtitle(title)                      
    print(r)
}
###Plot the FDR distribution of differential abundance
plot_FDR <- function(guide_names, FDR, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=FDR, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("non-essential", "essential", "NC"))
    s1t1$X2 <- as.numeric(FDR)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=-log(X2, base=10), colour=s1t1$X3, fill=s1t1$X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(), axis.title.x=element_blank(),
                                                                                                                                                        axis.text.x=element_blank(), 
                                                                                                                                                        axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue")) + xlab(length(guide_names))+ ylab("-log10 FDR") + ggtitle(title)                        
    print(r)
}
timepoint_bar <- function(value,title){
    comparison<-"targeting - nt"
    classes<-rep(c("essential","non-essential","NC"),1)
    value <-value
    data <- data.frame(comparison,classes,value)
    r <- ggplot(data,aes(fill=classes,y=value,x=comparison))+ geom_bar(position="dodge", stat="identity")+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+ylab("logFC")
    print(r)
}
lcpm_group <- function(y,title) {
    nt <- cpm(y[y$genes$type=="NC",], log=TRUE, normalized.lib.sizes=TRUE)
    ess<- cpm(y[y$genes$type=="essential",], log=TRUE, normalized.lib.sizes=TRUE)
    noness <- cpm(y[y$genes$type=="non-essential",], log=TRUE, normalized.lib.sizes=TRUE)
    lcpm.m <- rbind(melt(nt),melt(ess),melt(noness))
    classes<-c(rep("NC",dim(nt)[1]*8),rep("essential",dim(ess)[1]*8),rep("non-essential",dim(noness)[1]*8))
    value <-lcpm.m$value
    samples <- lcpm.m$Var2
    data <- data.frame(samples,classes,value)
    r <- ggplot(data,aes(fill=samples,y=value,x=classes))+ geom_boxplot()+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("lcpm")
    print(r)
}
fraction_group <- function(y,title){
    nt <- cpm(y[y$genes$type=="NC",], log=FALSE, normalized.lib.sizes=TRUE)
    ess<- cpm(y[y$genes$type=="essential",], log=FALSE, normalized.lib.sizes=TRUE)
    noness <- cpm(y[y$genes$type=="non-essential",], log=FALSE, normalized.lib.sizes=TRUE)
    cs <- rbind(colSums(nt),colSums(ess),colSums(noness))
    cs <- cbind(library=c(dim(nt)[1],dim(ess)[1],dim(noness)[1]),cs)
    cs <- prop.table(cs,margin = 2) *100
    rownames(cs) <- c("NC","essential","non-essential")
    cs.m <- melt(cs)
    classes <- cs.m$Var1
    data <- data.frame(classes,cs.m$Var2,cs.m$value)
    r <- ggplot(data,aes(fill=classes,y=cs.m$value,x=cs.m$Var2))+ geom_bar(position="fill", stat="identity")+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+xlab("samples")+ylab("percentage of cpm")
    print(r)
    print(cs)
    write.table(as.data.frame(cbind(rownames(cs),cs[,1:5])),file = "./group_fraction.csv",sep = '\t',row.names = FALSE)
}



#build count matrix
cols <- c(paste("nt_",rep(1:2), sep=""), paste("targeting_",rep(1:2), sep=""))
count_mat <- tab[,c("guide_name", "character","KEGG_pathway",cols)]
groups <- c(rep("nt",2), rep("targeting",2))
y <- DGEList(count_mat[,4:7], group=groups, genes=count_mat[,1])
y$genes$type<-count_mat[,2]
y$genes$kegg<-count_mat[,3]


#filter by CPM
dim(y)
keep <- rowSums(cpm(y$counts) > 1) >= 2
y <- y[keep, , keep.lib.sizes=FALSE]
dim(y)

# normalize on non-targeting crRNAs
y_n <- y
nt <- y[y$genes$type=="NC",]
dim(nt)
nt <- calcNormFactors(nt, method="TMM")
y_n$samples$norm.factors <- nt$samples$norm.factors


pdf(file = "cpm_plots.pdf")
#plots
fraction_group(y,"Percentage of counts of different classes")
# unnormalized barplot
lcpm <- cpm(y, log=TRUE, normalized.lib.sizes=TRUE)
boxplot(lcpm[,], las=2, main="")
title(main="Unnormalised data", ylab="Log-cpm")
#normalized bar plot
sink("norm.factors.txt",append = FALSE)
print(y_n$samples)
sink()
lcpm <- cpm(y_n, log=TRUE,normalized.lib.sizes=TRUE)
boxplot(lcpm[,], las=2, main="")
title(main="Normalised data", ylab="Log-cpm")
write.table(lcpm,file="lcpm.csv",sep = "\t",col.names=NA)

lcpm_group(y,"unnormalized")
lcpm_group(y_n,"normalized")
dev.off()


#MDS libraries
pdf(file = "./dispersion.pdf")
plotMDS(y_n)
#design matrix
groups <- factor(groups)
design <- model.matrix(~0+groups)
colnames(design) <- sub("groups","",colnames(design))
# dispersions
y_n <- estimateDisp(y_n, design, robust=TRUE)
plotBCV(y_n)
fit <- glmQLFit(y_n, design, robust=TRUE)
plotQLDisp(fit)
dev.off()

### differential abundance analysis
pdf(file = "DE.pdf")
str <- "targeting-nt"
cont <- makeContrasts(str, levels=design)
res <- glmQLFTest(fit, contrast=cont)
tt <- topTags(res, n=Inf)
tt_mod <- cbind(guides = tt$table[,1], type=tt$table[,2], kegg=tt$table[,3],tt$table[,4:8],stringsAsFactors=FALSE)
write.table(tt_mod,file = paste(str,"_QLFTest.csv",sep=""),sep = "\t",row.names = FALSE)


###scatter plot for logFC of crRNAs targeting different genes
hist(tt_mod$PValue,breaks=100,main = paste("p value distribution (",str,")",sep = ""),xlab = "p value")
plot(tt_mod[which(tt_mod$type=="non-essential"),"logCPM"], tt_mod[which(tt_mod$type=="non-essential"),"logFC"], pch=20, main=str, xlab="logCPM", ylab="logFC", col="grey")
points(tt_mod[which(tt_mod$type=="NC"),"logCPM"], tt_mod[which(tt_mod$type=="NC"),"logFC"], pch=20, col=alpha("black", 0.4))
points(tt_mod[which(tt_mod$type=="essential"),"logCPM"], tt_mod[which(tt_mod$type=="essential"),"logFC"], pch=20, col=alpha("red", 0.4))
points(tt_mod[grep("rrf|rrl|rrs",tt_mod$guides),"logCPM"], tt_mod[grep("rrf|rrl|rrs",tt_mod$guides),"logFC"], pch=20, col=alpha("blue", 1))
points(tt_mod[grep(",",tt_mod$guides),"logCPM"], tt_mod[grep(",",tt_mod$guides),"logFC"], pch=20, col=alpha("yellow", 0.4))
legend("bottomright", legend=c("non-essential","essential","NC","rRNA","duplicated"),col=c("grey","red","black","blue","yellow"), pch=c(20),cex=0.8)
plot_logFC(tt_mod$guides,tt_mod$logFC,tt_mod$type,str)###Barplot the logFC of different gene groups

###mean and median logFC of crRNA for each gene group###
mean_value<-vector()
median_value<-vector()
for (j in c("essential","non-essential","NC")){
    print(dim(tt_mod[tt_mod$type==j,]))
    print(mean(tt_mod[tt_mod$type==j,]$logFC))
    print(median(tt_mod[tt_mod$type==j,]$logFC))
    mean_value<-append(mean_value,mean(tt_mod[tt_mod$type==j,]$logFC),after = length(mean_value))
    median_value<-append(median_value,median(tt_mod[tt_mod$type==j,]$logFC),after = length(median_value))
}    
timepoint_bar(mean_value,paste("mean logFC between timepoints (",substr(str,1,1)," enzyme)",sep=""))
timepoint_bar(median_value,paste("median logFC between timepoints (",substr(str,1,1)," enzyme)",sep=""))
###violin plot and boxplot of logFC for crRNAs in each gene group
values_ess <- cbind(tt_mod[which(tt_mod$type=="essential"),"logFC"])
colnames(values_ess)[dim(values_ess)[2]] <- str
values_noness <- cbind(tt_mod[which(tt_mod$type=="non-essential"),"logFC"])
colnames(values_noness)[dim(values_noness)[2]] <- str
values_nc <- cbind(tt_mod[which(tt_mod$type=="NC"),"logFC"])
colnames(values_nc)[dim(values_nc)[2]] <- str
values_rrna <-  cbind(tt_mod[grep("rrf|rrl|rrs",tt_mod$guides),"logFC"])
colnames(values_rrna)[dim(values_rrna)[2]] <- str
tt_logFC <- rbind(melt(values_nc),melt(values_ess),melt(values_noness),melt(values_rrna))
classes<-c(rep("NC",dim(values_nc)[1]*dim(values_nc)[2]),rep("essential",dim(values_ess)[1]*dim(values_ess)[2]),rep("non-essential",dim(values_noness)[1]*dim(values_noness)[2]),rep("rRNA",dim(values_rrna)[1]*dim(values_rrna)[2]))
value <-tt_logFC$value
data <- data.frame(tt_logFC$Var2,classes,value)
r <- ggplot(data,aes(y=value,x=classes))+ geom_violin(trim=FALSE)+stat_summary(fun.data=mean_sdl, mult=1,geom="pointrange", color="red")+ggtitle(str)+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("logFC")+ylim(-10,10)
print(r)
r <- ggplot(data,aes(y=value,x=classes))+ geom_boxplot()+ggtitle(str)+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("logFC")+ylim(-10,10)
print(r)
dev.off()

    
    



