
library("ggplot2")
library("RUVSeq")
library("edgeR")
library("plyr")
library(scales)
library(stringr)
library(reshape2)
library(RColorBrewer)
# library("forcats")
tab <- read.table("sample_counts.csv", sep="\t", header=T, stringsAsFactors = F, quote="\"")
rownames(tab) <- tab$guide_name

gene <- vapply(strsplit(unlist(strsplit(tab$guide_name,",")),"_"),'[',1,FUN.VALUE=character(1))
gene <- str_replace(gene,"random.*","NC")
gene <- unique(gene)
length(gene)

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

plot_input <- function(guide_names, counts, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=counts, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("non-essential", "essential", "NC"))
    s1t1$X2 <- as.numeric(counts)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=log(X2+1, base=10), colour=s1t1$X3, fill=s1t1$X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(),axis.title.x=element_blank(),
                                                                                                                                                                     axis.text.x=element_blank(),
                                                                                                                                                                     axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue"))+ ylab("log10 counts") +ggtitle(title)
    print(r)
}
plot_input(tab$guide_name, tab$s1t1,tab$character,"s1t1")
plot_input(tab$guide_name, tab$s2t1,tab$character,"s2t1")
plot_input(tab$guide_name, tab$e1t1,tab$character,"e1t1")
plot_input(tab$guide_name, tab$e2t1,tab$character,"e2t1")
dev.off()


plot_logFC <- function(guide_names, logFC, classes,title){
    s1t1 <- data.frame(cbind(X1=guide_names, X2=logFC, X3=classes))
    s1t1$X3 <- factor(classes, levels=c("non-essential", "essential", "NC"))
    s1t1$X2 <- as.numeric(logFC)
    r <- ggplot(s1t1, aes(x=reorder(X1, -X2), y=X2, colour=s1t1$X3, fill=s1t1$X3)) + geom_bar(stat="identity") + theme_classic() + theme(legend.title = element_blank(), axis.title.x=element_blank(),
                                                                                                                                         axis.text.x=element_blank(), 
                                                                                                                                         axis.line.x =element_blank(), axis.line.y =element_blank() ,axis.ticks.x=element_blank(),plot.title = element_text(hjust = 0.5)) + scale_color_manual(values = c("black", "red", "blue")) + scale_fill_manual(values = c("black", "red", "blue"))+ ylab("logFC")+ggtitle(title)                      
    print(r)
}

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
    timepoints<-c(rep("t1-t2",3),rep("t1-t3",3),rep("t1-t4",3))
    classes<-rep(c("essential","non-essential","NC"),3)
    value <-value
    data <- data.frame(timepoints,classes,value)
    r <- ggplot(data,aes(fill=classes,y=value,x=timepoints))+ geom_bar(position="dodge", stat="identity")+ggtitle(title)+theme(plot.title = element_text(hjust = 0.5))+ylab("logFC")
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
    # sink(file='./fraction_percentage.txt',append = TRUE)
    print(cs)
    # sink()
    write.table(as.data.frame(cbind(rownames(cs),cs)),file = "./group_fraction.csv",sep = '\t',row.names = FALSE)
}
cpm_logFC <- function(y,title){
    cpm_cols <- c(paste(colnames(y$counts)[2],colnames(y$counts)[1],sep="-"),paste(colnames(y$counts)[3],colnames(y$counts)[1],sep="-"),paste(colnames(y$counts)[4],colnames(y$counts)[1],sep="-"),paste(colnames(y$counts)[6],colnames(y$counts)[5],sep="-"),paste(colnames(y$counts)[7],colnames(y$counts)[5],sep="-"),paste(colnames(y$counts)[8],colnames(y$counts)[5],sep="-"))
    print(cpm_cols)
    nt <- cpm(y[y$genes$type=="NC",], log=TRUE, normalized.lib.sizes=TRUE, prior.count=2)
    nt <- cbind(nt[,2]-nt[,1],nt[,3]-nt[,1],nt[,4]-nt[,1],nt[,6]-nt[,5],nt[,7]-nt[,5],nt[,8]-nt[,5])
    colnames(nt) <- cpm_cols
    ess<- cpm(y[y$genes$type=="essential",], log=TRUE, normalized.lib.sizes=TRUE, prior.count=2)
    ess <- cbind(ess[,2]-ess[,1],ess[,3]-ess[,1],ess[,4]-ess[,1],ess[,6]-ess[,5],ess[,7]-ess[,5],ess[,8]-ess[,5])
    colnames(ess) <- cpm_cols
    noness <- cpm(y[y$genes$type=="non-essential",], log=TRUE, normalized.lib.sizes=TRUE, prior.count=2)
    noness <- cbind(noness[,2]-noness[,1],noness[,3]-noness[,1],noness[,4]-noness[,1],noness[,6]-noness[,5],noness[,7]-noness[,5],noness[,8]-noness[,5])
    colnames(noness) <- cpm_cols
    cpm.log2FC <- rbind(melt(nt),melt(ess),melt(noness))
    classes<-c(rep("NC",dim(nt)[1]*6),rep("essential",dim(ess)[1]*6),rep("non-essential",dim(noness)[1]*6))
    value <-cpm.log2FC$value
    samples <- cpm.log2FC$Var2
    data <- data.frame(samples,classes,value)
    r <- ggplot(data,aes(fill=samples,y=value,x=classes))+ geom_boxplot()+ggtitle(paste(title,"cpm log2FC between time points",sep=" "))+theme(plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("cpm log2FC")
    print(r)
    d <- ggplot_build(r)
    # log2FC_mean <- rbind(as.matrix(apply(nt,2,mean)),as.matrix(apply(ess,2,mean)),as.matrix(apply(noness,2,mean)))
    # log2FC_mean <- log(2**log2FC_mean-1,base=2)
    log2FC_mean <- data.frame(aggregate(value,list("samples" = cpm.log2FC$Var2,'classes' = data$classes) ,  mean))
    r <- ggplot(log2FC_mean,aes(fill=log2FC_mean$samples,y=log2FC_mean$x,x=log2FC_mean$classes))+ geom_bar(position="dodge", stat="identity")+ggtitle(paste(title,"mean of cpm log2FC",sep=" "))+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("cpm log2FC")
    print(r)

    log2FC_median <- data.frame(aggregate(value,list("samples" = cpm.log2FC$Var2,'classes' = data$classes) ,  median))
    # log2FC_median <- log(2**log2FC_median-1,base=2)
    # classes<-c(rep("NC",6),rep("essential",6),rep("non-essential",6))
    # data <- data.frame(log2FC_median[,1],classes,rownames(log2FC_median))
    r <- ggplot(log2FC_median,aes(fill=log2FC_median$samples,y=log2FC_median$x,x=log2FC_median$classes))+ geom_bar(position="dodge", stat="identity")+ggtitle(paste(title,"median of cpm log2FC",sep=" "))+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("cpm log2FC")
    print(r)
}

#build count mat e enzyme
# cols <- c(paste("e1t",rep(1:4), sep=""), paste("e2t",rep(1:4), sep=""))
# e_count_mat <- tab[,c("guide_name", "character","KEGG_pathway",cols)]
# groups <- c(paste("et",rep(1:4), sep=""), paste("et",rep(1:4), sep=""))
# batch <- c(rep(1, times=4), rep(2, times=4))
# y <- DGEList(e_count_mat[,4:11], group=groups, genes=e_count_mat[,1])
# y$genes$type<-e_count_mat[,2]
# y$genes$kegg<-e_count_mat[,3]
#build count mat s enzyme
cols <- c(paste("s1t",rep(1:4), sep=""), paste("s2t",rep(1:4), sep=""))
s_count_mat <- tab[,c("guide_name", "character","KEGG_pathway",cols)]
groups <- c(paste("st",rep(1:4), sep=""), paste("st",rep(1:4), sep=""))
batch <- c(rep(1, times=4), rep(2, times=4))
y <- DGEList(s_count_mat[,4:11], group=groups, genes=s_count_mat[,1])
y$genes$type<-s_count_mat[,2]
y$genes$kegg<-s_count_mat[,3]
# cols <- c(paste("s1t",rep(1:3), sep=""), paste("s2t",rep(1:3), sep=""))
# s_count_mat <- tab[,c("guide_name", "character",cols)]
# groups <- c(paste("st",rep(1:3), sep=""), paste("st",rep(1:3), sep=""))
# batch <- c(rep(1, times=3), rep(2, times=3))
# y <- DGEList(s_count_mat[,3:8], group=groups, genes=s_count_mat[,1])
# y$genes$type<-s_count_mat[,2]

#filter by cpm
dim(y)
keep <- rowSums(cpm(y$counts) > 1) >= 2
y <- y[keep, , keep.lib.sizes=FALSE]
dim(y)


# try norm on non-targeting
y_n <- y
nt <- y[y$genes$type=="NC",]
dim(nt)
nt <- calcNormFactors(nt, method="TMM")
y_n$samples$norm.factors <- nt$samples$norm.factors

#ruvseq
nt <- rownames(y)[grep("^random", rownames(y))]
set <- newSeqExpressionSet(as.matrix(y),phenoData = data.frame(groups, row.names=colnames(y)))
set <- betweenLaneNormalization(set, which="upper")
colors <- brewer.pal(4, "Set2")
pdf(file="./ruvseq.pdf")
plotRLE(set, outline=FALSE, col=colors[groups])
plotPCA(set, col=colors[groups], cex=1.2)


#RUVs
differences <- makeGroups(groups)
set1 <- RUVs(set, nt, k=1, differences)
plotRLE(set1, outline=FALSE, ylim=c(-4, 4), col=colors[groups])
plotPCA(set1, col=colors[groups], cex=1.2)
pData(set1)
dev.off()

design <- model.matrix(~0+groups + W_1, data=pData(set1))
colnames(design) <- c(groups[1:4],"W_1")
sink('ruv.txt')
print(pData(set1))
print(design)
sink()


pdf(file = "cpm_plots.pdf")
#plots
fraction_group(y,"Percentage of counts of different classes")
# unnormalized barplot
lcpm <- cpm(y, log=TRUE, normalized.lib.sizes=TRUE)
boxplot(lcpm[,c(1,5,2,6,3,7,4,8)], las=2, main="")
title(main="Unnormalised data", ylab="Log-cpm")
#normalized bar plot
sink("norm.factors.txt",append = FALSE)
print(y_n$samples)
sink()
lcpm <- cpm(y_n, log=TRUE,normalized.lib.sizes=TRUE)
boxplot(lcpm[,c(1,5,2,6,3,7,4,8)], las=2, main="")
title(main="Normalised data", ylab="Log-cpm")
write.table(lcpm,file="lcpm.csv",sep = "\t",col.names=NA)

lcpm_group(y,"unnormalized")
lcpm_group(y_n,"normalized")
cpm_logFC(y,"unnormalized")
cpm_logFC(y_n,"normalized")
dev.off()


#MDS libraries
pdf(file = "./dispersion.pdf")
plotMDS(y_n)
y_n <- estimateDisp(y_n, design, robust=TRUE)
plotBCV(y_n)
fit <- glmQLFit(y_n, design, robust=TRUE)
plotQLDisp(fit)
dev.off()

# differential expression analysis
mean_value<-vector()
median_value<-vector()
values_ess <- vector()
values_noness <- vector()
values_nc <- vector()
pdf(file = "DE.pdf")
for (i in c(2,3,4)){
    # str <- paste("et",i,"-et1",sep="")
    str <- paste("st",i,"-st1",sep="")
    cont <- makeContrasts(str, levels=design)
    res <- glmQLFTest(fit, contrast=cont)
    tt <- topTags(res, n=Inf)
    
    tt_mod <- cbind(guides = tt$table[,1], type=tt$table[,2], kegg=tt$table[,3],tt$table[,4:8],stringsAsFactors=FALSE)
    write.table(tt_mod,file = paste(str,"_QLFTest.csv",sep=""),sep = "\t",row.names = FALSE)
    
    values_ess <- cbind(values_ess,tt_mod[which(tt_mod$type=="essential"),"logFC"])
    colnames(values_ess)[dim(values_ess)[2]] <- str
    values_noness <- cbind(values_noness,tt_mod[which(tt_mod$type=="non-essential"),"logFC"])
    colnames(values_noness)[dim(values_noness)[2]] <- str
    values_nc <- cbind(values_nc,tt_mod[which(tt_mod$type=="NC"),"logFC"])
    colnames(values_nc)[dim(values_nc)[2]] <- str
    
    hist(tt_mod$PValue,breaks=100,main = paste("p value distribution (",str,")",sep = ""),xlab = "p value")
    plot(tt_mod[which(tt_mod$type=="non-essential"),"logCPM"], tt_mod[which(tt_mod$type=="non-essential"),"logFC"], pch=20, main=str, xlab="logCPM", ylab="logFC", col="grey")
    points(tt_mod[which(tt_mod$type=="NC"),"logCPM"], tt_mod[which(tt_mod$type=="NC"),"logFC"], pch=20, col=alpha("black", 0.4))
    points(tt_mod[which(tt_mod$type=="essential"),"logCPM"], tt_mod[which(tt_mod$type=="essential"),"logFC"], pch=20, col=alpha("red", 0.4))
    points(tt_mod[grep("araC",tt_mod$guides),"logCPM"], tt_mod[grep("araC",tt_mod$guides),"logFC"], pch=20, col=alpha("blue", 1))
    points(tt_mod[grep("lacY",tt_mod$guides),"logCPM"], tt_mod[grep("lacY",tt_mod$guides),"logFC"], pch=20, col=alpha("yellow", 1))
    points(tt_mod[grep("lacZ",tt_mod$guides),"logCPM"], tt_mod[grep("lacZ",tt_mod$guides),"logFC"], pch=20, col=alpha("green", 1))
    points(tt_mod[grep("lacI",tt_mod$guides),"logCPM"], tt_mod[grep("lacI",tt_mod$guides),"logFC"], pch=20, col=alpha("orange", 1))
    # points(tt_mod[grep(",",tt_mod$guides),"logCPM"], tt_mod[grep(",",tt_mod$guides),"logFC"], pch=20, col=alpha("yellow", 1))
    
    legend("bottomright", legend=c("non-essential","essential","NC","araC","lacY","lacZ","lacI"),col=c("grey","red","black","blue", "yellow","green","orange"), pch=c(20),cex=0.8)
    #     #points(sel_tab$dTA_TE, sel_tab$TA_TE, pch=21, col="red")
    plot_logFC(tt_mod$guides,tt_mod$logFC,tt_mod$type,str)
    
    for (j in c("essential","non-essential","NC")){
        print(dim(tt_mod[tt_mod$type==j,]))
        print(mean(tt_mod[tt_mod$type==j,]$logFC))
        print(median(tt_mod[tt_mod$type==j,]$logFC))
        mean_value<-append(mean_value,mean(tt_mod[tt_mod$type==j,]$logFC),after = length(mean_value))
        median_value<-append(median_value,median(tt_mod[tt_mod$type==j,]$logFC),after = length(median_value))
    }
}
timepoint_bar(mean_value,paste("mean logFC between timepoints (",substr(str,1,1)," enzyme)",sep=""))
timepoint_bar(median_value,paste("median logFC between timepoints (",substr(str,1,1)," enzyme)",sep=""))

tt_logFC <- rbind(melt(values_nc),melt(values_ess),melt(values_noness))
classes<-c(rep("NC",dim(values_nc)[1]*dim(values_nc)[2]),rep("essential",dim(values_ess)[1]*dim(values_ess)[2]),rep("non-essential",dim(values_noness)[1]*dim(values_noness)[2]))
value <-tt_logFC$value
data <- data.frame(tt_logFC$Var2,classes,value)
r <- ggplot(data,aes(fill=tt_logFC$Var2,y=value,x=classes))+ geom_boxplot()+ggtitle(paste("logFC between timepoints (",substr(str,1,1)," enzyme)",sep=""))+theme(legend.title = element_blank(),plot.title = element_text(hjust = 0.5))+xlab("classes")+ylab("logFC")+ylim(-10,10)
print(r)
dev.off()
    



