setwd("D:/aaa/singlecell")
zedata<-read.table("Zeisel.csv",header=T,row.names=1,sep=",",check.names=F)
zedata = t(zedata)
library(Seurat)
ze <- CreateSeuratObject(counts = zedata, project = "zeisel", min.cells = 3, min.features = 200)
ze <- NormalizeData(ze, normalization.method = "LogNormalize", scale.factor = 10000)
all.genes <- rownames(ze)
ze <- ScaleData(ze, features = all.genes)
ze<-RunPCA(ze, features = all.genes)
ze <- RunTSNE(ze, dims = 1:10)
tsne<-ze@reductions[["tsne"]]@cell.embeddings
tsnepc1 =data.frame(var_x=tsne[,1],var_y=tsne[,2])

class.label<- read.table("cluster_Zeisel.csv", header=T,sep=",",check.names=F) 
class.label<-as.matrix(class.label)
class.label<- class.label[,2]
cluster=class.label
cluster<-as.factor(cluster)



p<-ggplot(data=tsnepc1, aes(x=var_x, y=var_y,color=cluster)) + geom_point(size=1)+
  #labs(title="full", x="TSNE1", y="TSNE2")+
  theme(plot.title = element_text(hjust = 0.5))+theme_classic()
#p<-p+theme(legend.position="none")
p<-p+labs(x =NULL,y = NULL)
p<-p+labs(x ="tSNE_1",y = "tSNE_2")
p<-p+theme(legend.position="none")
p<-p+theme(axis.text=element_text(size=10),axis.title = element_text(face = "bold")) + theme(legend.title = element_blank())

p


class.label<- read.table("label_Zeisel.csv", header=T,sep=",",check.names=F) 
class.label<-as.matrix(class.label)
class.label<- class.label[,2]
label=class.label
label<-as.factor(label)


p<-ggplot(data=tsnepc1, aes(x=var_x, y=var_y,color=label)) + geom_point(size=1)+
  #labs(title="full", x="TSNE1", y="TSNE2")+
  theme(plot.title = element_text(hjust = 0.5))+theme_classic()
#p<-p+theme(legend.position="none")
p<-p+labs(x =NULL,y = NULL)
p<-p+labs(x ="tSNE_1",y = "tSNE_2")
p<-p+theme(legend.position="none")
p<-p+theme(axis.text=element_text(size=10),axis.title = element_text(face = "bold")) + theme(legend.title = element_blank())

p




