setwd("D:/aaa/singcell")

rm(list = ls())
gc()
library(monocle3)

lpsdata<-read.table("petrolour_scCAEs.csv",header=T,row.names=1,sep=",",check.names=F)

#lpsdata<-read.table("petropoulos.csv",header=T,row.names=1,sep=",",check.names=F)
#data <- as(as.matrix(lpsdata), 'sparseMatrix')

lpsdata<-t(lpsdata)



# cell<-read.table("petrolcell.csv",header=T,row.names=1,sep=",",check.names=F)
# gene<-read.table("petrolgene_raw.csv",header=T,row.names=1,sep=",",check.names=F)

cell<-read.table("petrolcell.csv",header=T,row.names=1,sep=",",check.names=F)
gene<-read.table("petrolgene_reduce.csv",header=T,row.names=1,sep=",",check.names=F)


rownames(cell)<-colnames(data)
rownames(gene)<-rownames(data)
cds <- new_cell_data_set(data,
                         cell_metadata = cell,
                         gene_metadata = gene)

cds <- preprocess_cds(cds, num_dim = 50)
cds <- reduce_dimension(cds)

class.label<-read.table("petrolabel.csv",header=T,row.names=1,sep=",",check.names=F)
class.label<-as.matrix(class.label)
class.label<- class.label[,1]
label=class.label
label<-as.factor(label)
pData(cds)$Time_Point<-label


cds <- cluster_cells(cds)
cds <- learn_graph(cds)

plot_cells(cds, label_groups_by_cluster=FALSE,  color_cells_by = "Time_Point", cell_size = 1.5,graph_label_size = 0,label_cell_groups=FALSE,label_branch_points=FALSE,trajectory_graph_segment_size = 1.2)

cds <- order_cells(cds)

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=0,cell_size = 1.5)






