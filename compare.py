#!/usr/bin/env python
# coding=utf-8

import Orange
import sys, os
import csv
from pygooglechart import Chart
from pygooglechart import SimpleLineChart
from pygooglechart import Axis

EPSILON = 0.00000000000001
PATH = "resultats/"
####################################################################################
#Ecriture disque
def writeResults(filename, resMatrix):
    result_matrix = [];
    header = [];
    #Pour chaque colonne j'ai mon cluster expérimenté
    for i in range(len(resMatrix[0])+1):
        if i>0:
            header.append("Cluster "+str(i))
        else:
            header.append("")
    result_matrix.append(header);
    
    #Pour chaque ligne j'ai un label de cluster referen
    for i,item1 in enumerate(resMatrix):
        temp = []
        temp.append("Reference cluster "+str(i+1))
        for j,item2 in enumerate(item1):
            temp.append(item2)
        result_matrix.append(temp)

    #Ecriture dans le fichier
    with open(filename, 'wb') as test_file:
        file_writer = csv.writer(test_file)
        file_writer.writerows(result_matrix)
    

def loadCSVasListOfCLusters(path):
    clusters = [];
    i=0;
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            clusters.append(row);
            i = i+len(row);
    return  clusters, i;

# Fonction qui charge en mémoire une matrice de distance stockée dans un fichier
# paramètre : matrixPath - Chemin vers le fichier
# retourne : les labels correspondant aux indices de la matrice de distance, la matrice de distancechargée dans un objet Orange
def loadDistanceMatrix(matrixPath):
    #print "Loading distance matrix"
    i=0
    j=0
    label = []
    buf = []
    distance = []
    
    with open(matrixPath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            if i > 0:
                j=0
                buf = []
                for item in row:
                    if j == 0:
                        label.append(item);
                    else:
                        buf.append(item);
                    j += 1
                distance.append(buf)
            i += 1;

    #On remplit les valeurs dans une matrice symMatrix utilisés par orange pour le clustering
    orangeDistance = Orange.misc.SymMatrix(len(distance));
    for i in range(len(distance)):
        row = distance[i]
        for j in range(len(row)):
            orangeDistance[i,j] = row[j]
    return label, orangeDistance

####################################################################################
#Calculs le score de précision de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant la precision (IR)
def precision(clusterReference, clusterPredit):
    tp=0.0;
    fp=0.0;
    
    for item1 in clusterPredit:
        if item1 in clusterReference:
            tp+=1.0;
        else:
            fp+=1.0;
    
    precision = (tp/(tp+fp))
    return precision

#Calculs le score de rappel de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant le rappel (IR)
def rappel(clusterReference, clusterPredit):
    tp=0.0;
    fn=0.0;
    
    for item1 in clusterReference:
        if item1 in clusterPredit:
            tp+=1.0;
        else:
            fn+=1.0;
    
    recall = (tp/(tp+fn))
    return recall;


#Calculs la fMesure de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant la fmesure (IR)
def fMesure(clusterReference, clusterPredit):
    precis = precision(clusterReference,clusterPredit)
    rapp = rappel(clusterReference, clusterPredit)
    
    return 2.0*((precis*rapp)/(precis+rapp+EPSILON))
####################################################################################
#Tests
def tests():
    listA = ["a", "b", "c"]
    listA2 = ["a", "c"]
    listB = ["a", "b", "c", "d"]
    listC = ["e", "f", "g"]
        
    assert precision(listA, listA) == 1.0
    assert precision(listA, listA2) == 1.0
    assert precision(listA, listB) == 0.75
    assert precision(listA, listC) == 0.0
    
    
    assert rappel(listA, listB) == 1.0
    assert rappel(listA, listA2) == 0.6666666666666666
    
    
    print "Tests ok"
    
####################################################################################
#
#Fonction qui effectue un CAH à partir d'une matrice de distance
# Paramètre : distanceMatrix : L'objet orange matrice de distance
# Retourne : un objet orange, représentant le résultat de la CAH 
def clusterDistances(distanceMatrix):
    #print "Clustering the distance matrix"
    clustering = Orange.clustering.hierarchical.HierarchicalClustering()
    clustering.linkage = Orange.clustering.hierarchical.WARD
    return clustering(distanceMatrix)




#Fonction qui représente sous forme de dendogramme les 3 premiers clusters
# Paramètres : root - le noeud racine de l'arbre de la CAH
def showTroisGroupes(root):
    topmost = sorted(Orange.clustering.hierarchical.top_clusters(root, 3), key=len)

    for n, cluster in enumerate(topmost):
        print "\n\n Cluster %i \n" % n
        for instance in cluster:
            print labels[instance]
    print
    
    my_colors = [(255,0,0), (0,255,0), (0,0,255)]
    colors = dict([(cl, col) for cl, col in zip(topmost, my_colors)])
    
    Orange.clustering.hierarchical.dendrogram_draw("hclust-dendrogram.png", root, labels=labels, cluster_colors=colors, color_palette=[(0,255,0), (0,0,0), (255,0,0)],
    gamma=0.5, minv=2.0, maxv=7.0)


# Fonction qui permet de générer un tableau CSV de précisions/rappels/FMesure pour chaque cluster de référence et prédit passés en paramètre. et retourne ce tableau
# En effet, le résultat est alors sous la forme :
# |     | predit 1              | predit ... | predit n
# |ref 1|(pres1,rap1,fmesure1)  |  ...       | ...
# | ... |
# |ref n|
def generateResults(filename, clusterReference, clusterPredit):
    i = 0;
    j = 0;
    result_matrix = [[0 for x in xrange(len(clusterPredit))] for x in xrange(len(clusterReference))];#En ligne les clusters référent, en colonne les cluster prédit de la CAH
    
    for i in range(len(clusterReference)):
        for j in range(len(clusterPredit)):
            currPrec = precision(clusterReference[i], clusterPredit[j])
            currRapp = rappel(clusterReference[i], clusterPredit[j])
            currFMes = fMesure(clusterReference[i], clusterPredit[j])
            result_matrix[i][j] = (currPrec, currRapp, currFMes)
    #print "** Ecriture du fichier :"+str(filename)
    writeResults(filename, result_matrix);
    return result_matrix


def printHierarchicalCluster(hc,labels):
    for n, cluster in enumerate(hc):
        print "\n\n Cluster %i \n" % n
        for instance in cluster:
            print instance
    print

def clustersIdToClustersLabel(cluster, labels):
    res = [];
    for n, cluster in enumerate(cluster):
        res.append([])
        for instance in cluster:
            res[n].append(labels[instance])
    return res;

# Fonction qui calcule la moyenne des FMesure sur notre matrice.
# Pour chaque cluster de référence, chercher le cluster prédit qui maximise la FMesure et l'éliminer de la liste des clusters disponnible
# Dans l'idéal si on trouve qu'un cluster déjà éliminé maximise encore un cluster de référence. On ira vérifier si la valeur est alors plus grande et on switchera alors
# Ca m'a lair compliqué.
def getFMesureAVG(result_matrix):
    fMesureMoyTemp = 0.0
    nbComparaison = 0
    eliminatedClust = []#à terme stocker [(ieme, FmesureVal)]
    for item1 in result_matrix:
        maxFmesure = 0.0
        for i,item2 in enumerate(item1):
            if item2[2] > maxFmesure and i not in eliminatedClust:
                maxFmesure = float(item2[2])
                fMesureMoyTemp = float(item2[2])#Récupération de la FMesure dans le tuple (precision,rappel, fmesure)
                nbComparaison += 1
                eliminatedClust.append(i)
            #else si FMesure supérieure
    return float(fMesureMoyTemp/nbComparaison)

def generateGraph(fMesureGlobalPerCent, nbComparison, resPath):
    max_y = 100#Fmesure à 100%
    chart = SimpleLineChart(500, 500, y_range=[0, max_y])
    chart.add_data(fMesureGlobalPerCent)
    chart.set_colours(['0000FF'])
    chart.fill_linear_stripes(Chart.CHART, 0, 'CCCCCC', 0.2, 'FFFFFF', 0.2)
    chart.set_grid(0, 25, 5, 5)
   
    left_axis = range(0, max_y + 1, 25)
    left_axis[0] = ''
    chart.set_axis_labels(Axis.LEFT, left_axis)


    y_axis = []
    for x in range(nbComparison):
        if x%5 == 0:
            y_axis.append(x)
    chart.set_axis_labels(Axis.BOTTOM, y_axis)
    chart.download(resPath+'FMesure-evolution.png')

def main():
    print "** Chargement de clusters de références"
    refClusters, i = loadCSVasListOfCLusters(sys.argv[1])
    print"Reference : Loaded "+str(len(refClusters))+" clusters and "+str(i)+" concepts."
    
    distancesFile = sys.argv[2::]
    for i, df in enumerate(distancesFile):
        print "** PROCESSING  DISTANCE FILE : "+df
        #Create result directory if not exists
        resPath = str(i)+"_"+PATH
        if not os.path.exists(resPath):
            os.makedirs(resPath)
            
        labels, matrix_distance = loadDistanceMatrix(df)
        root = clusterDistances(matrix_distance)
        #Pour chaque groupement possible faire topmost
        fMesureGlobalPerCent = []
        for i in range(len(matrix_distance[0])):
            clusters = sorted(Orange.clustering.hierarchical.top_clusters(root,i), key=len);
            clusters = clustersIdToClustersLabel(clusters,labels)
            result_matrix = generateResults(resPath+"Precision_Rappel"+str(i)+".csv", refClusters, clusters)
            fMesureGlobalPerCent.append(float(getFMesureAVG(result_matrix)*100))
        print "maximum is at index :"+str(fMesureGlobalPerCent.index(max(fMesureGlobalPerCent)))
        
        #Génération d'un graphe récapitulatif de la mesure
        generateGraph(fMesureGlobalPerCent, i, resPath)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        tests()
        main()
    else:
        sys.exit("too few arguments : usage python compare.py <refCluster1> <distanceFile1> <distanceFile2> ...")

#idées :
#   - Ne comparer que coupes qui ont au moin un nombre = au nombre de clusters prédit
#   - Faire une meilleure recherche pour le calcul des FMesure globales
########################################
#Garbage 
    #print "** les TOPS"
    #   On récupère le meilleur candidats pour chaque clusters de références
    #    topsPrecision=[-1 for x in range(len(refClusters))]
    #    topsRecall=[-1 for x in range(len(refClusters))]
    #    topsFMesure = [-1 for x in range(len(refClusters))]
    #    maxPrecision = -1.0;
    #    maxRappel = -1.0;
    #    maxFMesure = -1.0
    #    i=0;
    #    
    #    for ref1 in result_matrix:
    #        j=0;
    #        maxPrecision = -1.0
    #        maxRappel = -1.0
    #        for clust in ref1:
    #            if clust[0] > maxPrecision:
    #                maxPrecision = clust[0]
    #                topsPrecision[i] = (j+1, maxPrecision)
    #            if clust[1] > maxRappel:
    #                maxRappel = clust[1]
    #                topsRecall[i] = (j+1, maxRappel)
    #            if clust[2] > maxFMesure:
    #                maxFMesure = clust[2]
    #                topsFMesure[i] = (j+1, maxFMesure)
    #            j+=1
    #        i+=1
    #
    #    print "Top precision for each cluster ("+str(len(topsPrecision))+") :"+str(topsPrecision)
    #    print "Top recall for each cluster :"+str(topsRecall)
    #    print "Top FMesure for each cluster:"+str(topsFMesure)
