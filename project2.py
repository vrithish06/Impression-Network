import pandas as pd 
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#reading xlsx file in python.
dataframe = pd.read_excel('Impression_network.xlsx')

#storing the emails in var x. 
x=pd.DataFrame(dataframe,columns=["Email Address"])

#convert dataframe to list.
x=x.values.tolist()
nodes=[]

#those list values to unq,redable ids.
for i in x :
   nodes.append(i[0][0:4]+i[0][7:11])
#print(nodes)    

edge=[]  #all edges will be store in the list edge.
row = dataframe.loc[[i for i in range(0,133)]]   #reading rows .
row = row.values.tolist()    #converting those to list format for further calculations.
for i in range(0,len(row)):   #for every row we have total 32 columns stored in k in each loop.
    k=row[i]
    p=[]
    for j in range(2,len(k)):     #but edge of an email we will leave the time and email column. 
        if type(k[j]) == str :
            p.append(k[j][-11:-7]+k[j][-4: ]) #and appending unq ids in  list p.
    edge.append(p)                 #and appending those unq ids of each email id in edge list.
#print(edge)    

#now we have nodes in nodes as a list and edges of the nodes respectively in edge as a list in list.
#and we now have to run random walk on this by a random walk with a directional graph.
G=nx.DiGraph()
G.add_nodes_from(nodes)
for i in range(len(nodes)):
    for j in range (len(edge[i])):
        G.add_edge(nodes[i],edge[i][j])
#nx.draw(G,with_labels=True)
#plt.show()

def out(node):
    return edge[nodes.index(node)]

#now we have our graph moving on to the next part that is RANDOM WALK. 
def random_walk(G):    
    rand=[]
    points=[0 for i in range(len(G.nodes()))]
    s=random.choice(nodes)                    #selecting the random node to start with.
    while sum(points)<1000000:                #limiting the walk's steps.
        points[nodes.index(s)]=points[nodes.index(s)]+1       #increse a point for each step.
        r=random.random()                        
        if r > 0.15 :                          #if probability is greater than 0.15 we will see for any edges.
            if edge[nodes.index(s)]!= [0]:     #if edges are present .
                e=random.choice(edge[nodes.index(s)])   #selecting a random neighbour.
                if e in nodes:                 #if that randomm neighbour present in nodes.
                    s=e                        #then we will move to that neighbour.
                if e not in nodes:             #but if that random neighbour is not there in the nodes initially .
                    nodes.append(e)            #then edge will append 0 neighbours for it .
                    edge.append([0])           #and points will be zero for it initially.
                    points.append(0)           
                    rand.append(e)             #appending those edges which are not present in the nodes initially.
            else:
                s=random.choice(nodes)         #if there are no edges then teleporting.
        else:
            s=random.choice(nodes)             #if probability is less than 0.15 then teleport although there are edges for s .
    page_rank=[]
    for i in range(len(nodes)):
        page_rank.append([nodes[i],points[i]])
    page_rank=sorted(page_rank , key=lambda x: x[1])    #sorting the list of nodes and their points based on the 2nd element (points) in sublists.
    pr=[]
    for i in page_rank[::-1]:                     #for ascending order of page ranks.
        pr.append(i[0])                           
    return pr

#random_walk(G)

def kal(G):
    edges=[]
    for i in G.edges():
        edges.append(list(i)) 
    sym=[]
    unsym=[]
    for i in edges:
        if i[::-1] in edges:
            sym.append(i)
        else:
            unsym.append(i)
    return len(sym) , len(unsym) , len(edges)

#print(kal(G)[0] , kal(G)[1] , kal(G)[2])


def MeasureNonIdeal(G):
    values=[0]*len(nodes)

    for i in edge:
        for out in i:
            if out!=0 and out in nodes:
                values[nodes.index(out)]+=1
    imp=[]
    for i in range(len(nodes)):
        imp.append([nodes[i] , values[i]])
      
    imp=sorted(imp ,key=lambda x: x[1])
    imp=imp[::-1]
    V=imp[0][1]
    peop=[]
    freq=[0]*20
    p=5/100

    for j in range(0,20):
        k=[]
        for i in imp:
            if j*p*V<i[1]<=(j+1)*p*V :
                freq[j]+=1
                k.append(i[0])           
        peop.append(k)
    print(freq)
    print("Mean frequency :", np.mean(freq), "\nStandard deviation :",np.std(freq)) 
    print("Violation as observed from the graph:", peop)

    X=[]
    for i in range(20):
        X.append(i)
    plt.plot(X,freq)
    plt.bar(np.arange(len(freq)), freq, align='center', alpha=0.7)
    plt.title('Histogram of Frequency Distribution')
    plt.xticks(np.arange(len(freq)))
    plt.show()
    
    
#MeasureNonIdeal(G)

def adjacency_matrix(G):
    return nx.adjacency_matrix(G).toarray()

def A(G,i,j):
    k=adjacency_matrix(G)
    k=np.delete(k,i, axis=0)
    k=np.delete(k,j, axis=1)
    return k 

def B(G,i,j):
    k=adjacency_matrix(G)
    k=np.delete(k[i],j)
    return k

def X(G, i, j):
    A_matrix = A(G, i, j)
    B_vector = B(G, i, j)
    X = np.linalg.lstsq(A_matrix, B_vector, rcond=None)[0]
    return X

def C(G,i,j):
    k=adjacency_matrix(G)
    k=k[:,j]
    k=np.delete(k,i)
    return k

def value(G,i,j):
    return np.dot(C(G,i,j),X(G,i,j))

def Missing_edges(G):
    M=G.copy()
    missing_edges=[]
    for i in range(len(G.nodes())):
        for j in range(len(G.nodes())):
            if adjacency_matrix(G)[i][j] == 0:
                if value(M,i,j) > 0:
                    M.add_edge(i,j)
                    missing_edges.append((i,j))
    return missing_edges


Missing_edges(G)


                    
    





















