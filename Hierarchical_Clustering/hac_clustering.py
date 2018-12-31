import tfIdf_vectorConverter as vc
from priority_queue import PriorityQueue
from cluster_node import Cluster_Node, Cluster
import os
import numpy as np

document_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "IRTM")
output_dir = os.path.join(os.path.dirname(__file__), "docVector")
dictionary_file = os.path.join(os.path.dirname(__file__), "dictionary.txt")
c_matrix_file = os.path.join(os.path.dirname(__file__), "sim_matrix.txt")

K = 8

# create the folder if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def compute_sim_matrix(doc_list, output_file=c_matrix_file):
    print("Initializing C...")
    N = len(doc_list)
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            simularity = vc.cosine(os.path.join(output_dir, doc_list[i]), os.path.join(output_dir, doc_list[j]))
            C[i][j], C[j][i] = simularity, simularity
            print("Sim between {} and {}: {}".format(i, j, simularity), end='\r')
    np.savetxt(output_file, C)
    print()
    print("Finished C initialization.")

def efficient_hac(doc_list, C, K):
    '''
    efficient_hac with complete link clustering
    
    N: Total documents
    C[i][j]: the similarity between clusters i and j.
    I: indicate which clusters are still available to be merged.
    A: a list of merges
    P: an array of priority queue
    '''
    
    N = len(doc_list)
    I = [1 for i in range(N)]
    P = []
    A = []

    def complete_link(i, k1, k2):
        return(min(C[i][k1], C[i][k2]))

    def single_link(i, k1, k2):
        return(max(C[i][k1], C[i][k2]))

    # Initialize N, I, C, P, A

    for i in range(N):
        P.append(PriorityQueue())
        for j in range(N):
            if i != j:
                P[i].insert(Cluster_Node(i, j, C[i][j]))
    for i in range(N):
        A.append(Cluster({i}, doc_list))

    # Go through k times clustering to achieve K clusters
    for k in range(N - 1):
        
        # find the largest simulation
        maxSim = Cluster_Node(-1, -1, -1)
        for i in range(N):
            if I[i] == 1:
                maxSim = P[i].peek() if P[i].peek() > maxSim else maxSim
        
        k1, k2 = maxSim.i, maxSim.j
        
        # update I
        I[k2] = 0

        # update A
        A[k1].merge(A[k2])

        # update P
        P[k1] = PriorityQueue()
        updateIdx = (i for i in range(N) if I[i] == 1 and i != k1)
        for i in updateIdx:
            # new simularity between i and k1
            new_Sim = complete_link(i, k1, k2)
            C[i][k1], C[k1][i] = new_Sim, new_Sim

            P[i].delete(Cluster_Node(i, k1))
            P[i].delete(Cluster_Node(i, k2))
            P[i].insert(Cluster_Node(i, k1, new_Sim))
            P[k1].insert(Cluster_Node(i, k1, new_Sim))
        
        if sum(I) == K:
            break
        print("Clusters Num: {}".format(sum(I)), end='\r')
    
    result_A = [A[i] for i in range(N) if I[i] == 1]
    return(result_A)

if __name__ == "__main__":
    vc.construct_dictionary(document_dir, dictionary_file=dictionary_file)
    finished_num = 0
    for doc in os.listdir(document_dir):
        vc.doc_to_vector(os.path.join(document_dir, doc), dictionary_file, os.path.join(output_dir, doc), len(os.listdir(document_dir)))
        finished_num += 1
        print("Finish {}".format(finished_num), end='\r')

    docs = os.listdir(document_dir)
    compute_sim_matrix(docs)
    C = np.loadtxt(c_matrix_file)
    result = efficient_hac(docs, C, K)
    with open("{}.txt".format(K), "w") as f:
        for c in result:
            for doc in c:
                f.write("{}\n".format(doc[:-4]))
            f.write('\n')
