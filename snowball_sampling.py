import networkx as nx
import random as random

class Queue():
    #create a list to represent the queue
    def __init__(self):
        self.queue = list()
    
    #Adding elements to queue
    def enqueue(self,data):
        #Check to avoid duplicate entry
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False
    
    #Pop the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            exit()
    
    #Get the size of the queue
    def size(self):
        return len(self.queue)
    
    #print the elements of the queue
    def printQueue(self):
        return self.queue  



class Snowball():
    def __init__(self):
        self.G_new = nx.Graph()

    def snowball(self, G, target_precent, k, random_seed = 1):      
        # create a queue to hold the nodes to be explored 
        q=Queue() 
        # create a set to hold the set of nodes that has been explored
        explored_node_set = set()
        # create another set to hold the set of nodes that will in included in the sampled_graph
        new_graph_node_set = set()
        # get the list of nodes in the origional graph G
        list_nodes = list(G.nodes())
        
        # target sample size 
        target_size = target_precent * len(list_nodes)
        
        # sample a source node
        random.seed(random_seed)
        source_id = random.sample(list(G.nodes()),1)[0]
        q.enqueue(source_id)
        
        # while the number of samples haven't reached the target: 
        while(len(new_graph_node_set) < target_size):
            # if the queue is not empty:
            if(q.size() > 0):
                id = q.dequeue()
                # add this current node into the new graph
                new_graph_node_set.add(id)
            
                # if this node has not been explored before
                if(id not in explored_node_set):
                    explored_node_set.add(id)
                    list_neighbors = list(G.neighbors(id))
                    # random shuffle the neighbor list of id
                    #random.seed(random_seed)
                    random.shuffle(list_neighbors)
                    # if there are more than k neighbors
                    if(len(list_neighbors) > k):
                        for x in list_neighbors[:k]:
                            q.enqueue(x)
                            new_graph_node_set.add(x)
                    # else if there're less than k neighbors but there're still neighbor exists
                    elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                        for x in list_neighbors:
                            q.enqueue(x)
                            new_graph_node_set.add(x)
                            
                else:
                    continue
            # if the queue is empty, then sample another node which hasn't been seen before
            else:
                initial_nodes = random.sample(list(G.nodes()) and list(explored_node_set), 1)
                for id in initial_nodes:
                    q.enqueue(id) 
        self.G_new = G.subgraph(set(new_graph_node_set))
        #print(new_graph_node_set)
        return self.G_new







