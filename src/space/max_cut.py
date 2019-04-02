'''
Created on Mar 21, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

from dimod import ExactSolver, RandomSampler, SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import numpy as np
import copy

#---------------------------------------------------------------------------------
#    Copyright (C) 2004-2018 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Author:  Aric Hagberg (hagberg@lanl.gov),
#          Pieter Swart (swart@lanl.gov),
#          Dan Schult(dschult@colgate.edu)
"""Base class for undirected graphs.
"""

class Graph(object):
    """
    Base class for undirected graphs.
    """

    def __init__(self):
        """Initialize a graph with edges.
        """
        self._node = dict()
        self._adj = dict()
        self._edge = dict()
        
    @property
    def adj(self):
        return self._adj

    def add_nodes_from(self, nodes_for_adding):
        """Add multiple nodes.
        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.).
            OR
            A container of (node, attribute dict) tuples.
            Node attributes are updated using the attribute dict.

        """
        
        for n in nodes_for_adding:
            if n not in self._node:
                self._adj[n] = self._adj.get(n, {})
                self._node[n] = {}

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.
        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.).  If a node
            in the container is not in the graph it is silently
            ignored.
        """
        adj = self._adj
        for n in nodes:
            try:
                del self._node[n]
                for u in list(adj[n]):   # list handles self-loops
                    del adj[u][n]  # (allows mutation of dict in loop)
                del adj[n]
            except KeyError:
                pass

        # Get edges for nodes.
        edges = set()
        
        for r in self._adj:
            for c in self._adj[r]:
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
        
        self._edge = list(edges)

    @property
    def nodes(self):
        return self._node

    def number_of_nodes(self):
        """Returns the number of nodes in the graph.
        """
        return len(self._node)

    def add_edges_from(self, ebunch_to_add):
        """Add all the edges in ebunch_to_add.
        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as as 2-tuples (u, v).
        """
        
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 2:
                u, v = e
            elif ne == 3:
                u, v, _ = e
            else:
                raise ValueError(
                    "Edge tuple %s must be a 2-tuple." % (e,))
            if u not in self._node:
                self._adj[u] = dict()
                self._node[u] = dict()
            if v not in self._node:
                self._adj[v] = dict()
                self._node[v] = dict()
            datadict = self._adj[u].get(v, dict())

            self._adj[u][v] = datadict
            self._adj[v][u] = datadict
            
        # Get edges for nodes.
        edges = set()
        
        for r in self._adj:
            for c in self._adj[r]:
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
        
        self._edge = list(edges)
        
    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.
        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed
            from the graph.
        """
        adj = self._adj
        for e in ebunch:
            u, v = e
            if u in adj and v in adj[u]:
                del adj[u][v]
                if u != v:  # self loop needs only one entry removed
                    del adj[v][u]

        # Get edges for nodes.
        edges = set()
        
        for r in self._adj:
            for c in self._adj[r]:
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
        
        self._edge = list(edges)

    @property
    def edges(self):
        return self._edge 

    def getEdgesForNodes(self, nodes):
        
        # Get edges for nodes.
        edges = set()
        
        for r in nodes:
            
            # Check r is within self._node:
            if self._node.get(r, True):
                continue
            
            for c in self._adj[r]:
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
        
        return list(edges)       

    @property
    def degree(self):
        deg = dict()
        sorted_nodes = np.sort(np.asarray(list(self._node.keys())))
        for n in sorted_nodes: deg[n] = (n, len(self._adj[n]))
        
        return deg 
     
    def number_of_edges(self):
        """Returns the number of edges.
        """
        return len(self._edge)

    def copy(self):
        return copy.deepcopy(self) #?
    
    def subgraph(self, nodes):
        """Returns an independent subGraph induced on `nodes`.
        Parameters
        ----------
        nodes : list, iterable
            A container of nodes which will be iterated through once.
        Returns
        -------
        G : SubGraph.
        """

        subG = Graph()
        nodesSet = set(nodes)
        
        # Get edges for nodes.
        edges = set()
        
        for r in nodes:
            for c in self._adj[r]:
                
                # Check whether c is in nodes.
                if nodesSet.issuperset(set([c])) == False:
                    continue
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
        
        subG.add_nodes_from(nodes)
        subG.add_edges_from(list(edges))
        
        return subG
    
    @property
    def indexEdges(self):
        i2n = list(self._node) # Key order of dictionary.
        n2i = dict()
        
        for i, n in enumerate(i2n):
            n2i[n] = i

        # Get indexed edges for nodes.
        edges = set()
        iEdges = set()
        
        for r in self._adj:
            for c in self._adj[r]:
                
                # Check a symmetric edge.
                if edges.issuperset(set([(c, r)])):
                    continue
                
                edges.add((r, c))
                iEdges.add((n2i[r], n2i[c]))
        
        return list(iEdges)        

    @staticmethod
    def union(G1, G2):
        G = Graph()
        
        # G1.
        G.add_nodes_from(list(G1.nodes))
        G.add_edges_from(list(G1.edges))

        # G2.
        G.add_nodes_from(list(G2.nodes))
        G.add_edges_from(list(G2.edges))

        return G
    
#-------------------------------------------------------------------------------------

# Constants.
SA = 0 # Simulated Annealing mode.
QA = 1 # Quantum Annealing mode.

class QUBO(object):
    '''
        Quadratic unconstraint binary optimization  
    '''
    
    def __init__(self, n):
        '''
            Constructor.
            @param n: Number of qubits.
        '''
        
        self.Q = np.zeros((n, n), dtype=np.int64)
        self.QDim = n
        self.offset = 0
    
    def addCoeff(self, i, j, cv):
        '''
            Add a coefficient value for Q.
            @param i: Row index integer of Q.
            @param j: Column index integer of Q.
            @param cv: Coefficient integer value.
        '''
        
        if i > j:
            temp = i
            i = j 
            j = temp
        
        self.Q[i, j] += cv
    
    def addConstant(self, cv):
        '''
            Add a constant coefficient.
            @param cv: Constant coefficient integer value.
        '''
        
        self.offset += cv
        
    def getQDict(self):
        '''
            Q dictionary for quantum annealing in DWave system.
        '''
        
        QDict = {}
        
        for i in range(self.QDim):
            for j in range(i, self.QDim):
                
                # Remove weights being zero.
                if self.Q[i, j] == 0:
                    continue
                
                QDict[('x_' + str(i + 1), 'x_' + str(j + 1))] = self.Q[i, j]
                
        return QDict
    
    def getOffset(self):
        '''
            Get offset.
        '''
        
        return self.offset

def getLowBoundsViaSAQABasedMinCut(G, numMaxCutUBNodes = 1
                                 , numLBs = 2
                                 , numMinCutUBNodes = 1
                                 , maxLmtNodes = 1024
                                 , numIter = 1
                                 , mode = SA):
    '''
        Get low bounds via SA or QA based minimum cut.
        @param G: Graph model instance of networkx.
        @param numMaxCutUBNodes: Number of a max cut upper bound' nodes.
        @param numLBs: Number of low bounds set.
        @param numMinCutUBNodes: Number of a min cut upper bound' nodes.
        @param maxLmtNodes: Maximum number of nodes of a low bound.
        @param numIter: Number of iteration.
        @param mode: SA or QA.
    '''
    
    # Get an upper bound. Exception?
    maxCutUBNodes = []
    degrees = np.asarray(list(G.degree.values()))
    numNodes = len(degrees)
    
    for i in range(numMaxCutUBNodes):
        ubNode = degrees[np.argmax(degrees[:, 1]), 0]
        maxCutUBNodes.append(ubNode)
        degrees = degrees[degrees[:, 0] != ubNode]

    # Check exception. Exception?
    if numLBs == 1:
        maxCutLBs = [G]
        return maxCutLBs, maxCutUBNodes

    if (numNodes // numLBs + numNodes % numLBs \
        + numMaxCutUBNodes * numLBs) > maxLmtNodes:
        raise ValueError('numNodes // numLBs + numNodes % numLBs + numMaxCutUBNodes * numLBs <= maxLmtNodes')

    tG = G.copy() # Graph without an upper bound's nodes.
    tG.remove_nodes_from(maxCutUBNodes) 
        
    # Separate a graph into low bound graphs via digital annealing based minimum cut.
    # Get low bound graphs for minimum cut.
    minCutLBs = [] 
    numAssignedNodes = numNodes // numLBs - numMaxCutUBNodes #?
    
    # Separate a graph randomly.
    for i in range(numLBs - 1):
        ttG = tG.copy()
        nodes1 = np.random.choice(np.asarray(list(tG.nodes)), numAssignedNodes, replace = False)
        tG.remove_nodes_from(nodes1)
        nodes2 = list(tG.nodes)
        ttG.remove_nodes_from(nodes2)
        minCutLBs.append(ttG)
    
    minCutLBs.append(tG)
        
    # Separate a graph according to the number of iteration
    # And get graphs having a minimum cut value.
    optMinCutLBs = []
    minCutValR = 1.0
    
    for i in range(numIter):
        for k in range(len(minCutLBs) - 1):
            
            # Get a pair of graphs.
            partMinCutLBs = [minCutLBs[k], minCutLBs[k+1]]
            uG = Graph.union(partMinCutLBs[0], partMinCutLBs[1])

            # Get an upper bound. Exception?
            minCutUBNodes = []
            degrees = np.asarray(list(uG.degree.values()))
            numNodes = len(degrees)
            
            for _ in range(numMinCutUBNodes):
                ubNode = degrees[np.argmax(degrees[:, 1]), 0]
                minCutUBNodes.append(ubNode)
                degrees = degrees[degrees[:, 0] != ubNode]            
            
            # Create a new pair of graphs.
            uuG = uG.copy()
            uuG.remove_nodes_from(minCutUBNodes)
            nodes1 = np.random.choice(np.asarray(list(uuG.nodes))
                                      , len(uuG.nodes) // 2
                                      , replace = False)
            uuG.remove_nodes_from(nodes1)
            nodes2 = np.asarray(list(uuG.nodes))
            
            partMinCutLBs[0] = uG.copy()
            partMinCutLBs[0].remove_nodes_from(nodes2)
            partMinCutLBs[1] = uG.copy()
            partMinCutLBs[1].remove_nodes_from(nodes1)
            
            # Conduct min cut.
            biVarVals = np.zeros((uG.number_of_nodes())
                                 , dtype=np.int64) # Zero index based.        
            
            # Make a node index map.
            i2n = list(uG.nodes) # Key order of dictionary. Nodes not sorted.
            n2i = dict()
        
            for idx, n in enumerate(i2n):
                n2i[n] = idx            
                        
            # Get indexes of the upper bound's values for each graph.
            ubIdxes = [[], []]
            
            for ub in minCutUBNodes:
                ubIdxes[0].append((np.asarray(list(partMinCutLBs[0].nodes)) == ub).nonzero()[0][0])
                ubIdxes[1].append((np.asarray(list(partMinCutLBs[1].nodes)) == ub).nonzero()[0][0])
    
            biVarVals[[n2i[n] for n in minCutUBNodes]] = 1

            if mode == SA:
                for l, pG in enumerate(partMinCutLBs):
    
                    # Make a node index map.
                    pi2n = list(pG.nodes) # Key order of dictionary. Nodes not sorted.
    
                    vSet = set(list(pG.nodes))
                            
                    # Calculate Q.
                    Q = QUBO(pG.number_of_nodes())
            
                    # Apply objective and constraint conditions.
                    # Objective
                    vEdgeSet = set(list(G.getEdgesForNodes(list(vSet))))
                    
                    for i in range(len(vSet)):
                        for j in range(i + 1, len(vSet)):
                            q_ij = 1 if vEdgeSet.issuperset(set([(i, j)])) else 0 # i, j order?
                            
                            # q_ij(x_i + x_j - 2x_ix_j).
                            Q.addCoeff(i, i, 1 * q_ij)
                            Q.addCoeff(j, j, 1 * q_ij)
                            Q.addCoeff(i, j, -2 * q_ij)
                    
                    # Constraint.
                    '''
                    v1Vals = biVarVals[np.asarray(v1Set)] #?
                    v1ValsNZIdxs = (v1Vals == 1).nonzero()[0]
                    
                    for i in v1ValsNZIdxs:
                        Q1.addCoeff(i, i, 1)
                    '''
                    for i in range(len(vSet)):
                        Q.addCoeff(i, i, -1 * (2 * int(len(vSet) / 2) - 1))
                    
                    for i in range(len(vSet)):
                        for j in range(i + 1, len(vSet)):
                            Q.addCoeff(i, j, 2)

                    Q.addConstant(int(np.power(len(vSet) / 2, 2)))
                    
                    for ub in ubIdxes[l]:
                        Q.addCoeff(ub, ub, -1)
                        Q.addConstant(1)
                    
                    bqm = BinaryQuadraticModel.from_qubo(Q.getQDict(), offset = Q.getOffset())
                    
                    print('Sample solutions via SA...')
                    res = SimulatedAnnealingSampler().sample(bqm)
                    res = res.record
                    res = res.sample[res.energy == res.energy.min()]
                    
                    freq = {}
                    for v in res:
                        freq[tuple(v)] = freq.get(tuple(v), 0) + 1
                    
                    maxFreqSol = list(freq)[np.argmax(np.asarray(list(freq.values())))]
                    
                    # Update binary variable values.
                    biVarVals[[n2i[pi2n[idx]] for idx, v in enumerate(maxFreqSol) if v == 1]] = 1 #?
            else:
                for l, pG in enumerate(partMinCutLBs):
    
                    # Make a node index map.
                    pi2n = list(pG.nodes) # Key order of dictionary. Nodes not sorted.
    
                    vSet = set(list(pG.nodes))
                            
                    # Calculate Q.
                    Q = QUBO(pG.number_of_nodes())
            
                    # Apply objective and constraint conditions.
                    # Objective
                    vEdgeSet = set(list(G.getEdgesForNodes(list(vSet))))
                    
                    for i in range(len(vSet)):
                        for j in range(i + 1, len(vSet)):
                            q_ij = 1 if vEdgeSet.issuperset(set([(i, j)])) else 0 # i, j order?
                            
                            # q_ij(x_i + x_j - 2x_ix_j).
                            Q.addCoeff(i, i, 1 * q_ij)
                            Q.addCoeff(j, j, 1 * q_ij)
                            Q.addCoeff(i, j, -2 * q_ij)
                    
                    # Constraint.
                    '''
                    v1Vals = biVarVals[np.asarray(v1Set)] #?
                    v1ValsNZIdxs = (v1Vals == 1).nonzero()[0]
                    
                    for i in v1ValsNZIdxs:
                        Q1.addCoeff(i, i, 1)
                    '''
                    for i in range(len(vSet)):
                        Q.addCoeff(i, i, -1 * (2 * int(len(vSet) / 2) - 1))
                    
                    for i in range(len(vSet)):
                        for j in range(i + 1, len(vSet)):
                            Q.addCoeff(i, j, 2)

                    Q.addConstant(int(np.power(len(vSet) / 2, 2)))
                    
                    for ub in ubIdxes[l]:
                        Q.addCoeff(ub, ub, -1)
                        Q.addConstant(1)
                    
                    bqm = BinaryQuadraticModel.from_qubo(Q.getQDict(), offset = Q.getOffset())
                    
                    print('Sample solutions via QA...')
                    sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi'
                                                              , token='xxx'
                                                              , solver='DW_2000Q_2_1'))
                    res = sampler.sample(bqm, num_reads=10)
                    res = res.record
                    res = res.sample[res.energy == res.energy.min()]
                    
                    freq = {}
                    for v in res:
                        freq[tuple(v)] = freq.get(tuple(v), 0) + 1
                    
                    maxFreqSol = list(freq)[np.argmax(np.asarray(list(freq.values())))]
                    
                    # Update binary variable values.
                    biVarVals[[n2i[pi2n[idx]] for idx, v in enumerate(maxFreqSol) if v == 1]] = 1 #?                     
                
            #print(range(len(biVarVals)))
            #print(biVarVals)
            
            # Get group1, group2.
            group_1 = []
            group_2 = []
            
            for idx, bit in enumerate(biVarVals):
                if bit == 0:
                    group_1.append(i2n[idx])
                else:
                    group_2.append(i2n[idx])
            
            minCutLBs[k] = uG.copy()
            minCutLBs[k].remove_nodes_from(group_2)
            minCutLBs[k+1] = uG.copy()
            minCutLBs[k+1].remove_nodes_from(group_1)
        
        # Calculate a min cut value.
        tG = G.copy() # Graph without an upper bound's nodes.
        tG.remove_nodes_from(maxCutUBNodes) 
        
        minCutVal = calMinCutVal(minCutLBs, tG)
        #print(minCutVal)        
        
        # Select LBs with a less minimum cut value.
        if minCutVal[1] < minCutValR:
            optMinCutLBs = minCutLBs
            minCutValR = minCutVal[1]  
        
        # Rotate minCutLBs right.
        minCutLBs = [minCutLBs[-1]] + minCutLBs[:-1]

    minCutLBs = optMinCutLBs

    # Adjust the number of a low bound's nodes into <= maxLmtNodes - numMaxCutUBNodes.
    for k, g in enumerate(minCutLBs):
        
        # Check the number of a low bound's nodes.
        if len(list(g.nodes)) + numMaxCutUBNodes > maxLmtNodes:
            numRemoveNodes = maxLmtNodes - (len(list(g.nodes)) + numMaxCutUBNodes)
        else:
            continue
        
        # Remove nodes randomly.
        rNodes = np.random.choice(np.asarray(list(g.nodes)), numRemoveNodes, replace = False)
        g.remove_nodes_from(rNodes) 
        
    # Add maxCutUBNodes to each minCutLB.
    LBs = []
    for g in minCutLBs:
        tG = G.copy()
        tG.remove_nodes_from(list(g.nodes) + maxCutUBNodes)
        ttG = G.copy()
        ttG.remove_nodes_from(list(tG.nodes))
        LBs.append(ttG) # Exception?

    for k, g in enumerate(LBs): print(k, len(g.nodes)) #?
   
    return LBs, maxCutUBNodes

def calMinCutVal(gs, uG):
    '''
        Calculate a minimum cut value of graphs.
        @param gs: Graph instances list.
        @param uG: Union graph instance.
    '''
    
    numTotalEdges = len(list(uG.edges))    
    minCutVal = 0.0
    
    for i in range(len(gs) - 1):
        rG = gs[i]
        cG = gs[i+1]
        
        nodesCompSet = set(list(cG.nodes))  
        
        for n in list(rG.nodes):
            for nv in np.asarray(list(uG.adj[n])):
                nvS = set([nv])
                
                if nodesCompSet.issuperset(nvS):
                    minCutVal += 1.0
    '''
    rG = gs[-1]
    cG = gs[0]
    
    nodesCompSet = set(list(cG.nodes))  
    
    for n in list(rG.nodes):
        for nv in np.asarray(list(uG.adj[n])):
            nvS = set([nv])
            
            if nodesCompSet.issuperset(nvS):
                minCutVal += 1.0
    '''
    
    minCutValRatio = minCutVal / numTotalEdges        
            
    return minCutVal, minCutValRatio

def calMaxCutVal(gs, uG):
    '''
        Calculate a maximum cut value of graphs.
        @param gs: Graph instances list.
        @param uG: Union graph instance.
    '''
    
    numTotalEdges = len(list(uG.edges))    
    maxCutVal = 0.0
    
    for i in range(len(gs) - 1):
        rG = gs[i]
        cG = gs[i+1]
        
        nodesCompSet = set(list(cG.nodes))  
        
        for n in list(rG.nodes):
            for nv in np.asarray(list(uG.adj[n])):
                nvS = set([nv])
                
                if nodesCompSet.issuperset(nvS):
                    maxCutVal += 1.0
    
    maxCutValRatio = maxCutVal / numTotalEdges        
            
    return maxCutVal, maxCutValRatio
                 
def doMaxCut(G,numMaxCutUBNodes = 1
                                 , numLBs = 2
                                 , numMinCutUBNodes = 1
                                 , maxLmtNodes = 1024
                                 , numIterMaxCut = 1
                                 , numIterMinCut = 1
                                 , mode = SA):
    '''
        Do max cut.
        @param G: Graph model instance of networkx.
        @param numMaxCutUBNodes: Number of a max cut upper bound' nodes.
        @param numLBs: Number of low bounds set.
        @param numMinCutUBNodes: Number of a min cut upper bound' nodes.
        @param maxLmtNodes: Maximum number of nodes of a low bound.
        @param numIterMaxCut: Number of iteration for max cut.
        @param numIterMinCut: Number of iteration for min cut.
        @param mode: SA or QA.
    '''

    biVarVals = np.zeros((G.number_of_nodes()), dtype=np.int64) # Zero index based.

    # Make a node index map.
    i2n = list(G.nodes) # Key order of dictionary. Nodes not sorted.
    n2i = dict()

    for i, n in enumerate(i2n):
        n2i[n] = i  
    
    # Get low bounds via minimum cut.
    LBs, UBNodes = getLowBoundsViaSAQABasedMinCut(G
                                                , numMaxCutUBNodes = numMaxCutUBNodes
                                           , numLBs = numLBs
                                           , numMinCutUBNodes = numMinCutUBNodes
                                           , maxLmtNodes = maxLmtNodes
                                           , numIter = numIterMinCut
                                           , mode = mode) 
    
    # Get indexes of the upper bound's values for each graph.
    ubIdxes = [[] for _ in range(len(LBs))]
    
    for ub in UBNodes:
        for i, g in enumerate(LBs):
            ubIdxes[i].append((np.asarray(list(g.nodes)) == ub).nonzero()[0][0]) # Nodes not sorted.

    biVarVals[[n2i[n] for n in UBNodes]] = 1        
    
    # Conduct max cut.
    for i in range(numIterMaxCut):
        if mode == SA:   
            for k, tG in enumerate(LBs):        
                
                # Make a node index map.
                pi2n = list(tG.nodes) # Key order of dictionary. Nodes not sorted.
                pn2i = dict()
                
                for i, n in enumerate(pi2n):
                    pn2i[n] = i     
                
                # Calculate Q for a vertex set.
                vSet = set(list(tG.nodes))
                Q = QUBO(len(vSet))
        
                # Apply objective and constraint conditions.
                # Objective
                vEdgeSet = set(list(G.getEdgesForNodes(list(vSet))))
                
                for i in range(len(vSet)):
                    for j in range(i + 1, len(vSet)):
                        q_ij = 1 if vEdgeSet.issuperset(set([(i, j)])) else 0 # i, j order?
                        
                        # q_ij(x_i + x_j - 2x_ix_j).
                        Q.addCoeff(i, i, -1 * q_ij)
                        Q.addCoeff(j, j, -1 * q_ij)
                        Q.addCoeff(i, j, 2 * q_ij)
                
                # Constraint.
                if k > 0:
                    fixedVarsBeingOne \
                        = searchForFixedVarsAsOne(biVarVals, n2i, tG, LBs[k-1], G, ubIdxes[k])
                    
                    for n in fixedVarsBeingOne:
                        Q.addCoeff(pn2i[n], pn2i[n], -1)
                        Q.addConstant(1)
                
                for ub in ubIdxes[k]:
                    Q.addCoeff(ub, ub, -1)
                    Q.addConstant(1)
                
                bqm = BinaryQuadraticModel.from_qubo(Q.getQDict(), offset = Q.getOffset())
                
                print('Sample solutions via SA...')
                res = SimulatedAnnealingSampler().sample(bqm)
                res = res.record
                res = res.sample[res.energy == res.energy.min()]
                
                freq = {}
                for v in res:
                    freq[tuple(v)] = freq.get(tuple(v), 0) + 1
                
                maxFreqSol = list(freq)[np.argmax(np.asarray(list(freq.values())))]
                
                # Update binary variable values.
                biVarVals[[n2i[pi2n[idx-1]] for idx, v in enumerate(maxFreqSol) if v == 1]] = 1 #?
        else:
                # Make a node index map.
                pi2n = list(tG.nodes) # Key order of dictionary. Nodes not sorted.
                pn2i = dict()
                
                for i, n in enumerate(pi2n):
                    pn2i[n] = i     
                
                # Calculate Q for a vertex set.
                vSet = set(list(tG.nodes))
                Q = QUBO(len(vSet))
        
                # Apply objective and constraint conditions.
                # Objective
                vEdgeSet = set(list(G.getEdgesForNodes(list(vSet))))
                
                for i in range(len(vSet)):
                    for j in range(i + 1, len(vSet)):
                        q_ij = 1 if vEdgeSet.issuperset(set([(i, j)])) else 0 # i, j order?
                        
                        # q_ij(x_i + x_j - 2x_ix_j).
                        Q.addCoeff(i, i, -1 * q_ij)
                        Q.addCoeff(j, j, -1 * q_ij)
                        Q.addCoeff(i, j, 2 * q_ij)
                
                # Constraint.
                if k > 0:
                    fixedVarsBeingOne \
                        = searchForFixedVarsAsOne(biVarVals, n2i, tG, LBs[k-1], G, ubIdxes[k])
                    
                    for n in fixedVarsBeingOne:
                        Q.addCoeff(pn2i[n], pn2i[n], -1)
                        Q.addConstant(1)
                
                for ub in ubIdxes[k]:
                    Q.addCoeff(ub, ub, -1)
                    Q.addConstant(1)
                
                bqm = BinaryQuadraticModel.from_qubo(Q.getQDict(), offset = Q.getOffset())
                
                print('Sample solutions via QA...')
                sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi'
                                                          , token='xxx'
                                                          , solver='DW_2000Q_2_1'))
                res = sampler.sample(bqm, num_reads=10)
                res = res.record
                res = res.sample[res.energy == res.energy.min()]
                
                freq = {}
                for v in res:
                    freq[tuple(v)] = freq.get(tuple(v), 0) + 1
                
                maxFreqSol = list(freq)[np.argmax(np.asarray(list(freq.values())))]
                
                # Update binary variable values.
                biVarVals[[n2i[pi2n[idx-1]] for idx, v in enumerate(maxFreqSol) if v == 1]] = 1 #?
            
        #print(range(len(biVarVals)))
        #print(biVarVals)
    
        # Rotate LBs and ubIdxes right.
        LBs = [LBs[-1]] + LBs[:-1]
        ubIdxes = [ubIdxes[-1]] + ubIdxes[:-1]
        
    # Get group1, group2.
    group_1 = []
    group_2 = []
    
    for i, bit in enumerate(biVarVals):
        if bit == 0:
            group_1.append(i + 1)
        else:
            group_2.append(i + 1)
    
    # Calculate a max cut value.
    GG1 = G.copy()
    GG2 = G.copy()
    GG1.remove_nodes_from(np.asarray(group_2) - 1)
    GG2.remove_nodes_from(np.asarray(group_1) - 1)
    
    maxCutVal = calMaxCutVal([GG1, GG2], G)
    #print('Max cut value: ', maxCutVal)
    
    return group_1, group_2, maxCutVal

def searchForFixedVarsAsOne(biVarVals, n2i, rG, cG, G, ubIdxes):
    rG = rG.copy()
    rG.remove_nodes_from(ubIdxes)
    
    fixedVarsBeingOne = list()
    cNodeSet = set(list(cG.nodes))
    
    for rNode in list(rG.nodes):
        rNodeAdj = G._adj[rNode]
        numOneWeight = 0
        numZeroWeight = 0
        
        for n in list(rNodeAdj.keys()):
            
            # Check whether n is within cG.
            if cNodeSet.issuperset(set([n])):
                if biVarVals[n2i[n]] == 1:
                    numOneWeight +=1
                else:
                    numZeroWeight +=1
        
        if numZeroWeight > numOneWeight:
            fixedVarsBeingOne.append(rNode)
    
    return fixedVarsBeingOne
             
def main(n_node, edges, mode):
    
    # Create a graph with edges.
    G = Graph()
    G.add_nodes_from(range(n_node))
    G.add_edges_from(edges)
    
    # Determine the minimum number of low bounds according to the number of nodes
    # and the number of iteration for min cut.
    if G.number_of_nodes() <= 2000:
        numMaxCutUBNodes = 1  
        numLBs = 2
        numMinCutUBNodes = 1
        numIterMaxCut = 1
        numIterMinCut = int((G.number_of_nodes() / numLBs) * 0.01)
    else:
        numMaxCutUBNodes = 1  
        numLBs = 5
        numMinCutUBNodes = 1
        numIterMaxCut = 1
        numIterMinCut = int((G.number_of_nodes() / numLBs) * 0.01)
        
    group_1, group_2, maxCutVal = doMaxCut(G
                                 ,numMaxCutUBNodes = numMaxCutUBNodes
                                 , numLBs = numLBs
                                 , numMinCutUBNodes = numMinCutUBNodes
                                 , maxLmtNodes = 1024
                                 , numIterMaxCut = numIterMaxCut
                                 , numIterMinCut = numIterMinCut
                                 , mode = mode)
    
    return (group_1, group_2, maxCutVal)

if __name__ == '__main__':
    #n_node, edges = load_graph('sample_graph.txt')
    
    import networkx as nx
    
    sG = nx.dense_gnm_random_graph(200, 10000)
    G = Graph()
    G.add_nodes_from(list(sG.nodes))
    G.add_edges_from(list(sG.edges))
    len(G.nodes), len(G.edges)
    n_node, edges = len(G.nodes), list(G.edges)   
     
    answer = main(n_node, edges, QA)
    print('Group 1', answer[0])
    print('Group 2', answer[1])