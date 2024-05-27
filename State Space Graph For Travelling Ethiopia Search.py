#!/usr/bin/env python
# coding: utf-8

# # 1. Consider Figure 1 (A generic state space graph for traveling Ethiopia search problem) to solve the following problems.
# i. Convert Figure 1, a State space graph for traveling Ethiopia search problem, into some
# sort of manageable data structure such as, stack or queue.
# ii.  Write a class that takes the converted state space graph, initial state, goal state and a
# search strategy and return the corresponding solution/path according to the given strategy.
# Please consider only breadth-first search and depth-first search strategies for this question.

# In[94]:


# 1.1 Convert Figure 1, a State space graph for traveling Ethiopia search problem, into some queue data structure.

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Initialize the graph
graph = {
    'Addis Ababa': ['Adama', 'Debre Berhan', 'Ambo'],
    'Adama': ['Addis Ababa', 'Asella', 'Batu', 'Metahara'],
    'Debre Berhan': ['Addis Ababa', 'Debre Sina'],
    'Ambo': ['Addis Ababa', 'Wolkite', 'Nekemte'],
    'Debre Sina': ['Debre Berhan', 'Kemissie', 'Debre Markos'],
    'Asella': ['Adama', 'Asasa'],
    'Metahara': ['Adama', 'Awash'],
    'Batu': ['Adama', 'Butajira','Shashamane'],
    'Wolkite': ['Ambo', 'Jimma', 'Worabe'],
    'Nekemte': ['Ambo', 'Gimbi'],
    'Kemissie': ['Dessie', 'Debre Sina'],
    'Debre Markos': ['Finota Selam', 'Debre Sina'],
    'Awash': ['Gabi Rasu', 'Chiro', 'Metahara'],
    'Asasa': ['Dodola', 'Asella'],
    'Butajira': ['Worabe', 'Batu'],
    'Shashamane': ['Hawasa', 'Dodola','Batu'],
    'Jimma': ['Bedele', 'Bonga','Wolkite'],
    'Worabe': ['Hosana', 'Wolkite','Butajira'],
    'Gimbi': ['Dembi Dolo', 'Nekemte'],
    'Dessie': ['Woldia', 'Kemissie'],
    'Finota Selam': ['Injibara', 'BahirDar', 'Debre Markos'],
    'Gabi Rasu': ['Samara', 'Awash'],
    'Chiro': ['DireDawa', 'Awash'],
    'Dodola': ['Bale', 'Asasa','Shashamane'],
    'Hawasa': ['Dilla', 'Shashamane'],
    'Bedele': ['Gore', 'Nekemte','Jimma'],
    'Bonga': ['Tepi', 'Mezan Tefari','Dawro','Jimma'],
    'Tepi':['Mezan Tefari','Bonga','Gore'],
    'Hosana': ['Woliata Sodo', 'Worabe','Shashamane'],
    'Dembi Dolo': ['Asosa', 'Gambella','Gimbi'],
    'Woldia': ['Lalibela', 'Alamata','Samara','Dessie'],
    'Injibara': ['BahirDar', 'Finota Selam'],
    'BahirDar': ['Azezo', 'Debre Tabor','Metekel','Injibara','Finota Selam'],
    'Debre Tabor': ['Lalibela', 'BahirDar'],
    'Samara': ['Fanti Rasu', 'Alamata','Woldia','Gabi Rasu'],
    'DireDawa': ['Harar', 'Chiro'],
    'Bale': ['Liben', 'SofOmar','Goba'],
    'Dilla': ['BuleHora', 'Hawasa'],
    'Gore': ['Gambella', 'Tepi'],
    'Mezan Tefari': ['Basketo', 'Tepi','Bonga'],
    'Dawro': ['Woliata Sodo', 'Basketo','Bonga'],
    'Woliata Sodo': ['ArbaMinch', 'Dawro','Hosana'],
    'Asosa': ['Metekel', 'Dembi Dolo'],
    'Gambella': ['Gore', 'Dembi Dolo'],
    'Lalibela': ['Sekota', 'Debre Tabor','Woldia'],
    'Alamata': ['Mekele', 'Sekota','Samara'],
    'Azezo': ['Gondar', 'Metema','BahirDar'],
    'Fanti Rasu': ['Kilbet Rasu', 'Samara'],
    'Harar': ['Babile', 'Diredawa'],
    'Liben': ['Bale'],
    'SofOmar': ['Goba', 'Kebri Dehar','Bale'],
    'Goba': ['Dega Habur', 'SofOmar','Bale'],
    'BuleHora': ['Yaballo', 'Dilla'],
    'Basketo': ['Bench Meji', 'ArbaMinch','Mizen Teferi','Dawro'],
    'ArbaMinch': ['Konso', 'Basketo','Woliata Sodo'],
    'Metekel': ['BahirDar', 'Asosa'],
    'Mekele': ['Adwa', 'Adigrat','Sekota','Alamata'],
    'Gondar': ['Debarke', 'Humera','Metema','Azezo'],
    'Metema': ['Kartum', 'Gondar','Azezo'],
    'Kilber Rasu': ['Fanti Rasu'],
    'Babile': ['Jijiga', 'Harar'],
    'Kebri Dehar': ['Werder', 'Gode','Dega Habur','SofOmar'],
    'Dega Habur': ['Jijiga', 'Kebri Dehar','Goba'],
    'Yaballo': ['Moyale', 'Konso','Bule Hora'],
    'Benchi Meji': ['Juba', 'Basketo'],
    'Konso': ['Yaballo', 'ArbaMinch'],
    'Adwa': ['Adigrat', 'Aksum','Mekele'],
    'Adigrat': ['Asmera', 'Adwa','Mekele'],
    'Debarke': ['Shire', 'Gondar'],
    'Humara': ['Kartum', 'Shire','Gondar'],

}

# Function to convert adjacency list to queue

def graph_to_queue(graph):
    queue = deque()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            queue.append((node, neighbor))
    return queue

queue = graph_to_queue(graph)

# Print queue contents
print("Queue contents:")
while queue:
    print(queue.popleft())


# In[95]:


# 1.2 Write a class that takes the converted state space graph, initial state, goal state and a
# search strategy and return the corresponding solution/path according to the given strategy.
# I consider only breadth-first search for this question.

class GraphSearch:
    def __init__(self, graph):
        self.graph = graph

    def bfs(self, start, goal):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            (vertex, path) = queue.popleft()
            if vertex in visited:
                continue

            visited.add(vertex)

            for next in self.graph[vertex]:
                if next not in visited:
                    if next == goal:
                        return path + [next]
                    queue.append((next, path + [next]))

        return None
search = GraphSearch(graph)
path = search.bfs('Addis Ababa', 'Moyale')
print("BFS Path:", path)

# Create a graph for visualization
G = nx.Graph()

# Add edges to the graph
for node, neighbors in graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Define node positions for visualization
pos = nx.spring_layout(G, seed=42)  # Positions for all nodes

# Draw the graph
plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")

# Highlight the path found by BFS
if path:
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color="red")

plt.title("Graph Visualization with BFS Path from 'Addis Ababa' to 'Moyale'")
plt.show()


# Question One Summary
# Graph Representation: The graph is represented using an adjacency list based on the image.
# BFS Implementation: I used BFS data Structure to find the path from 'Addis Ababa' to 'Moyale'.
# Graph Visualization: NetworkX and Matplotlib are used to draw the graph and highlight the BFS path.
# he path from 'Addis Ababa' to 'Moyale' has been highlighted in red.

# # 2. Given Figure 2, a state space graph with backward cost for the traveling Ethiopia search problem.
# 2.1 Convert Figure 2 into some sort of manageable data structure such as, stack or queue.
# 2.2 Assuming “Addis Ababa” as an initial state, write a program that use uniform cost search
# to generate a path to “Lalibela”.
# 2.3 Given “Addis Ababa” as an initial state and “Axum”, “Gondar”, “Lalibela”, Babile,
# “Jimma”, “Bale”, “Sof Oumer”, and “Arba Minch” as goal states;in no specific order, write
# a customized uniform cost search algorithm to generate a path that let a visitor visit all those
# goal states preserving the local optimum.

# In[101]:


# 2.1 Convert Figure 2 into some sort of manageable data structure such as, stack or queue

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Initialize the graph with the specified costs
graph = {
    'Addis Ababa': [('Adama', 3), ('Debre Berhan', 5),('Ambo', 5) ],
    'Adama': [('Addis Ababa', 3),('Asella', 4), ('Batu', 4), ('Metahara', 3)],   
    'Debre Berhan': [('Addis Ababa', 5), ('Debre Sina', 2)],
    'Ambo': [('Addis Ababa', 5), ('Wolkite', 6), ('Nekemte', 9)],
    'Debre Sina': [('Debre Berhan', 2), ('Kemissie', 6), ('Debre Markos', 17)],
    'Asella': [('Adama', 4), ('Asasa', 4)],
    'Metahara': [('Adama', 3), ('Awash', 1)],
    'Batu': [('Adama', 4), ('Butajira', 2),('Shashamane', 3)],
    'Wolkite': [('Ambo', 6), ('Jimma', 8), ('Worabe', 5)],
    'Nekemte': [('Ambo', 9), ('Gimbi', 4)],
    'Kemissie': [('Dessie', 4), ('Debre Sina', 6)],
    'Debre Markos': [('Finota Selam', 3), ('Debre Sina', 17)],
    'Awash': [('Gabi Rasu', 5), ('Chiro', 4), ('Metahara', 1)],
    'Asasa': [('Dodola', 1), ('Asella', 4)],
    'Butajira': [('Worabe', 2), ('Batu', 2)],
    'Shashamane': [('Hawasa', 1), ('Dodola', 3), ('Batu', 3),('Hosana', 7)],
    'Jimma': [('Bedele', 7), ('Bonga', 4),('Wolkite', 8)],
    'Worabe': [('Hosana', 2), ('Wolkite', 5),('Butajira', 2)],
    'Gimbi': [('Dembi Dolo', 6), ('Nekemte', 4)],
    'Dessie': [('Woldia', 6), ('Kemissie', 4)],
    'Finota Selam': [('Injibara', 2), ('BahirDar', 6), ('Debre Markos', 3)],
    'Gabi Rasu': [('Samara', 9), ('Awash', 5)],
    'Chiro': [('DireDawa', 8), ('Awash', 4)],
    'Dodola': [('Bale', 13), ('Asasa', 1),('Shashamane', 3)],
    'Hawasa': [('Dilla', 3), ('Shashamane', 1)],
    'Bedele': [('Gore', 6), ('Nekemte',0 ), ('Jimma', 7)],
    'Bonga': [('Tepi', 8), ('Mezan Tefari', 4), ('Dawro', 10), ('Jimma', 4)],
    'Tepi':[('Mezan Tefari', 4),('Bonga', 8),('Gore', 9)],
    'Hosana': [('Woliata Sodo', 4), ('Worabe', 2), ('Shashamane', 7)],
    'Dembi Dolo': [('Asosa', 12), ('Gambella', 4), ('Gimbi', 6)],
    'Woldia': [('Lalibela', 7), ('Alamata', 3), ('Samara', 8), ('Dessie', 6)],
    'Injibara': [('BahirDar', 4), ('Finota Selam', 2)],
    'BahirDar': [('Azezo', 7), ('Debre Tabor', 4), ('Metekel', 11), ('Injibara', 4), ('Finota Selam', 6)],
    'Debre Tabor': [('Lalibela', 8), ('BahirDar', 4)],
    'Samara': [('Fanti Rasu', 7), ('Alamata', 11), ('Woldia', 8), ('Gabi Rasu', 10)],
    'DireDawa': [('Harar', 4), ('Chiro', 8)],
    'Bale': [('Liben', 11), ('SofOmar', 23) , ('Goba', 18)],
    'Dilla': [('BuleHora', 4), ('Hawasa', 3)],
    'Gore': [('Gambella', 5), ('Tepi', 9)],
    'Mezan Tefari': [('Tepi', 4), ('Bonga', 4)],
    'Dawro': [('Woliata Sodo', 6), ('Bonga', 10)],
    'Woliata Sodo': [('ArbaMinch', 0), ('Dawro', 6), ('Hosana', 4)],
    'Asosa': [('Dembi Dolo', 12)],
    'Gambella': [('Gore', 5), ('Dembi Dolo', 4)],
    'Lalibela': [('Sekota', 6), ('Debre Tabor', 8), ('Woldia', 7)],
    'Alamata': [('Mekele', 5), ('Sekota', 6),('Samara', 11)],
    'Azezo': [('Gondar', 1), ('Metema', 7),('BahirDar', 7)],
    'Fanti Rasu': [('Kilbet Rasu', 6), ('Samara', 7)],
    'Harar': [('Babile', 2), ('DireDawa', 4)],
    'Liben': [('Bale', 11)],
    'SofOmar': [('Goba', 6), ('Bale', 23)],
    'Goba': [('Babile', 28), ('SofOmar', 6),('Bale', 18)],
    'BuleHora': [('Yaballo', 3), ('Dilla', 4)],
    'Basketo': [('Bench Meji', 5), ('ArbaMinch', 10)],
    'ArbaMinch': [('Konso', 4), ('Basketo', 10),('Woliata Sodo',0)],
    'Metekel': [('BahirDar', 11)],
    'Mekele': [('Adwa', 11), ('Adigrat', 4),('Sekota', 9),('Alamata', 5)],
    'Gondar': [('Debarke', 4), ('Humera', 9),('Metema', 7),('Azezo', 1)],
    'Metema': [('Kartum', 19), ('Gondar', 7),('Azezo', 7)],
    'Kilber Rasu': [('Fanti Rasu', 6)],
    'Babile': [('Harar', 2)],
    'Kebri Dehar': [('Werder', 6), ('Gode', 5),('Dega Habur', 6)],
    'Dega Habur': [('Kebri Dehar', 6)],
    'Yaballo': [('Konso', 3), ('BuleHora', 3)],
    'Benchi Meji': [('Juba', 22), ('Basketo', 5)],
    'Konso': [('Yaballo', 3), ('ArbaMinch', 4)],
    'Adwa': [('Adigrat', 4), ('Aksum', 1), ('Mekele', 7)],
    'Adigrat': [('Asmera', 6), ('Adwa', 4), ('Mekele', 4)],
    'Debarke': [('Shire', 7) , ('Gondar', 4)],
    'Humara': [('Kartum',21), ('Shire', 8),('Gondar', 9)],

}

# Function to convert adjacency list to queue

def graph_to_queue(graph):
    queue = deque()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            queue.append((node, neighbor))
    return queue

queue = graph_to_queue(graph)

# Print queue contents
print("Queue contents:")
while queue:
    print(queue.popleft())


# In[102]:


# 2.2 Assuming “Addis Ababa” as an initial state, write a program that use uniform cost search to generate a path to “Lalibela”

import heapq

class GraphSearch:
    def __init__(self, graph):
        self.graph = graph

    def ucs(self, start, goal):
        queue = [(0, start, [])]  # priority queue of (cost, current_node, path)
        visited = set()  # set of visited nodes

        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue

            visited.add(node)
            path = path + [node]

            if node == goal:
                return path, cost

            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + edge_cost, neighbor, path))

        return None, float('inf')

# Perform Uniform Cost Search to find the path from 'Addis Ababa' to 'Lalibela'
search = GraphSearch(graph)
path, cost = search.ucs('Addis Ababa', 'Lalibela')
print("UCS Path:", path)
print("Total Cost:", cost)


# In[103]:


# 2.3 Given “Addis Ababa” as an initial state and “Axum”, “Gondar”, “Lalibela”, Babile,
# “Jimma”, “Bale”, “Sof Oumer”, and “Arba Minch” as goal states;in no specific order, write
# a customized uniform cost search algorithm to generate a path that let a visitor visit all those
# goal states preserving the local optimum.


# In[104]:


class GraphSearch:
    def __init__(self, graph):
        self.graph = graph

    def ucs_multi_goal(self, start, goals):
        queue = [(0, start, [])]  # priority queue of (cost, current_node, path)
        visited = set()  # set of visited nodes
        goals_set = set(goals)  # set of goal nodes
        all_visited_paths = []  # store all visited paths and costs

        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue

            visited.add(node)
            path = path + [node]

            # If the current node is a goal, add the path to all_visited_paths
            if node in goals_set:
                goals_set.remove(node)
                all_visited_paths.append((path, cost))
                # If all goals have been visited, return the paths
                if not goals_set:
                    return all_visited_paths

            for neighbor, edge_cost in self.graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + edge_cost, neighbor, path))

        return all_visited_paths

# Perform UCS to find the paths from 'Addis Ababa' to all specified goals
search = GraphSearch(graph)
goals = ['Axum', 'Gondar', 'Lalibela', 'Babile', 'Jimma', 'Bale', 'Sof Oumer', 'Arba Minch']
paths = search.ucs_multi_goal('Addis Ababa', goals)

for path, cost in paths:
    print("Path:", path)
    print("Total Cost:", cost)


# # 3. Given Figure 3, a state space graph with heuristic and backward cost.
# Write a class that use A* search to generate a path from the initial state “Addis Ababa” to goal state “Moyale”.

# In[107]:



# Initialize the graph with the specified costs
graph = {
    'Addis Ababa': [('Adama', 3), ('Debre Berhan', 5),('Ambo', 5),('Debre Markos',13) ],
    'Adama': [('Addis Ababa', 3),('Asella', 4), ('Batu', 4), ('Metahara', 3)],   
    'Debre Berhan': [('Addis Ababa', 5), ('Debre Sina', 2)],
    'Ambo': [('Addis Ababa', 5), ('Wolkite', 6), ('Nekemte', 9)],
    'Debre Sina': [('Debre Berhan', 2), ('Kemissie', 6), ('Debre Markos', 17)],
    'Asella': [('Adama', 4), ('Asasa', 4)], 
    'Metahara': [('Adama', 3), ('Awash', 1)],
    'Batu': [('Adama', 4), ('Butajira', 2),('Shashamane', 3)],
    'Wolkite': [('Ambo', 6), ('Jimma', 8), ('Worabe', 5)],
    'Nekemte': [('Ambo', 9), ('Gimbi', 4)],
    'Kemissie': [('Dessie', 4), ('Debre Sina', 6)],
    'Debre Markos': [('Finota Selam', 3), ('Debre Sina', 17)],
    'Awash': [('Gabi Rasu', 5), ('Chiro', 4), ('Metahara', 1)],
    'Asasa': [('Dodola', 1), ('Asella', 4)],
    'Butajira': [('Worabe', 2), ('Batu', 2)],
    'Shashamane': [('Hawasa', 1), ('Dodola', 3), ('Batu', 3),('Hosana', 7)],
    'Jimma': [('Bedele', 7), ('Bonga', 4),('Wolkite', 8)],
    'Worabe': [('Hosana', 2), ('Wolkite', 5),('Butajira', 2)],
    'Gimbi': [('Dembi Dolo', 6), ('Nekemte', 4)],
    'Dessie': [('Woldia', 6), ('Kemissie', 4)],
    'Finota Selam': [('Injibara', 2), ('BahirDar', 6), ('Debre Markos', 3)],
    'Gabi Rasu': [('Samara', 9), ('Awash', 5)],
    'Chiro': [('DireDawa', 8), ('Awash', 4)],
    'Dodola': [('Bale', 13), ('Asasa', 1),('Shashamane', 3)],
    'Hawasa': [('Dilla', 3), ('Shashamane', 1)],
    'Bedele': [('Gore', 6), ('Nekemte',0 ), ('Jimma', 7)],
    'Bonga': [('Tepi', 8), ('Mezan Tefari', 4), ('Dawro', 10), ('Jimma', 4)],
    'Tepi':[('Mezan Tefari', 4),('Bonga', 8),('Gore', 9)],
    'Hosana': [('Woliata Sodo', 4), ('Worabe', 2), ('Shashamane', 7)],
    'Dembi Dolo': [('Asosa', 12), ('Gambella', 4), ('Gimbi', 6)],
    'Woldia': [('Lalibela', 7), ('Alamata', 3), ('Samara', 8), ('Dessie', 6)],
    'Injibara': [('BahirDar', 4), ('Finota Selam', 2)],
    'BahirDar': [('Azezo', 7), ('Debre Tabor', 4), ('Metekel', 11), ('Injibara', 4), ('Finota Selam', 6)],
    'Debre Tabor': [('Lalibela', 8), ('BahirDar', 4)],
    'Samara': [('Fanti Rasu', 7), ('Alamata', 11), ('Woldia', 8), ('Gabi Rasu', 10)],
    'DireDawa': [('Harar', 4), ('Chiro', 8)],
    'Bale': [('Liben', 11), ('SofOmar', 23) , ('Goba', 18)],
    'Dilla': [('BuleHora', 4), ('Hawasa', 3)],
    'Gore': [('Gambella', 5), ('Tepi', 9)],
    'Mezan Tefari': [('Tepi', 4), ('Bonga', 4)],
    'Dawro': [('Woliata Sodo', 6), ('Bonga', 10)],
    'Woliata Sodo': [('ArbaMinch', 0), ('Dawro', 6), ('Hosana', 4)],
    'Asosa': [('Dembi Dolo', 12)],
    'Gambella': [('Gore', 5), ('Dembi Dolo', 4)],
    'Lalibela': [('Sekota', 6), ('Debre Tabor', 8), ('Woldia', 7)],
    'Alamata': [('Mekele', 5), ('Sekota', 6),('Samara', 11)],
    'Azezo': [('Gondar', 1), ('Metema', 7),('BahirDar', 7)],
    'Fanti Rasu': [('Kilbet Rasu', 6), ('Samara', 7)],
    'Harar': [('Babile', 2), ('DireDawa', 4)],
    'Liben': [('Bale', 11)],
    'SofOmar': [('Goba', 6), ('Bale', 23)],
    'Goba': [('Babile', 28), ('SofOmar', 6),('Bale', 18)],
    'BuleHora': [('Yaballo', 3), ('Dilla', 4)],
    'Basketo': [('Bench Meji', 5), ('ArbaMinch', 10)],
    'ArbaMinch': [('Konso', 4), ('Basketo', 10),('Woliata Sodo',0)],
    'Metekel': [('BahirDar', 11)],
    'Mekele': [('Adwa', 11), ('Adigrat', 4),('Sekota', 9),('Alamata', 5)],
    'Gondar': [('Debarke', 4), ('Humera', 9),('Metema', 7),('Azezo', 1)],
    'Metema': [('Kartum', 19), ('Gondar', 7),('Azezo', 7)],
    'Kilber Rasu': [('Fanti Rasu', 6)],
    'Babile': [('Harar', 2)],
    'Kebri Dehar': [('Werder', 6), ('Gode', 5),('Dega Habur', 6)],
    'Dega Habur': [('Kebri Dehar', 6)],
    'Yaballo': [('Konso', 3), ('BuleHora', 2),('Moyale', 6)],
    'Moyale': [('Yaballo', 6), ('Nairobi', 22),('Liben', 6),('Mokadisho', 40)],
    'Benchi Meji': [('Juba', 22), ('Basketo', 5)],
    'Konso': [('Yaballo', 3), ('ArbaMinch', 4)],
    'Adwa': [('Adigrat', 4), ('Aksum', 1), ('Mekele', 7)],
    'Adigrat': [('Asmera', 6), ('Adwa', 4), ('Mekele', 4)],
    'Debarke': [('Shire', 7) , ('Gondar', 4)],
    'Humara': [('Kartum',21), ('Shire', 8),('Gondar', 9)],

}

# heuristic values 
heuristic = {
    'Addis Ababa': 26,
    'Adama': 23,
    'Debre Berhan': 31,
    'Ambo': 31,
    'Debre Sina': 33,
    'Asella': 22,
    'Metahara': 26,
    'Batu': 19,
    'Wolkite': 25,
    'Nekemte': 39,
    'Kemissie': 40,
    'Debre Markos': 39,
    'Awash': 27,
    'Asasa': 18,
    'Butajira': 21,
    'Shashamane': 16,
    'Jimma': 33,
    'Worabe': 22,
    'Gimbi': 43,
    'Dessie': 44,
    'Finota Selam': 42,
    'Gabi Rasu': 32,
    'Chiro': 31,
    'Dodola': 19,
    'Hawasa': 15,
    'Bedele': 40,
    'Bonga': 33,
    'Tepi': 41,
    'Hosana': 21,
    'Dembi Dolo': 49,
    'Woldia': 50,
    'Injibara': 44,
    'BahirDar': 48,
    'Debre Tabor': 52,
    'Samara': 42,
    'DireDawa': 31,
    'Bale': 22,
    'Dilla': 12,
    'Gode': 35,
    'Dawro': 23,
    'Woliata Sodo': 17,
    'Asosa': 51,
    'Gambella': 51,
    'Lalibela': 57,
    'Alamata': 53,
    'Azezo': 55,
    'Fanti Rasu': 49,
    'Harar': 35,
    'Liben': 11,
    'SofOmar': 45,
    'Goba': 40,
    'BuleHora': 8,
    'Basketo': 23,
    'ArbaMinch': 13,
    'Metekel': 59,
    'Mekele': 58,
    'Gondar': 56,
    'Metema': 62,
    'Kilbet Rasu': 55,
    'Babile': 37,
    'Kebri Dehar': 40,
    'Dega Habur': 45,
    'Yaballo': 6,
    'Moyale': 0,
    'Bench Meji': 28,
    'Konso': 9,
    'Adwa': 65,
    'Adigrat': 62,
    'Debarke': 60,
    'Humara': 65
}

class AStarSearch:
    def __init__(self, graph, heuristic):
        self.graph = graph
        self.heuristic = heuristic

    def astar(self, start, goal):
        queue = [(0 + self.heuristic[start], 0, start, [])]  # (f, g, node, path)
        visited = set()

        while queue:
            f, g, node, path = heapq.heappop(queue)

            if node in visited:
                continue

            path = path + [node]
            visited.add(node)

            if node == goal:
                return path, g

            for neighbor, cost in self.graph.get(node, []):
                if neighbor not in visited:
                    g_new = g + cost
                    f_new = g_new + self.heuristic.get(neighbor, float('inf'))
                    heapq.heappush(queue, (f_new, g_new, neighbor, path))

        return None, float('inf')

# Perform A* search to find the path from 'Addis Ababa' to 'Moyale'
search = AStarSearch(graph, heuristic)
path, cost = search.astar('Addis Ababa', 'Moyale')
print("A* Path:", path)
print("Total Cost:", cost)


# # 4. Assume an adversary joins the Traveling Ethiopia Search Problem as shown in Figure 4. 
# The goalof the agent would be to reach to a state where it gains good quality of Coffee. Write a class that
# shows how MiniMax search algorithm directs an agent to the best achievable destination.

# To solve the Traveling Ethiopia Adversarial Search Problem using the Minimax algorithm, we haveto implement a class that models the problem and 
# applies the Minimax algorithm to find the best achievable destination for the agent, considering the adversary's moves. Here's how we can approach this:
# 
# Model the Problem: Create a graph representation of the state space.
# Define the Minimax Algorithm: Implement the Minimax algorithm to evaluate the utility values of terminal nodes and propagate these values up the 
# tree to determine the best move.
# Class Implementation: Write a class that encapsulates the problem, including the graph, utility values, and the Minimax search.
# The Implementation will be done as follows:-

# In[114]:


class TravelingEthiopiaAdversarial:
    def __init__(self, graph, utilities, is_terminal):
        self.graph = graph
        self.utilities = utilities
        self.is_terminal = is_terminal
    
    def minimax(self, state, depth, maximizing_player):
        if depth == 0 or self.is_terminal[state]:
            return self.utilities[state]
        
        if maximizing_player:
            max_eval = float('-inf')
            for neighbor in self.graph[state]:
                eval = self.minimax(neighbor, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for neighbor in self.graph[state]:
                eval = self.minimax(neighbor, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
    
    def find_best_move(self, initial_state, depth):
        best_value = float('-inf')
        best_move = None
        for neighbor in self.graph[initial_state]:
            move_value = self.minimax(neighbor, depth - 1, False)
            if move_value > best_value:
                best_value = move_value
                best_move = neighbor
        return best_move, best_value

# graph representation
graph = {
    'Addis Ababa': ['Ambo', 'Adama', 'Butajira'],
    'Gedo': ['Shambu'],
    'Adama': ['Mojo', 'DireDawa'],
    'Ambo': ['Nekemte', 'Gedo'],
    'Nekemte': ['Gimbi','Limu'],
    'Shambu': [],
    'Mojo': ['Kaffa','Dilla'],
    'DireDawa': ['Harar', 'Chiro'],
    'Wolkite': ['Benchi Naji', 'Tepi'],
    'Butajira': ['Worabe','Wolkite'],
    'Harar': [],
    'Chiro': [],
    'Gimbi': [],
    'Worabe': ['Hossana','Durame'],
    'Hossana': [],
    'Limu': [],
    'Tepi': [],
    'Kaffa': [],
    'Dilla': [],
    'Durame': [],
    'Bench Naji': [],
    'Dilla': [],
}

# Utility values for terminal nodes 
utilities = {
    'Addis Ababa': 0,
    'Gedo': 4,
    'Adama': 3,
    'Ambo': 5,
    'Nekemte': 8,
    'Shambu': 4,
    'Mojo': 2,
    'DireDawa': 6,
    'Wolkite': 6,
    'Butajira': 6,
    'Harar': 10,
    'Chiro': 6,
    'Gimbi': 8,
    'Worabe': 6,
    'Hossana': 6,
    'Limu': 8,
    'Tepi': 6,
    'Kaffa': 7,
    'Dilla': 9,
    'Durame': 5,
    'Benchi Naji': 5,
}

# Terminal nodes
is_terminal = {state: (state not in graph or not graph[state]) for state in utilities}

# Create the adversarial search instance
adversarial_search = TravelingEthiopiaAdversarial(graph, utilities, is_terminal)

# Perform the search
initial_state = 'Addis Ababa'
depth = 3  # Adjust depth as needed
best_move, best_value = adversarial_search.find_best_move(initial_state, depth)

print("Best Move:", best_move)
print("Best Value:", best_value)


# In[ ]:




