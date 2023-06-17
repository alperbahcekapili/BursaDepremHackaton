from time import sleep


class Node:
    def __init__(self, locx, locy):
        self.locx = locy
        self.locy = locx

        self.neighbors = []

    def get_neighboors(self):
        return self.neighbors
    def add_neighboor(self, n):
        self.neighbors.append(n)
        if len(self.neighbors)>8:
            raise Exception("There is a problem with adding neighboors")
    def is_my_neighboor(self, n):
        return n in self.neighbors
    def get_diff(self, n):
        return max(abs(self.locx - n.locx), abs(self.locy - n.locy))
    def get_loc(self):
        return f"({self.locx}, {self.locy})"
    def __str__(self):
        return f"({self.locx}, {self.locy})"

class Graph:
    def __init__(self, nodes):
        # expects (x,y) doublets in a list
        self.nodes = nodes
        for n in nodes:
            self.set_nodes_for(n)
        
    # set neighboor nodes of n 
    def set_nodes_for(self, n):
        for node in self.nodes:
            diff = n.get_diff(node)
            if diff == 1:
                n.add_neighboor(node)
    
    def get_route(self, start, end):
        to_be_excluded= []
        return self.bfs(start, to_be_excluded, end)

    def bfs(self, start, toexclude, target):

        
        if start.is_my_neighboor(target):
            return [start]
        
        start_neighboors = start.get_neighboors()
        traversal_so_far = {}

        to_be_exclueded = [start]
        to_be_looked_fors = [start]
        
        def which_contains(dic, val):
            print("Şu node' dan geriye path çizmeye çalışıyorum:" + val.get_loc())
            for k in dic.keys():
                for node in dic[k]:
                    if node.get_diff(val) == 0:
                        return k

        def generate_path(n):
            generated_path = [target, n]
            cur_node = n
            print("Target Reached!! Generating root...")
            while True:
                print("Generated root: " + str(generated_path))
                new_base = which_contains(traversal_so_far, cur_node)
                if new_base == None:
                    # means base is reached
                    generated_path.append(base)
                    return generated_path
                
                generated_path.append(new_base)
                cur_node = new_base
                

        while True:
            if len(to_be_looked_fors) == 0:
                return []
            
            new_to_be_looked_fors = []
            base = to_be_looked_fors.pop(0)
            
            for n in base.get_neighboors():    
                # first time we travel base 
                if n == target:
                    print(traversal_so_far)
                    return generate_path(base)
                
                if n not in to_be_exclueded:
                    # now  add them to queue
                    new_to_be_looked_fors.append(n)


            traversal_so_far[base]= new_to_be_looked_fors
            to_be_looked_fors.extend(new_to_be_looked_fors)
            to_be_exclueded.extend(new_to_be_looked_fors)
            print("New to be looked for: " + str(new_to_be_looked_fors))
            
            
            

        

        toexclude.extend(start_neighboors)


class RoadFinder:
    def __init__(self, road_mask, resolution = 20):
        self.road_mask = road_mask
        self.resolution = resolution
    """
        
       Gridlere böl merkezleri yolda mı ?   +
       Yoldaysa o nokta için bir node atalım
       Node' ları oluşturduktan sonra komşuları için(her node' un graph' daki lokasyonu o grid' in index' i dir) 
       maks  (1,1) abs uzaklıktaki node' lar komşular' dı
       graph olustu. Bundan sonra BFS atıyoruz 


        
    """
    def get_grid_values(self):
        # 10 .. 30
        # x/10 daki grid' in merkezi yoldan mı geçiyor?
        print("Shape of this openres mask is: " + str(self.road_mask.shape))
        h,w = self.road_mask.shape
        self.grid_h = int(h/self.resolution)
        # 680 / 20 -> 32
        self.grid_w = int(w/self.resolution)

        grid_vals = []
        for i in range(int(self.grid_h/2), h, self.grid_h):
            tmp = []
            for j in range(int(self.grid_w/2), w, self.grid_w):
                xmean = int(i)
                ymean = int(j)
                value = self.road_mask[xmean, ymean]
                tmp.append(value)
            grid_vals.append(tmp)
        self.grid_vals = grid_vals
        print("shape of gridvals:" + str(len(grid_vals)))



    def filter_nodes(self):
        filtered_nodes = []
        for i in range(self.resolution):
            for j in range(self.resolution):
                if self.grid_vals[i][j] >0:
                    filtered_nodes.append(Node(j,i))
        self.filtered_nodes = filtered_nodes


    def generate_graph(self):

    #    Node' ları oluşturduktan sonra komşuları için(her node' un graph' daki lokasyonu o grid' in index' i dir) 
    #    maks  (1,1) abs uzaklıktaki node' lar komşular' dı,
        # filter nodes so they have positive values on

        
        g = Graph(self.filtered_nodes)
        

        start = self.filtered_nodes[-7]
        target = self.filtered_nodes[-55]
        
        print("Starting poing for route:" + start.get_loc() + ", Target: " + target.get_loc())
        print("Which corresponds to pixels: " + str(start.locx * self.grid_h)  +","+ str(start.locy*self.grid_w)+  "and")
        print(str(target.locx * self.grid_h)  +","+ str(target.locy*self.grid_w))
        route = g.get_route(start, target )
        print("Generated route is :" + str(route))
        return start, target, route


