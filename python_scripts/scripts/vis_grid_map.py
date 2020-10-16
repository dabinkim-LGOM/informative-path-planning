import grid_map_ipp_module as grid 
import obstacles as obs 
import numpy as np
import itertools as iter 
import math 
import os 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.patches as patches

class visualization():
    def __init__(self, pos, mapsize, resol, lidar_belief, save=False, reward_function=None, frontier=None, selected_ft=None, SFC=None, is_frontier=False):
        '''
        - mapsize : Axis length of the map (m)
        - resol : Resolution of grid 
        - Lidar class(C++ binding) which contains belief map
        '''
        self.pos = pos 
        self.mapsize = mapsize
        self.resol = resol
        self.lidar = lidar_belief 
        self.save = save #Bool value
        self.reward_function = reward_function
        self.all_frontier = frontier
        self.selected_ft = selected_ft 
        self.is_frontier = is_frontier
        self.SFC = SFC 

    def visualization(self, t):
        data = self.iterator()
        fig = self.show(data)

        if self.save:
            if not os.path.exists('./figures/nonmyopic/'+str(self.reward_function)+'/GridMap/'):
                os.makedirs('./figures/nonmyopic/'+str(self.reward_function)+'/GridMap/')
            fig.savefig('./figures/nonmyopic/'+str(self.reward_function)+'/GridMap/' + str(t) + '.png')


    def iterator(self):
        num_idx = math.floor(self.mapsize/ self.resol) 
        data = np.random.rand(100, 100) * 2 - 0.5
        # print(type(num_idx))
        for i in range(int(num_idx)):
            for j in range(int(num_idx)):
                x = self.resol/2.0 + i * self.resol 
                y = self.resol/2.0 + j * self.resol

                cur_val = self.lidar.get_occ_value(x, y)
                if cur_val < 0.15:
                    data[j,i] = 1.0
                elif cur_val > 0.85:
                    data[j,i] = 0.0
                else:
                    data[j,i] = 0.5
                
                # print(data[i,j])
        return data 

    def show(self, data):
        fig, ax = plt.subplots()
        cmap = mcolors.ListedColormap(['white', 'gray', 'black'])
        # bounds = [0.0, 1.0, 0.5]
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(data, cmap="gray", vmin=0, vmax=1, origin="lower")
        grid = np.arange(-self.resol/2.0, self.mapsize+1, self.resol)

        plt.scatter(x=self.pos[0], y=self.pos[1], c='y', s=6)

        if(self.is_frontier):
            self.show_frontier()
            self.show_SFC(ax)
        xmin, xmax, ymin, ymax = -self.resol/2.0, self.mapsize + self.resol/2.0, -self.resol/2.0, self.mapsize + self.resol/2.0
        # plt.show()
        return fig
        
    def show_frontier(self):
        if self.all_frontier is not None:
            for pt in self.all_frontier:
                # print('x=', pt[0], 'y=', pt[1])
                plt.scatter(x=pt[0], y=pt[1], c='r', s=3)
        if self.selected_ft is not None:
            for pt in self.selected_ft:
                # print('x=', pt[0], 'y=', pt[1])
                plt.scatter(x=pt[0], y=pt[1], c='b', s=3)
        

    def show_SFC(self, ax):
        if self.SFC is not None: 
            for box_vec in self.SFC:
                # for box in box_vec:
                box = box_vec[0]
                rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1, edgecolor='k',facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)


        

# fig, ax = plt.subplots()
# cmap = mcolors.ListedColormap(['white', 'black'])
# bounds = [-0.5, 0.5, 1.5]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# data = np.random.rand(100, 100) * 2 - 0.5
# im = ax.imshow(data, cmap=cmap, norm=norm)

# grid = np.arange(-0.5, 101, 1)
# print(grid)
# xmin, xmax, ymin, ymax = -0.5, 100.5, -0.5, 100.5
# lines = ([[(x, y) for y in (ymin, ymax)] for x in grid]
#          + [[(x, y) for x in (xmin, xmax)] for y in grid])
# grid = mcoll.LineCollection(lines, linestyles='solid', linewidths=2,
#                             color='teal')
# # ax.add_collection(grid)

# def animate(i):
#     data = np.random.rand(100, 100) * 2 - 0.5
#     im.set_data(data)
#     # return a list of the artists that need to be redrawn
#     return [im, grid]

# anim = animation.FuncAnimation(
#     fig, animate, frames=200, interval=0, blit=True, repeat=False)
# plt.show()

