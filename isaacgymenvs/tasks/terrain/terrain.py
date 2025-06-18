# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg['mesh_type']
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg['terrain_length']
        self.env_width = cfg['terrain_width']

        self.cfg['num_sub_terrains'] = cfg['num_rows'] * cfg['num_cols']
        self.env_origins = np.zeros((cfg['num_rows'], cfg['num_cols'], 3))

        self.width_per_env_pixels = int(self.env_width / cfg['horizontal_scale'])
        self.length_per_env_pixels = int(self.env_length / cfg['horizontal_scale'])

        self.border = int(cfg['border_size'] / cfg['horizontal_scale'])
        self.tot_cols = int(cfg['num_cols'] * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg['num_rows'] * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        self.random_terrain = True
        
        self.self_play_terrain()
        # self.demo_terrain()
        
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            print("Converting heightmap to trimesh...")
            if cfg['hf2mesh_method'] == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                    cfg['horizontal_scale'],
                                                    cfg['vertical_scale'],
                                                    cfg['slope_treshold'])
                half_edge_width = int(cfg['edge_width_thresh'] / cfg['horizontal_scale'])
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if cfg['simplify_grid']:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg['hf2mesh_method'] == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, cfg['horizontal_scale'], cfg['vertical_scale'], max_error=cfg['max_error'])
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def self_play_terrain(self):
        for i in range(self.cfg['num_rows']):
            for j in range(self.cfg['num_cols']):
                # make a random terrain
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg['vertical_scale'],
                                horizontal_scale=self.cfg['horizontal_scale'])

                # add four walls
                wall_width = 0.2
                max_height = 3.5
                wall_width_int = max(1, int(wall_width/terrain.horizontal_scale))
                max_height_int = int(max_height / terrain.vertical_scale)
                terrain_length = terrain.length
                height2width_ratio = max_height_int / wall_width_int
                xs = np.arange(terrain_length)
                heights = height2width_ratio * xs 
                # if i % 2 == 0:
                terrain.height_field_raw[:wall_width_int, :] = heights[:wall_width_int].reshape(-1, 1)

                wall_length = 0.2
                wall_length_int = max(1, int(wall_length/terrain.horizontal_scale))
                max_height_int = int(max_height / terrain.vertical_scale)
                terrain_width = terrain.width
                height2width_ratio = max_height_int / wall_length_int
                ys = np.arange(terrain_width)
                heights = height2width_ratio * ys
                # if j == 0:
                terrain.height_field_raw[:, :wall_length_int] = heights[:wall_length_int].reshape(1, -1)

                # add walls to the opposite side
                if i == self.cfg['num_rows'] - 1 :
                    terrain.height_field_raw[-wall_width_int:, :] = heights[:wall_width_int].reshape(-1, 1)
                if j == self.cfg['num_cols'] - 1 :
                    terrain.height_field_raw[:, -wall_length_int:] = heights[:wall_length_int].reshape(1, -1)

                # add a "safe" area at the top right of the terrain, where have four walls but have only one door size = (2x2)
   
                #! Start making the complex terrain
                # Complex Terrain #1
                '''random slopes'''
                # complex_terrain1(terrain)
                # complex_terrain2(terrain)

                '''hurdle terrain'''
                # parkour_hurdle_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=6, stone_len=0.3, x_range=[1.5, 2.4], y_range=[-0.4, 0.4], half_valid_width=[0.4, 0.8], hurdle_height_range=[0.05, 0.1], pad_width=0.1, pad_height=0.3, flat=False)

                '''stepping stones'''
                # stones_terrain(terrain, height_range=[0.05, 0.3])
                # add roughness
                # self.add_roughness(terrain, difficulty=2)
                # slope_terrain(terrain, slope_angle_range=[0.5, 1.5])

                # ! General Methods
                # difficulty = i
                # types = j
                # self.make_terrain(terrain, difficulty, types, random_terrain=self.random_terrain)
                gap_terrain(terrain, gap_width_range=[1.0,1.1], random_terrain=False)

                safe_xs = np.arange(terrain_length/3)
                safe_heights = height2width_ratio * safe_xs
                terrain.height_field_raw[:wall_width_int, :50] = safe_heights[:wall_width_int].reshape(-1, 1)
                terrain.height_field_raw[150:150+wall_width_int, :30] = safe_heights[:wall_width_int].reshape(-1, 1)
                terrain.height_field_raw[-50:, :wall_length_int] = safe_heights[:wall_length_int].reshape(1, -1)
                terrain.height_field_raw[-50:, 45:45+wall_length_int] = safe_heights[:wall_length_int].reshape(1, -1)

                
                # map coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = start_x + self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = start_y + self.width_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
                
                # set the origin of the terrain
                env_origin_x = i * self.env_length
                env_origin_y = j * self.env_width
                env_origin_z = 0.0
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
    
    def demo_terrain(self):
        types = 0
        self.cfg.num_rows = 2
        self.cfg.num_cols = 2
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                # make a random terrain
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

                # add four walls
                wall_width = 0.2
                max_height = 3.0
                wall_width_int = max(1, int(wall_width/terrain.horizontal_scale))
                max_height_int = int(max_height / terrain.vertical_scale)
                terrain_length = terrain.length
                height2width_ratio = max_height_int / wall_width_int
                xs = np.arange(terrain_length)
                heights = height2width_ratio * xs 
                if i == 0:
                    terrain.height_field_raw[:wall_width_int, :] = heights[:wall_width_int].reshape(-1, 1)

                wall_length = 0.2
                wall_length_int = max(1, int(wall_length/terrain.horizontal_scale))
                max_height_int = int(max_height / terrain.vertical_scale)
                terrain_width = terrain.width
                height2width_ratio = max_height_int / wall_length_int
                ys = np.arange(terrain_width)
                heights = height2width_ratio * ys
                if j == 0:
                    terrain.height_field_raw[:, :wall_length_int] = heights[:wall_length_int].reshape(1, -1)

                # add walls to the opposite side
                if i == self.cfg.num_rows - 1:
                    terrain.height_field_raw[-wall_width_int:, :] = heights[:wall_width_int].reshape(-1, 1)
                if j == self.cfg.num_cols - 1:
                    terrain.height_field_raw[:, -wall_length_int:] = heights[:wall_length_int].reshape(1, -1)
                
                #! Start making the complex terrain
                # Complex Terrain #1
                '''random slopes'''
                # complex_terrain1(terrain)
                # complex_terrain2(terrain)

                '''hurdle terrain'''
                # parkour_hurdle_terrain(terrain, platform_len=2.5, platform_height=0., num_stones=6, stone_len=0.3, x_range=[1.5, 2.4], y_range=[-0.4, 0.4], half_valid_width=[0.4, 0.8], hurdle_height_range=[0.05, 0.1], pad_width=0.1, pad_height=0.3, flat=False)

                '''stepping stones'''
                # stones_terrain(terrain, height_range=[0.05, 0.3])
                # add roughness
                # self.add_roughness(terrain, difficulty=2)
                # slope_terrain(terrain, slope_angle_range=[0.5, 1.5])

                # ! General Methods
                difficulty = 2
                 
                self.make_terrain(terrain, difficulty, types, random_terrain=True)

                # map coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = start_x + self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = start_y + self.width_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
                
                # set the origin of the terrain
                env_origin_x = i * self.env_length
                env_origin_y = j * self.env_width
                env_origin_z = 0.0
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

                types += 1
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg['height'][1] - self.cfg['height'][0]) * difficulty + self.cfg['height'][0]
        height = random.uniform(self.cfg['height'][0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg['downsampled_scale'])

    def make_terrain(self, terrain, difficulty, types, random_terrain=False):
        hurdle_height_range = [0.10 + (difficulty) * 0.01, 0.15 + (difficulty) * 0.02]
        gap_width_range = [0.10 + (difficulty) * 0.05, 0.15 + (difficulty) * 0.05]
        slope_angle_range = [0.40 + (difficulty) * 0.1, 0.50 + (difficulty) * 0.2]
        stone_height_range = [0.10 + (difficulty) * 0.035, 0.15 + (difficulty) * 0.035]
        hurdle_stone_len = 0.15 + 0.02 * (difficulty)
        types = types % 4

        # if(difficulty == 0):
        #     ''' simplest terrain (flat) '''
        #     return 
        
        # if (difficulty == 1):
        #     ''' roughness'''
        #     self.add_roughness(terrain, difficulty=1)
        #     return
        
        
        if (types == 0):
            ''' hurdle  '''
            hurdle_terrain(terrain, hurdle_height_range=hurdle_height_range, random_terrain=random_terrain, stone_len = hurdle_stone_len)
            self.add_roughness(terrain, difficulty=1)

            return
        if (types == 1):
            ''' gap  '''
            # hurdle_terrain(terrain, hurdle_height_range=hurdle_height_range, random_terrain=random_terrain)
            gap_terrain(terrain, gap_width_range=gap_width_range, random_terrain=random_terrain)
            self.add_roughness(terrain, difficulty=1)

            return
        if (types == 2):
            ''' slope  '''
            slope_terrain(terrain, slope_angle_range=slope_angle_range, random_terrain=random_terrain)
            # hurdle_terrain(terrain, hurdle_height_range=hurdle_height_range, random_terrain=random_terrain)
            self.add_roughness(terrain, difficulty=1)

            return
        if (types == 3):
            ''' stepping stones '''
            # hurdle_terrain(terrain, hurdle_height_range=hurdle_height_range, random_terrain=random_terrain)
            stones_terrain(terrain, height_range=stone_height_range, random_terrain=random_terrain)
            self.add_roughness(terrain, difficulty=1)

            return
        

def slope_terrain(terrain, wall_width=4, start2center=1.7, slope_angle_range=[0.2, 0.3], random_terrain=False):
    '''fixed slopes for robot can climb'''
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height = np.random.uniform(slope_angle_range[0], slope_angle_range[1])
    # print(max_height)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale)
    terrain_length = slope_start + 20
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)

    # flip the hights
    reversed_heights = heights[::-1]
    # terrain.height_field_raw[slope_start:terrain_length, 20:60] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

    # add slope opposite position in the terrain
    terrain_width = terrain.width
    slope_width_start = np.random.randint(20, 70)
    # Random
    # terrain.height_field_raw[slope_start+35:terrain_length+35, slope_width_start:slope_width_start+100] = heights[:, None]
    # terrain.height_field_raw[slope_start+terrain_length:terrain_length+terrain_length, slope_width_start:slope_width_start+100] = reversed_heights[:, None]
    # Fill
    num_slope = 3
    for i in range(num_slope):
        v_or_h = np.random.randint(0, 2)
        if not random_terrain:
            v_or_h = 0
        if not v_or_h:
            terrain.height_field_raw[slope_start:terrain_length, 5:195] = heights[:, None]
            terrain.height_field_raw[slope_start+20:terrain_length+20, 5:195] = reversed_heights[:, None]
        else:
            terrain.height_field_raw[5:195, slope_start:terrain_length] = heights[:, None].T
            terrain.height_field_raw[5:195, slope_start+20:terrain_length+20] = reversed_heights[:, None].T
        slope_start += 40
        terrain_length += 40
        # terrain.height_field_raw[slope_start+50:terrain_length+50, 5:195] = heights[:, None]
        # terrain.height_field_raw[slope_start+70:terrain_length+70, 5:195] = reversed_heights[:, None]
        # terrain.height_field_raw[slope_start+90:terrain_length+90, 5:195] = heights[:, None]
        # terrain.height_field_raw[slope_start+110:terrain_length+110, 5:195] = reversed_heights[:, None]
    # terrain.height_field_raw[slope_start+200:terrain_length+200, 3:78] = reversed_heights[:, None]
    # terrain.height_field_raw[slope_start+25:terrain_length+25, 40:78] = heights[:, None]
    # terrain.height_field_raw[slope_start+50:terrain_length+50, 40:78] = reversed_heights[:, None]
    # terrain.height_field_raw[slope_start+100:terrain_length+100, 20:50] = heights[:, None]
    # terrain.height_field_raw[slope_start+125:terrain_length+125, 20:50] = reversed_heights[:, None]
    # terrain.height_field_raw[slope_start+175:terrain_length+175, 3:15] = heights[:, None]
    # terrain.height_field_raw[slope_start+200:terrain_length+200, 3:15] = reversed_heights[:, None]
    # terrain.height_field_raw[slope_start+200:terrain_length+200, 20:60] = heights[:, None]


def gap_terrain(terrain, gap_width_range=[0.1, 0.2], platform_size=3.3, random_terrain=False):
    gap_size = np.random.uniform(gap_width_range[0], gap_width_range[1])
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
    
    start_x = platform_size
    # start_x = 2.0
    for i in range(1):
        y_start = np.random.randint(20, 70)
        # random
        # terrain.height_field_raw[start_x: start_x+gap_size, y_start : y_start+100] = -400
        v_or_h = np.random.randint(0, 2)
        if not random_terrain:
            v_or_h = 0
        if not v_or_h:
            terrain.height_field_raw[start_x: start_x+gap_size, 55: 150] = -400
        else:
            terrain.height_field_raw[55: 100, start_x: start_x+gap_size] = -400
        start_x += gap_size + platform_size
    # terrain.height_field_raw[center_x-x1 : center_x + x1, 3:78] = 0


def hurdle_terrain(terrain,
                           platform_len=1.0, 
                           platform_height=0., 
                           num_stones=5,
                           stone_len=0.3,
                           x_range=[1.2, 1.7],
                           y_range=[1.5, 4.5],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           flat=False,
                           random_terrain=False):
    # goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    # print(hurdle_height_range)
    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    # terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    # goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        y_start = np.random.randint(dis_y_min, dis_y_max)
        v_or_h = np.random.randint(0, 2)
        if not random_terrain:
            v_or_h = 0
        if not flat:
            # random
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, y_start:y_start+100] = np.random.randint(hurdle_height_min, hurdle_height_max)
            if v_or_h == 0:
                height = np.random.randint(hurdle_height_min, hurdle_height_max)
                # print(height)
                terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, 5:195] = height
            else:
                height = np.random.randint(hurdle_height_min, hurdle_height_max)
                # print(height)
                terrain.height_field_raw[5:195, dis_x-stone_len//2:dis_x+stone_len//2] = height


def stones_terrain(terrain, stone_size=0.8, stone_distance=1.0, height_range=[0.2,0.3], platform_size=0.8, depth=-1, random_terrain=False):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    low_height = int(height_range[0] / terrain.vertical_scale)
    max_height = int(height_range[1] / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(low_height, max_height, step=1)

    start_x = stone_distance + stone_size
    start_y = 0
    # terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_x < terrain.width-stone_size - stone_distance:
            stop_x = min(terrain.width, start_x + stone_size)
            y_start = np.random.randint(20, 70)
            # random
            # terrain.height_field_raw[start_x: stop_x, y_start: y_start+100] = np.random.choice(height_range)
            v_or_h = np.random.randint(0, 2)
            if not random_terrain:
                v_or_h = 0
            if not v_or_h:
                terrain.height_field_raw[start_x: stop_x, 5: 195] = np.random.choice(height_range)
            else:
                terrain.height_field_raw[5: 195, start_x: stop_x] = np.random.choice(height_range)
                # start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance
    elif terrain.width > terrain.length:
        print("width > length")
        while start_x < terrain.width-stone_size - stone_distance:
            stop_x = min(terrain.width, start_x + stone_size)
            terrain.height_field_raw[start_x: stop_x, 3: 78] = np.random.choice(height_range)
                # start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance


    return terrain
        
    
def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                B(x2,y2)
                /|
                / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0