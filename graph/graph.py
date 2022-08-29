import numpy as np


class Node(object):
    def __init__(self, info=None):
        self.node_num = None
        self.time_t = None
        self.neighbors = []
        self.neighbors_node_num = []
        self.embedding = None
        self.misc_info = None
        self.action = -1
        self.visited_time = []
        self.visited_memory = []
        if info is not None:
            for k, v in info.items():
                setattr(self, k, v)


class Graph(object):
    def __init__(self, cfg):
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = cfg.IMG_SHAPE
        self.feature_dim = cfg.memory.embedding_size
        self.M = cfg.memory.memory_size
        # self.torch_device = "cpu"#device

    def num_node(self):
        return len(self.node_position_list)

    # def num_node_max(self):
    #     return self.graph_mask.sum().max()

    def reset(self):
        self.node_position_list = []  # This position list is only for visualizations
        self.node_rotation_list = []  # This position list is only for visualizations

        self.graph_memory = np.zeros([self.M, self.feature_dim])
        # self.graph_act_memory = np.zeros([self.M], dtype=torch.uint8)
        self.graph_memory_pose = np.zeros([self.M, 3], dtype=np.uint8)
        self.graph_memory_map_pose = np.zeros([self.M, 3], dtype=np.uint8)

        self.A = np.zeros([self.M, self.M], dtype=np.bool)
        self.distance_mat = np.full([self.M, self.M], fill_value=float('inf'), dtype=np.float32)
        self.connectivity_mat = np.full([self.M, self.M], fill_value=0, dtype=np.float32)

        self.graph_mask = np.zeros(self.M)
        self.graph_time = np.zeros(self.M)

        self.pre_last_localized_node_idx = np.zeros([1], dtype=np.int32)
        self.last_localized_node_idx = np.zeros([1], dtype=np.int32)
        self.last_local_node_num = np.zeros([1])
        self.last_localized_node_embedding = np.zeros([self.feature_dim], dtype=np.float32)

    def initialize_graph(self, new_embeddings, positions, rotations, poses):
        self.add_node(node_idx=0, embedding=new_embeddings, time_step=0, position=positions, rotation=rotations, map_pose=poses)
        self.record_localized_state(node_idx=0, embedding=new_embeddings)

    def add_node(self, node_idx, embedding, time_step, position, rotation, map_pose, dists=None, connectivity=None):
        self.node_position_list.append(position)
        self.node_rotation_list.append(rotation)
        self.graph_memory[node_idx] = embedding
        # self.graph_act_memory[node_idx] = action
        self.graph_memory_map_pose[node_idx] = map_pose
        self.graph_mask[node_idx] = 1.0
        self.graph_time[node_idx] = time_step
        if dists is not None:
            self.distance_mat[node_idx, :node_idx] = dists
            self.distance_mat[:node_idx, node_idx] = dists
        if connectivity is not None:
            self.connectivity_mat[node_idx, :node_idx] = connectivity
            self.connectivity_mat[:node_idx, node_idx] = connectivity

    def record_localized_state(self, node_idx, embedding):
        self.pre_last_localized_node_idx = self.last_localized_node_idx
        self.last_localized_node_idx = node_idx
        self.last_localized_node_embedding = embedding

    def add_edge(self, node_idx_a, node_idx_b):
        self.A[node_idx_a, node_idx_b] = 1.0
        self.A[node_idx_b, node_idx_a] = 1.0
        return

    def add_edges(self, node_idx_as, node_idx_b):
        for node_idx_a in node_idx_as:
            self.A[node_idx_a, node_idx_b] = 1.0
            self.A[node_idx_b, node_idx_a] = 1.0
        return

    def update_node(self, node_idx, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[node_idx] = embedding
        self.graph_time[node_idx] = time_info
        return

    def update_nodes(self, node_indices, time_infos, embeddings=None):
        if embeddings is not None:
            self.graph_memory[node_indices] = embeddings
        self.graph_time[node_indices] = time_infos

    def get_positions(self, a=None):
        if a is None:
            return self.node_position_list
        else:
            return self.node_position_list[a]

    def get_neighbor(self, node_idx, return_mask=False):
        if return_mask:
            return self.A[node_idx]
        else:
            return np.where(self.A[node_idx])[0]

    def calculate_multihop(self, hop):
        return np.matrix_power(self.A[:, :self.num_node(), :self.num_node()].float(), hop)

class ObjectGraph(object):
    def __init__(self, cfg):
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.feature_dim = cfg.features.object_feature_dim
        self.M = cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS * cfg.memory.num_objects
        self.MV = cfg.memory.memory_size
        self.num_obj = cfg.memory.num_objects
        self.task = cfg['ARGS']['task']

    def num_node(self):
        return len(self.node_position_list)

    def reset(self):
        self.node_position_list = []  # This position list is only for visualizations

        self.graph_memory = np.zeros([self.M, self.feature_dim])
        self.graph_memory_pose = np.zeros([self.M, 3])
        self.graph_category = np.zeros([self.M])
        self.graph_score = np.zeros([self.M])
        self.graph_map_pose = np.zeros([self.M, 3]) #x1y1x2y2
        self.graph_id = np.zeros([self.M])

        self.A_OV = np.zeros([self.M, self.MV], dtype=np.bool)

        self.graph_mask = np.zeros(self.M)
        self.graph_time = np.zeros([self.M], dtype=np.int32)

        self.last_localized_node_idx = 0

    def initialize_graph(self, new_embeddings, object_scores, object_categories, object_ids, masks, positions, map_poses):
        if sum(masks == 1) == 0:
            masks[0] = 1
        self.add_node(node_idx=0, embedding=new_embeddings, object_score=object_scores, object_category=object_categories, object_id=object_ids, time_step=0, mask=masks, position=positions, map_pose=map_poses,
                      vis_node_idx=0)

    def add_node(self, node_idx, embedding, object_score, object_category, object_id, mask, time_step, position, map_pose, vis_node_idx):
        node_idx_ = node_idx
        i = 0
        while True:
            if self.task == "objgoalnav":
                cond = mask[i] == 1 and np.all(np.sqrt(np.sum((position[i] - self.graph_memory_pose) ** 2, 1)) > 0.5)
            else:
                cond = mask[i] == 1
            if cond:
                self.node_position_list.append(position[i])
                self.graph_memory[node_idx_] = embedding[i]
                self.graph_memory_pose[node_idx_] = position[i]
                self.graph_map_pose[node_idx_] = map_pose[i]
                self.graph_score[node_idx_] = object_score[i]
                self.graph_category[node_idx_] = object_category[i]
                self.graph_id[node_idx_] = object_id[i]
                self.graph_mask[node_idx_] = 1.0
                self.graph_time[node_idx_] = time_step
                self.add_vo_edge([node_idx_], vis_node_idx)
                node_idx_ += 1
            i += 1
            if i == len(position):
                break

    def add_vo_edge(self, node_idx_obj, curr_vis_node_idx):
        for node_idx_obj_i in node_idx_obj:
            self.A_OV[node_idx_obj_i, curr_vis_node_idx] = 1.0

    def update_node(self, node_idx, time_info, node_score, node_category, node_id, curr_vis_node_idx, position, map_pose, embedding=None):
        if embedding is not None:
            self.graph_memory[node_idx] = embedding
        self.graph_memory_pose[node_idx] = position
        self.graph_map_pose[node_idx] = map_pose
        self.graph_score[node_idx] = node_score
        self.graph_category[node_idx] = node_category
        self.graph_id[node_idx] = node_id
        self.graph_time[node_idx] = time_info
        self.A_OV[node_idx, curr_vis_node_idx] = 1

    def get_positions(self, a=None):
        if a is None:
            return self.node_position_list
        else:
            return self.node_position_list[a]