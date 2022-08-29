import numpy as np
import torch.nn as nn
import torch
from graph.PCL.resnet_pcl import resnet18
from graph.PCL.resnet_obj_pcl import resnet18 as resnet18_obj_pcl
from torchvision.models import resnet18 as resnet18_rgb
import os
from graph.graph import Graph, ObjectGraph
from PIL import Image
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
import imageio, cv2, time, joblib
from copy import deepcopy


class GraphEnv():
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.project_dir = args.project_dir
        self.img_node_th = args.img_node_th
        self.obj_node_th = args.obj_node_th
        self.torch_device = "cuda:0"
        self.graph = Graph(args)
        self.objectgraph = ObjectGraph(args)
        self.visual_encoder = self.load_visual_encoder(args.image_feature_dim)
        self.object_encoder = self.load_object_encoder(args.object_feature_dim)
        self.reset_all_memory()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        eval_augmentation = [
            transforms.ToTensor(),
            normalize
        ]
        self.transform_eval = transforms.Compose(eval_augmentation)
        self.episode_iter = -1

    def load_visual_encoder(self, feature_dim):
        dn = "gibson"
        if self.args.noisydepth:
            visual_encoder = resnet18(num_classes=feature_dim)
            dim_mlp = visual_encoder.fc.weight.shape[1]
            visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
            ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'PCL_encoder_{dn}_noisydepth.pth.tar')
            ckpt = torch.load(ckpt_pth, map_location='cpu')
            visual_encoder.load_state_dict({k[len('module.encoder_k.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_k.' in k})
        elif self.args.nodepth:
            visual_encoder = resnet18_rgb(num_classes=feature_dim)
            dim_mlp = visual_encoder.fc.weight.shape[1]
            visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
            ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'PCL_encoder_{dn}_nodepth.pth.tar')
            ckpt = torch.load(ckpt_pth, map_location='cpu')
            visual_encoder.load_state_dict({k[len('module.encoder_q.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_q.' in k})
        else:
            visual_encoder = resnet18(num_classes=feature_dim)
            dim_mlp = visual_encoder.fc.weight.shape[1]
            visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
            ckpt_pth = os.path.join(self.project_dir, 'model/PCL', 'PCL_encoder.pth')
            ckpt = torch.load(ckpt_pth, map_location='cpu')
            visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval().to(self.torch_device)
        return visual_encoder

    def load_object_encoder(self, feature_dim):
        dataset = "gibson"
        object_encoder = resnet18_obj_pcl(num_classes=feature_dim)
        dim_mlp = object_encoder.fc.weight.shape[1]
        object_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), object_encoder.fc)
        ckpt_pth = os.path.join(self.project_dir, 'model/PCL', f'ObjPCL_{dataset.split("_")[0]}.pth.tar')
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        state_dict = {k[len('module.encoder_k.'):]: v for k, v in ckpt['state_dict'].items() if 'module.encoder_k.' in k}
        object_encoder.load_state_dict(state_dict)
        object_encoder.eval().to(self.torch_device)
        return object_encoder

    def reset_all_memory(self, B=None):
        self.graph.reset()
        self.objectgraph.reset()

    def is_close(self, embed_a, embed_b, return_prob=False, th=0.75):
        # with torch.no_grad():
        logits = np.matmul(embed_a, embed_b.transpose(1, 0))  # .squeeze()
        close = (logits > th)
        if return_prob:
            return close, logits
        else:
            return close

    def update_object_graph(self, object_embedding, obs, done):
        object_score, object_category, object_id, object_mask, object_position, object_map_pose, object_bboxes, time = obs['object_score'], obs['object_category'], obs['object_id'], obs['object_mask'], \
                                                                                           obs['object_pose'], obs['object_map_pose'], obs['object'], obs['step']
        # The position is only used for visualizations. Remove if object features are similar
        # 물체 masking
        object_score = object_score[object_mask==1]
        object_category = object_category[object_mask==1]
        object_id = object_id[object_mask==1]
        object_position = object_position[object_mask==1]
        # object_bboxes = object_bboxes[object_mask==1]
        object_map_pose = object_map_pose[object_mask==1]
        object_embedding = object_embedding[object_mask==1]
        object_mask = object_mask[object_mask==1]
        #     def initialize_graph(self, new_embeddings, object_scores, object_categories, object_ids, masks, positions, map_poses):
        if done:
            self.objectgraph.reset()
            self.objectgraph.initialize_graph(object_embedding, object_score, object_category, object_id, object_mask, object_position, object_map_pose)

        # 만약 vis node가 메모리 노드에 localize 됐을시 해당 메모리 노드에 있는 물체들과 현재 물체들을 비교하여
        # 현재 물체에 새로운 물체가 있을 경우 추가해주고
        # 과거에 있던 물체가 사라졌을 경우 노드를 없애자
        if self.args.update_object_graph:
            if self.found_in_memory:
                # 메모리 물체 인코딩
                memory_obj = np.where(self.objectgraph.A_OV[:, self.last_localized_node_idx])[0]
                if len(memory_obj) > 0:
                    memory_obj_feat = self.objectgraph.graph_memory[memory_obj]
                    # memory_obj_score = self.objectgraph.graph_score[memory_obj]
                    memory_obj_cat = self.objectgraph.graph_category[memory_obj].astype(np.int32)
                    memory_obj_with_cat = np.concatenate((np.eye(80)[memory_obj_cat], memory_obj_feat), -1)
                    memory_obj_with_cat = memory_obj_with_cat / np.sum(memory_obj_with_cat**2, -1)[...,None] ** 0.5
                    # memory_obj_with_cat = nn.functional.normalize(memory_obj_with_cat, dim=-1).cpu().detach().numpy()

                    # 현재 물체 인코딩
                    curr_obj_with_cat =np.concatenate((np.eye(80)[object_category.astype(np.int32)], object_embedding), -1)
                    curr_obj_with_cat = curr_obj_with_cat / np.sum(curr_obj_with_cat**2, -1)[...,None] ** 0.5
                    # curr_obj_with_cat = nn.functional.normalize(curr_obj_with_cat, dim=-1).cpu().detach().numpy()

                    # hungarian matching 으로 물체들 비교
                    # prob = np.matmul(memory_obj_feat, object_embedding[object_mask==1].transpose(1, 0)) #메모리 x 현재
                    prob_cat = np.matmul(memory_obj_with_cat, curr_obj_with_cat.transpose(1, 0)) #메모리 x 현재
                    match_pair = np.stack(linear_sum_assignment(1 - prob_cat), 1)
                    match_score = prob_cat[match_pair[:, 0], match_pair[:, 1]]
                    matched_mem_obj = match_pair[:, 0]
                    matched_cur_obj = match_pair[:, 1]
                    memory_obj_idx = np.arange(len(memory_obj))

                    # 물체가 사라졌을 경우 (다시 detect 되지 않은 경우) 노드 score를 낮춰보자
                        # 메모리에는 현재 위치에서는 보이지 않는 그 장소의 물체들을 갖고 있을 수 있다
                        # 그렇기때문에 물체가 사라졌을 경우를 판단하는게 쉽지 않은데....
                    not_matched_memory_obj = np.setdiff1d(memory_obj_idx, matched_mem_obj)
                    if len(not_matched_memory_obj) > 0:
                        self.objectgraph.graph_score[memory_obj[not_matched_memory_obj]] = np.clip(self.objectgraph.graph_score[memory_obj[not_matched_memory_obj]]-0.2, 0.001, 1.)

                #새로운 물체가 나타났을때는 어차피 뒤에서 추가하니깐 괜춘

            # 만약 직전 노드에 계속 localize 되고 있는 경우라면 이전에 detection된 물체들이 다시 들어오는 경우가 많다
            # 지속적으로 detection 되는 물체들만 남겨두고 나머지는 삭제하는 것이 좋을 듯
            if self.found_prev:
                pass

        if self.config.OBJECTGRAPH.SPARSE:
            not_found = ~self.found  # Sparse
        else:
            not_found = not done  # Dense
        to_add = [True] * int(sum(object_mask))
        if not_found:
            hop1_vis_node = self.graph.A[self.graph.last_localized_node_idx]
            hop1_obj_node_mask = np.sum(self.objectgraph.A_OV.transpose(1, 0)[hop1_vis_node], 0) > 0
            curr_obj_node_mask = self.objectgraph.A_OV[:, self.graph.last_localized_node_idx]
            neighbor_obj_node_mask = (hop1_obj_node_mask + curr_obj_node_mask) > 0
            neighbor_node_embedding = self.objectgraph.graph_memory[neighbor_obj_node_mask]
            # memory_node_embedding = self.objectgraph.graph_memory[curr_obj_node_mask]
            neighbor_obj_memory_idx = np.where(neighbor_obj_node_mask)[0]
            neighbor_obj_memory_score = self.objectgraph.graph_score[neighbor_obj_memory_idx]
            neighbor_obj_memory_cat = self.objectgraph.graph_category[neighbor_obj_memory_idx]
            # print([DETECTION_CATEGORIES[int(neighbor_obj_memory_cat_)] for neighbor_obj_memory_cat_ in neighbor_obj_memory_cat])

            close, prob = self.is_close(neighbor_node_embedding, object_embedding, return_prob=True, th=self.obj_node_th)
            for c_i in range(prob.shape[1]):
                close_mem_indices = np.where(close[:, c_i] == 1)[0]
                # detection score 높은 순으로 체크
                for m_i in close_mem_indices:
                    is_same = False
                    to_update = False
                    # m_i = neighbor_obj_memory_idx[close_idx]
                    if (object_category[c_i] == neighbor_obj_memory_cat[m_i]) and object_category[c_i] != -1:
                        is_same = True
                        if object_score[c_i] >= neighbor_obj_memory_score[m_i]:
                            to_update = True

                    if is_same:
                        # 만약 새로 detect한 물체가 이미 메모리에 있는 물체라면 새로 추가하지 않는다
                        to_add[c_i] = False

                    if to_update:
                        # 만약 새로 detect한 물체가 이미 메모리에 있는 물체고 새로 detect한 물체의 score가 높다면 메모리를 새 물체로 업데이트 해준다
                        # def update_node(self, node_idx, time_info, node_score, node_category, node_id, curr_vis_node_idx, position, map_pose, embedding=None):
                        self.objectgraph.update_node(m_i, time, object_score[c_i], object_category[c_i], object_id[c_i], int(self.graph.last_localized_node_idx), object_position[c_i], object_map_pose[c_i],
                                                     object_embedding[c_i])
                        break

            # Add new objects to graph
            if sum(to_add) > 0:
                start_node_idx = self.objectgraph.num_node()
                new_idx = np.where(np.stack(to_add))[0]
                self.objectgraph.add_node(start_node_idx, object_embedding[new_idx], object_score[new_idx],
                                          object_category[new_idx], object_id[new_idx], object_mask[new_idx], time,
                                          object_position[new_idx], object_map_pose[new_idx], int(self.graph.last_localized_node_idx))

    def update_image_graph(self, new_embedding, curr_obj_embeding, obs, done):
        # The position is only used for visualizations.
        position, rotation, map_pose, time = obs['position'], obs['rotation'], obs['map_pose'], obs['step']
        if done:
            self.graph.reset()
            self.graph.initialize_graph(new_embedding, position, rotation, map_pose)

        obj_close = True
        obj_graph_mask = self.objectgraph.graph_score[self.objectgraph.A_OV[:, self.graph.last_localized_node_idx]] > 0.5
        if len(obj_graph_mask) > 0:
            curr_obj_mask = obs['object_score'] > 0.5
            if np.sum(curr_obj_mask) / len(curr_obj_mask) >= 0.5:
                close_obj, prob_obj = self.is_close(self.objectgraph.graph_memory[self.objectgraph.A_OV[:, self.graph.last_localized_node_idx]], curr_obj_embeding, return_prob=True, th=self.obj_node_th)
                close_obj = close_obj[obj_graph_mask, :][:, curr_obj_mask]
                category_mask = self.objectgraph.graph_category[self.objectgraph.A_OV[:, self.graph.last_localized_node_idx]][obj_graph_mask][:, None] == obs['object_category'][curr_obj_mask]
                close_obj[~category_mask] = False
                if len(close_obj) >= 3:
                    clos_obj_p = close_obj.any(1).sum() / (close_obj.shape[0])
                    if clos_obj_p < 0.1:  # Fail to localize (find the same object) with the last localized frame
                        obj_close = False

        close, prob = self.is_close(self.graph.last_localized_node_embedding[None], new_embedding[None], return_prob=True, th=self.img_node_th)
        # print("image prob", prob[0])

        found = (np.array(done) + close.squeeze()) & np.array(obj_close).squeeze()
        # found = np.array(done) + close.squeeze()  # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        self.found_prev = False
        self.found = found
        if found:
            self.graph.update_nodes(self.graph.last_localized_node_idx, time)
            self.found_prev = True
        self.found_in_memory = False
        # 모든 메모리 노드 체크
        check_list = 1 - self.graph.graph_mask[:self.graph.num_node()]
        # 바로 직전 노드는 체크하지 않는다.
        check_list[self.graph.last_localized_node_idx] = 1.0
        if found: #localize 됐다면 다른 메모리 노드들을 체크하지 않는다
            check_list = np.ones_like(check_list)
        to_add = False
        while not found:
            not_checked_yet = np.where((1 - check_list))[0]
            neighbor_embedding = self.graph.graph_memory[not_checked_yet]
            num_to_check = len(not_checked_yet)
            if num_to_check == 0:
                # 과거의 노드와도 다르고, 메모리와도 모두 다르다면 새로운 노드로 추가
                to_add = True
                break
            else:
                # 메모리 노드에 존재하는지 체크
                close, prob = self.is_close(new_embedding[None], neighbor_embedding, return_prob=True, th=self.img_node_th)
                close = close[0];  prob = prob[0]
                close_idx = np.where(close)[0]
                if len(close_idx) >= 1:
                    found_node = not_checked_yet[prob.argmax()]
                else:
                    found_node = None
                if found_node is not None:
                    found = True
                    # 메모리 노드에 존재하고 그게 직전 노드가 아니라면 노드를 업데이트 해준다
                    self.found_prev = True
                    if found_node != self.graph.last_localized_node_idx:
                        self.found_prev = False
                        if abs(time - self.graph.graph_time[found_node]) > 10:
                            self.found_in_memory = True #만약 새롭게 찾은 노드가 오랜만에 돌아온 노드라면 found_in_memory를 True로 바꿔준다
                        self.graph.update_node(found_node, time, new_embedding)
                        self.graph.add_edge(found_node, self.graph.last_localized_node_idx)
                        self.graph.record_localized_state(found_node, new_embedding)
                    else:
                        print("aa")
                check_list[found_node] = 1.0

        if to_add:
            new_node_idx = self.graph.num_node()
            self.graph.add_node(new_node_idx, new_embedding, time, position, rotation, map_pose)
            self.graph.add_edge(new_node_idx, self.graph.last_localized_node_idx)
            self.graph.record_localized_state(new_node_idx, new_embedding)
        self.last_localized_node_idx = self.graph.last_localized_node_idx

    def embed_obs(self, obs):
        with torch.no_grad():
            if self.args.nodepth:
                img_tensor = (torch.tensor(obs['panoramic_rgb'][None]).to(self.torch_device).float() / 255).permute(0, 3, 1, 2)
            else:
                img_tensor = torch.cat((torch.tensor(obs['panoramic_rgb'][None]).to(self.torch_device).float() / 255.,
                                        torch.tensor(obs['panoramic_depth'][None]).to(self.torch_device).float()), 3).permute(0, 3, 1, 2)
            vis_embedding = nn.functional.normalize(self.visual_encoder(img_tensor).view(-1, self.args.image_feature_dim), dim=1)
        return vis_embedding[0].cpu().detach().numpy()

    def embed_object(self, obs):
        with torch.no_grad():
            im = Image.fromarray(np.uint8(obs['panoramic_rgb']))
            img_tensor = self.transform_eval(im)[None].to(self.torch_device)
            feat = self.object_encoder(img_tensor, torch.tensor(obs['object']).to(self.torch_device).float()[None])
            obj_embedding = nn.functional.normalize(feat, dim=-1)
        return obj_embedding[0].cpu().detach().numpy()

    def build_graph(self, obs, reset=False):
        if not reset:
            obs, reward, done, info = obs
        curr_vis_embeddings = self.embed_obs(obs)
        curr_object_embedding = self.embed_object(obs)
        if reset:
            self.graph.reset()
            self.graph.initialize_graph(curr_vis_embeddings, obs['noisy_position'], obs['rotation'], obs['map_pose'])
        else:
            self.update_image_graph(curr_vis_embeddings, curr_object_embedding, obs, done=False)
        global_memory_dict = self.get_global_memory()
        if reset:
            self.objectgraph.reset()
            self.objectgraph.initialize_graph(curr_object_embedding, obs['object_score'], obs['object_category'], obs['object_id'], obs['object_mask'], obs['object_pose'],  obs['object_map_pose'])
        else:
            self.update_object_graph(curr_object_embedding, obs, done=False)
        object_memory_dict = self.get_object_memory()
        obs = self.add_memory_in_obs(obs, global_memory_dict, object_memory_dict)
        # if self.args.render:
        #     self.draw_graphs()
        #     self.draw_semantic_map_()
        #     self.render('human')
        return obs

    def reset(self):
        if self.args.record > 0:
            self.records = []
            self.record_graphs = []
            self.record_maps = []
            self.record_local_maps = []
            self.record_objects = []
            self.imgs = []
            self.episode_iter += 1
        obs_list = super().reset()
        obs = self.build_graph(obs_list, reset=True)
        # if self.args.render:
        #     self.draw_graphs()
        #     self.render('human')
        return obs

    def add_memory_in_obs(self, obs, memory_dict, object_memory_dict):
        # add memory to obs
        obs.update(memory_dict)
        obs.update(object_memory_dict)
        obs.update({'object_localized_idx': self.objectgraph.last_localized_node_idx})
        obs.update({'localized_idx': self.graph.last_localized_node_idx})
        if 'distance' in obs.keys():
            obs['distance'] = obs['distance']  # .unsqueeze(1)
        return obs

    def step(self, action):
        with torch.no_grad():
            local_step = 0
            g_reward = 0
            stop_epi = False
            effective_step = 0
            reset_global_step = True
            while True:
                obs, reward, done, info = super().step(action, reset_global_step, stop_epi)
                # if self.args.render:
                #     self.render('human')
                if self.args.record > 0:
                    if done:
                        self.save_record(info)
                curr_vis_embedding = self.embed_obs(obs)
                curr_object_embedding = self.embed_object(obs)
                self.update_image_graph(curr_vis_embedding, curr_object_embedding, obs, done)
                global_memory_dict = self.get_global_memory()
                self.update_object_graph(curr_object_embedding, obs, done)
                object_memory_dict = self.get_object_memory()
                obs = self.add_memory_in_obs(obs, global_memory_dict, object_memory_dict)

                if self.args.render > 0 or self.args.record > 0:
                    # self.draw_graphs()
                    # self.draw_semantic_map_()
                    if not done:
                        self.stack_records(action, obs, info)
                if info['terminate_local_nav'] or stop_epi:# or info['replan']:
                    stop_epi = True
                    pass
                else:
                    g_reward += reward
                    effective_step += 1
                if done:
                    break
                reset_global_step = False
                local_step += 1
                if local_step == self.num_local_steps or stop_epi:
                    break

        info['local_step'] = effective_step
        # g_reward += (info['lg_start_dist']-info['lg_dist'])/(info['lg_start_dist']+0.0001) * (info['lg_start_dist']/10.) * 0.1
        return obs, g_reward, done, info

    def save_record(self, info):
        scene_name = self.current_episode.scene_id.split('/')[-1][:-4]
        episode_id = self.episode_iter
        if self.args.record > 0:
            spl = info['spl']
            if np.isnan(spl):
                spl = 0.0
                print('spl nan!', self.habitat_env._sim.geodesic_distance(self.current_episode.start_position, self.current_episode.goals[0].position))
            if np.isinf(spl):
                spl = 0.0
            ep_list = {
                'house': scene_name,
                'ep_id': self.current_episode.episode_id,
                'start_pose': [list(self.current_episode.start_position), list(self.current_episode.start_rotation)],
                'total_step': self.timestep,
                'success': info['success'],
                'spl': spl,
                'distance_to_goal': info['distance_to_goal'],
                'goal_name': self.goal_name,
                'goals':self.current_episode.goals,
            }
            video_name = os.path.join(VIDEO_DIR, '%04d_%s_success=%.1f_spl=%.1f.mp4' % (episode_id, scene_name, info['success'], spl))
            with imageio.get_writer(video_name, fps=30) as writer:
                im_shape = self.imgs[-1].shape
                for im in self.imgs:
                    if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                        im = cv2.resize(im, (im_shape[1], im_shape[0]))
                    writer.append_data(im.astype(np.uint8))
                writer.close()
            if self.args.record > 1:
                file_name = os.path.join(OTHER_DIR, '%04d_%s_data_success=%.1f_spl=%.1f.dat.gz' % (episode_id, scene_name, info['success'], spl))
                data = {'position': self.records, 'graph': self.record_graphs, 'map': self.record_maps, 'local_map': self.record_local_maps, 'episode': ep_list, 'objects': self.record_objects}
                joblib.dump(data, file_name)
                del data

    def stack_records(self, action, obs, info=None):
        if self.args.record > 1:
            self.records.append([self.habitat_env._sim.get_agent_state().position,  self.habitat_env._sim.get_agent_state().rotation.components, action])
            if hasattr(self.mapper, 'node_list'):
                max_num_node = self.graph.num_node()
                global_memory_dict = {
                    'global_memory': self.graph.graph_memory[:, :max_num_node],
                    'global_memory_pose': np.stack(self.graph.node_position_list),
                    'global_memory_map_pose': self.graph.graph_memory_map_pose[:, :max_num_node],
                    'global_mask': self.graph.graph_mask[:, :max_num_node],
                    'global_A': self.graph.A[:, :max_num_node, :max_num_node],
                    'global_idx': self.graph.last_localized_node_idx,
                    'global_time': self.graph.graph_time[:, :max_num_node]
                }
                max_num_vis_node = self.graph.num_node()
                max_num_node = self.objectgraph.num_node()
                object_memory_dict = {
                    'object_memory': self.objectgraph.graph_memory[:, :max_num_node],
                    'object_memory_pose': np.stack(self.objectgraph.node_position_list[0]),
                    'object_memory_score': self.objectgraph.graph_score[:, :max_num_node],
                    'object_memory_map_pose': self.objectgraph.graph_map_pose[:, :max_num_node],
                    'object_memory_category': self.objectgraph.graph_category[:, :max_num_node],
                    'object_memory_mask': self.objectgraph.graph_mask[:, :max_num_node],
                    'object_memory_A_OV': self.objectgraph.A_OV[:, :max_num_node, :max_num_vis_node],
                    'object_memory_time': self.objectgraph.graph_time[:, :max_num_node]
                }
                global_memory_dict.update(object_memory_dict)
                self.record_graphs.append(global_memory_dict)
                self.record_objects.append({
                    "object": deepcopy(obs['object'][0][:, 1:]),
                    "object_score": deepcopy(obs['object_score'][0]),
                    "object_category": deepcopy(obs['object_category'][0]),
                    "object_pose": deepcopy(obs['object_pose'][0])
                })
            if info is not None:
                self.record_maps.append({
                    'agent_angle': deepcopy(info['ortho_map']['agent_rot']),
                    'agent_loc': deepcopy(info['ortho_map']['agent_loc']),
                })
                try:
                    local_map, fmm_dist, prev_poses, map_loc, stg_loc, map_resolution, goal = self.get_local_map_info()
                    self.record_local_maps.append({
                        'local_map': deepcopy(local_map),
                        'fmm_dist': deepcopy(fmm_dist),
                        'prev_poses': deepcopy(prev_poses),
                        'map_loc': deepcopy(map_loc),
                        'stg_loc': deepcopy(stg_loc),
                        'map_resolution': deepcopy(map_resolution),
                        'goal': deepcopy(goal),
                        'timestep': self.timestep,
                        'subgoal_position': np.stack(self.subgoal_position)
                    })
                except:
                    self.record_local_maps.append({
                        'timestep': self.timestep
                    })

            else:
                lower_bound, upper_bound = self.habitat_env._sim.pathfinder.get_bounds()
                self.record_maps.append({
                    'ortho_map': deepcopy(self.ortho_rgb),
                    'P': deepcopy(np.array(self.P)),
                    'target_loc': np.array(self.habitat_env._current_episode.goals[0].position),
                    'lower_bound': deepcopy(lower_bound),
                    'upper_bound': deepcopy(upper_bound),
                    'WIDTH': self.habitat_env._config.SIMULATOR.ORTHO_RGB_SENSOR.WIDTH,
                    'HEIGHT': self.habitat_env._config.SIMULATOR.ORTHO_RGB_SENSOR.HEIGHT
                })
                self.record_local_maps.append({
                    'timestep': self.timestep
                })
        if self.args.record > 0:
            img = obs['panoramic_rgb'][0]
            self.imgs.append(img)

    def get_global_memory(self):
        # num_node = self.graph.num_node()
        global_memory_dict = {
            'global_memory': self.graph.graph_memory,
            'global_memory_pose': self.graph.graph_memory_pose,
            'global_memory_map_pose': self.graph.graph_memory_map_pose,
            'global_mask': self.graph.graph_mask,
            'global_A': self.graph.A,
            'global_idx': self.graph.last_localized_node_idx,
            'global_time': self.graph.graph_time
        }
        return global_memory_dict

    def get_object_memory(self):
        # num_node = self.objectgraph.num_node()
        object_memory_dict = {
            'object_memory': self.objectgraph.graph_memory,
            'object_memory_score': self.objectgraph.graph_score,
            'object_memory_pose': self.objectgraph.graph_memory_pose,
            'object_memory_map_pose': self.objectgraph.graph_map_pose,
            'object_memory_category': self.objectgraph.graph_category,
            'object_memory_mask': self.objectgraph.graph_mask,
            'object_memory_A_OV': self.objectgraph.A_OV,
            'object_memory_time': self.objectgraph.graph_time
        }
        return object_memory_dict