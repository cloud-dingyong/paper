import sys

sys.path.append('sim/')
sys.path.append('../sim/')
sys.path.append('../')
# from important_ops_simulator import ImportantOpsSimulator
# from grouper.group_pruner import neighbor_merge_pruner
import pickle
import networkx as nx
from sim.tf_placement_sim.tf_pl_simulator import ImportantOpsSimulator
from hierarchical.hierarchical_controller_test import HierarchicalControllerTest
from tensorflow.python.grappler import item as gitem
from tensorflow.python.framework import errors
from hierarchical.readfile import ReadFile


class PPTFItem(object):

    def __init__(self, pickled_inp_file, n_devs, simplify_tf_reward_model=False,
                 final_size=None, use_new_sim=False, sim_mem_usage=False):
        # todo 1
        device_names = ['/device:GPU:%d' % i for i in range(n_devs)]
        gpu_devices = filter(lambda dev: 'GPU' in dev, device_names)
        gpu_devices = list(sorted(gpu_devices)) #  GPU设备的列表
        # with里面的部分都是去读取数据集文件，获得计算图和设备执行性能监控的相关信息
        # TODO 这里可以查询一下有关配置文件的相关信息是什么意思
        # gcluster = HierarchicalControllerTest(n_devs, allow_soft_placement=True, disable_detailed_stats=False, disable_timeline=False,).getCluster()
        # rf = ReadFile(pickled_inp_file=pickled_inp_file, gcluster=gcluster, reset_devs=False)
        # mg, G, ungroup_map = rf.get_triple()
        # item = gitem.Item(mg, ignore_colocation=False, ignore_user_placement=False)
        try:
            """返回指定item的训练成本
                Args:
                  item: 需要衡量运行成本的item
                Returns: 返回三元组，分别是op_perfs, runtime, step_stats.
                """
            # 在此处op_perfs和step_stats后续没用到
            # op_perf, original_run_time, step_stats = gcluster.MeasureCosts(item)
            # step_stats = rf.get_stats()
            cost_d, out_d, temp_mem, mem_info = [None] * 4
            # print("Runtime for original placement: " + str(original_run_time))
        except errors.OpError as e:
            print("Original placement isn't feasible: " + str(e))

        # todo 2
        with open(pickled_inp_file, 'rb') as f:
            j = pickle.load(f)

            # 其中mg的结构为MetaGraphDef，是tensorflow计算图的高级数据结构，包含静态结构信息和元数据信息，可导出信息
            # 其中mg的结构为MetaGraphDef 持久化数据格式，通过saver.export_meta_graph，应该是ENVS得到的神经网络图结构
            # ungroup_map是有关于分组的依据, 计算性能和跑单步的数据
            mg, G, ungroup_map = j['optim_mg'], j['G'], j['ungrouped_mapping']

            if 'op_perf' in j:
                # 这些的都是可以交个tensorflow自动生成的
                op_perf, step_stats = j['op_perf'], j['step_stats']
                cost_d, out_d, temp_mem, mem_info = [None] * 4
            else:
                cost_d, out_d, temp_mem = j['cost_d'], j['out_d'], j['temp_mem']
                mem_info = j['mem_info']
                op_perf, step_stats = None, None

        if len(nx.get_node_attributes(G, 'mem')) == 0:
            # 设置图中mem的属性
            G = correct_mem_param(G, step_stats, ungroup_map)

        # if final_size is not None:
        #     if len(G) > final_size:
        #       G, ungroup_map = neighbor_merge_pruner(G, ungroup_map, final_size)

        if simplify_tf_reward_model:
            assert False
            d = {}
            for n in G.nodes(): d[n] = 1
            # 设置节点cost，out_size和mem的相关属性
            nx.set_node_attributes(G, d, 'cost')
            nx.set_node_attributes(G, d, 'out_size')
            nx.set_node_attributes(G, d, 'mem')

        self.mg = mg
        self.ungroup_map = ungroup_map
        self.n_devs = n_devs
        self.gpu_devs = gpu_devices
        self.use_new_sim = use_new_sim
        self.grouped_G = G
        self.sim_mem_usage = sim_mem_usage
        self.cost_d, self.out_d = cost_d, out_d
        # self.gcluster = gcluster

        if self.use_new_sim:
            # 将模拟器新建出来注入到PPItem当中，模拟器中的一切准备工作都 就绪了，所有节点属性和父子节点
            self.sim = ImportantOpsSimulator(mg, op_perf, step_stats, device_names)
            # cost_d=cost_d, out_d=out_d, temp_mem=temp_mem,
            # mem_info=mem_info)
        else:
            raise Exception('Using old simulator is locked out by default')
            # from old_simulator import LegacySimulator
            # self.old_sim = LegacySimulator(self.cluster, True)

    """
      This function takes a dictionary from node-center-names to their placements
      To ungroup the placements, we first map each node to its corresponding group index
      and then map grp-index to the name. Finally, this mapped name is looked up in the input
      placement dictionary to find out placement of all the ops.
    """

    def new_simulate(self, p):

        for n in p:
            p[n] = str(p[n])

        ungrouped_pl = self.ungroup_pl(p)

        return self.simulate_full_pl(ungrouped_pl)

    def simulate_full_pl(self, ungrouped_pl):

        if self.sim_mem_usage:

            run_time, start_times, mem_utils = self.sim.simulate(ungrouped_pl, sim_mem_usage=True)

            return run_time / 1e6, start_times, mem_utils

        else:

            run_time, start_times = self.sim.simulate(ungrouped_pl)

            run_time /= 1e6  # convert to secs

            return run_time, start_times, None

    def simulate_without_ungroup(self, p):

        for n in p:
            p[n] = str(p[n])

        for op in self.mg.graph_def.node:
            if op.name not in p:
                raise Exception('Without ungrouping needs \
                  specifying full placement')

        return self.simulate_full_pl(p)

    def ungroup_pl(self, pl):
        ungroup_map = self.ungroup_map
        ungrouped_pl = {}

        for op in self.mg.graph_def.node:
            grp_ctr = ungroup_map[op.name]
            ungrouped_pl[op.name] = pl[grp_ctr]

        return ungrouped_pl

    def generate_placement(self, pl):
        ungroup_map = self.ungroup_map
        mg = self.mg
        for op in mg.graph_def.node:
            grp_ctr = ungroup_map[op.name]
            op.device = self.gpu_devs[pl[grp_ctr]]
        return mg

    # def degenerate_placement(self, mg, null_placement):
    #     for op in mg.graph_def.node:
    #



    def get_grouped_graph(self):
        return self.grouped_G

    def get_ungroup_map(self):
        return self.ungroup_map

    def simulate(self, *args, **kwargs):
        if self.use_new_sim:
            return self.new_simulate(*args, **kwargs)
        else:
            raise Exception('No longer supported')
            return self.old_simulate(*args, **kwargs)


def ungroup_map_to_adj_l(ungroup_map):
    d = {}
    for k, v in ungroup_map.items():
        if v in d:
            d[v].append(k)
        else:
            d[v] = [k]
    return d


def correct_mem_param(G, step_stats, ungroup_map):
    out_d_mg = {}
    for dev_stats in step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
            node = node_stats.node_name
            for output in node_stats.output:
                allocation = output.tensor_description.allocation_description
                num_bytes = allocation.requested_bytes
                out_d_mg[node] = num_bytes
                break

        adj = ungroup_map_to_adj_l(ungroup_map)
        out_d = {}

        for node in G:
            s = 0
            for n in adj[node]:
                s += out_d_mg.get(n, 0)
            out_d[node] = out_d.get(node, 0) + s

        nx.set_node_attributes(G, out_d, 'mem')
        return G
