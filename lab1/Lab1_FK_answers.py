import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """

    with open(bvh_file_path, 'r') as f:
        file_content = f.read().splitlines()

    joint_stack = []
    joint_list = []
    name = ""
    index = -1
    for line in file_content:
        if line.strip().startswith("ROOT") or line.strip().startswith("JOINT") or line.strip().startswith("End Site"):
            joint = {
                "name": "",
                "index": None,
                "parent_index": index,
                "offset": []
            }
            if line.strip().startswith("ROOT") or line.strip().startswith("JOINT"):
                name = line.split()[1]
                joint["name"] = name
            else:
                joint["name"] = name + '_end'
        if line.strip().startswith("{"):
            index += 1
            joint["index"] = index
            joint_stack.append(joint)
        if line.strip().startswith("OFFSET"):
            offset = line.split()[1:]
            joint_stack[-1]["offset"] = [float(x) for x in offset]
        if line.strip().startswith("}"):
            joint_tmp = joint_stack.pop()
            if len(joint_stack) > 0:
                joint_tmp["parent_index"] = joint_stack[-1]["index"]
            joint_list.append(joint_tmp)

    sorted_joint_list = sorted(joint_list, key=lambda item: item["index"])
    
    joint_name = [j["name"] for j in sorted_joint_list]
    joint_parent = [j["parent_index"] for j in sorted_joint_list]
    joint_offset = np.array([j["offset"] for j in sorted_joint_list])
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    
    motion_frame_data = motion_data[frame_id]

    joint_num = len(joint_name)
    joint_positions = []
    joint_orientations = []
    
    joint_positions.append(motion_frame_data[0: 3])
    joint_orientations.append(R.from_euler('XYZ', motion_frame_data[3: 6], degrees=True).as_quat())
    
    joint_count = 1
    for i in range(1, joint_num):
        parent = joint_parent[i]
        rotate = None
        if joint_name[i].endswith('_end'):
            rotate = R.from_euler('XYZ', [0., 0., 0.], degrees=True)
        else:
            rotate = R.from_euler('XYZ', motion_frame_data[3 + joint_count * 3: 6 + joint_count * 3], degrees=True)
            joint_count += 1

        parent_orientation = R.from_quat(joint_orientations[parent])
        joint_orientations.append((parent_orientation * rotate).as_quat())
        joint_positions.append(joint_positions[parent] + np.dot(R.from_quat(joint_orientations[parent]).as_matrix(), joint_offset[i]))

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
