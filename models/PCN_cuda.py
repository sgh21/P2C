from extensions.pointops.functions import pointops as p
from scipy.spatial import KDTree
import torch
"""
用Kd树来加速FPS,寻找全局最远点
"""
def fps_with_kdtree(points, num_points,start_points=None):
    b , N , _ = points.shape
    selected_points = torch.zeros((b,num_points, 3))
    for i in range(b):
        if  start_points is None:
            selected_points[i][0] = points[i][torch.randint(0, N)]
        else:
            selected_points[i][0] = start_points[i]
        distances = torch.full((N,), torch.inf)
        tree = KDTree(selected_points[i,:1, :].cpu().detach().numpy())
        
        for j in range(1, num_points):
            query_result = torch.from_numpy(tree.query(points[i].cpu().detach().numpy())[0]**2)
            distances = torch.minimum(distances, query_result)
            selected_points[i][j] = points[i][torch.argmax(distances)]
            tree = KDTree(selected_points[i,:j+1, :].cpu().detach().numpy())
    
    return selected_points # [b, num_points, 3]
def get_normal(pointcloud,num_points,group_size=100):
    '''
    为输入点云的每个点计算法向量
    '''
    b, n, _ = pointcloud.shape
    child_group_num = 3
    child_group_size = group_size/(child_group_num-1)
    # 对输入点云进行FPS采样和KNN搜索，实现分组
    sample_points = p.fps(pointcloud, num_points) # [b, num_points, 3]
    knn_index = p.knn(sample_points, pointcloud,group_size)[0] # [b, num_points, group_size]
    # print(knn_index.shape)
     # 将 pointcloud 和 knn_index 调整为适合 torch.gather 的形状
    pointcloud = pointcloud.unsqueeze(1).expand(-1, num_points, -1, -1)
    knn_index = knn_index.unsqueeze(-1).expand(-1, -1, -1, 3)

    group_points = torch.gather(pointcloud, 2, knn_index)
    # group_points = pointcloud[knn_index] # [b, num_points, group_size, 3]
    # 对每个分组计算法向量
    child_group_means = torch.zeros((b,num_points,child_group_num,3))
    means = torch.mean(group_points, dim=2) # [b, num_points, 3]
    for i in range(num_points):
        sample_points=fps_with_kdtree(points=group_points[:,i,:,:],num_points=child_group_num,start_points=means[:,i,:]).to(means.device) # [b, child_group_num, 3]
        child_knn_index = p.knn(src=group_points[:,i,:,:],x=sample_points, k=child_group_size)[0] # [b, child_group_num, child_group_size]
        # 将 group_points[:,i,:,:] 和 child_knn_index 调整为适合 torch.gather 的形状
        group_points_i = group_points[:,i,:,:].unsqueeze(1).expand(-1, child_group_num, -1, -1)
        child_knn_index = child_knn_index.unsqueeze(-1).expand(-1, -1, -1, 3)
        child_group_points = torch.gather(group_points_i, 2, child_knn_index) # [b, child_group_num, child_group_size, 3]
        child_group_means[:,i] = torch.mean(child_group_points, dim=2) # [b, child_group_num, 3]
    # for i in range(num_points):
    #    sample_points=fps_with_kdtree(points=group_points[:,i,:,:],num_points=child_group_num,start_points=means[:,i,:]).to(means.device) # [b, child_group_num, 3]
    #    child_knn_index = p.knn(src=group_points[:,i,:,:],x=sample_points, k=child_group_size)[0] # [b, child_group_num, child_group_size]
    #    child_group_points = group_points[:,i][child_knn_index] # [b, child_group_num, child_group_size, 3]
    #    child_group_means[:,i] = torch.mean(child_group_points, dim=2) # [b, child_group_num, 3]
    return group_points,child_group_means #返回的是分组后的点和每个分组的特征点，特征点格式是（b,num_points,child_group_num,3）

def compute_normal(group_means):
    '''
    为每个分组的特征点计算法向量中点和方向
    '''
    b, num_points, child_group_num, _ = group_means.shape
    normals = torch.zeros((b,num_points, 2,3))
    centriod = group_means[:,:,0,:] # [b, num_points, 3]
    direction = (group_means[:,:,1,:]-group_means[:,:,2,:])/torch.linalg.norm(group_means[:,:,1,:]-group_means[:,:,2,:]) # [b, num_points, 3]
    normals[:,:,0,:] = centriod
    normals[:,:,1,:] = direction
    return normals

def compute_distance(points1, points2):
    """
    计算两组点之间的距离
    return distance [N, M]
    """
    points1 = torch.unsqueeze(points1, dim=1)
    points2 = torch.unsqueeze(points2, dim=0)
    return torch.sqrt(torch.sum((points1-points2)**2, dim=-1))

def compute_angles(points1, points2):
    """
    计算两组点之间的夹角
    return angles [N, M]
    """
    points1 = torch.unsqueeze(points1, dim=1)
    points2 = torch.unsqueeze(points2, dim=0)
    # 计算点积
    dot_product = torch.sum(points1 * points2, dim=-1)
    # 计算范数
    norm1 = torch.linalg.norm(points1, dim=-1)
    norm2 = torch.linalg.norm(points2, dim=-1)
    # 计算夹角
    cos_angle = torch.abs(dot_product / (norm1 * norm2))
    angles = torch.acos(torch.clamp(cos_angle, -1, 1))  # 防止由于浮点误差导致的值超出[-1, 1]的范围

    return angles  # 将弧度转换为度

def get_connections_loss(points,m=0.9):
    """
    从点云中找到连接关系，并计算连接损失。
    points [N,2,3] N<=30
    """
    points_location = points[:, 0,:]
    points_direction = points[:, 1,:]
    point_remain=torch.zeros((points.shape[0]-1,2,3))
    i=0
    # 初始化未连接集和连接集
    connect_loss=0 #连接损失
    connect_state=torch.zeros(len(points)) #连接状态
    unconnected_set = list(range(len(points)))# 未连接集的索引
    connected_set = [unconnected_set.pop(torch.randint(0, len(unconnected_set),(1,)))]# 连接集的索引
    endpoints = [connected_set[0], connected_set[0]]# 用于存储连接集的端点的索引
    
    while unconnected_set:
        unconnected_points_location = points_location[unconnected_set]
        end_points_location = points_location[endpoints]
        unconnected_points_direction = points_direction[unconnected_set]
        end_points_direction = points_direction[endpoints]
        # 计算连接损失矩阵 [2，N]
        loss_matrix = m * compute_distance(end_points_location, unconnected_points_location) + (1 - m) * compute_angles(end_points_direction, unconnected_points_direction)  # 计算连接损失
        # 获取最小元素的索引
        index = torch.argmin(loss_matrix)

        # 获取最小元素的行列索引
        row = index // loss_matrix.shape[1]
        col = index % loss_matrix.shape[1]
        # 记录连接点
        point_remain[i] = torch.stack([points_location[unconnected_set[col]], points_location[endpoints[row]]])
        i += 1
        # 更新连接集和未连接集
        connected_set.append(unconnected_set[col])
        connect_state[unconnected_set[col]] += 1
        connect_state[endpoints[row]] += 1
        unconnected_set.pop(col)
        connect_loss += loss_matrix[row][col]
        endpoints = (connect_state == 1).nonzero(as_tuple=True)[0]
    return connect_loss,point_remain