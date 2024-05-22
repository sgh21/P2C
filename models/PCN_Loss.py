import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def draw_lines_between_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point_pair in points:
        point1 = point_pair[0]
        point2 = point_pair[1]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

    plt.show()
def visualize_selected_points(original_points, selected_points,point_pairs=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if point_pairs is not None:
        for point_pair in point_pairs:
            point1 = point_pair[0]
            point2 = point_pair[1]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])
    # 为每个邻居点分配一个颜色
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    original_points = original_points.reshape(-1, 3)
    # print(selected_points.shape)
    selected_points =selected_points.reshape(-1, 3)
    # points1=original_points[knn_index[0]]
    # points2=original_points[knn_index[1]]
    # points3=original_points[knn_index[2]]
    # # 绘制原始点云
    # ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], color='b')
    # ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='r')
    # ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='g')
    # ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2], color='b')
    # 绘制被选中的点
    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], color='r')

    # 设置相同的比例
    def set_axes_equal(ax):
        '''确保x, y, z轴具有相同的比例.'''
        extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        centers = np.mean(extents, axis=1)
        max_size = max(abs(extents[:, 1] - extents[:, 0]))
        r = max_size / 2
        ax.set_xlim([centers[0] - r, centers[0] + r])
        ax.set_ylim([centers[1] - r, centers[1] + r])
        ax.set_zlim([centers[2] - r, centers[2] + r])

    set_axes_equal(ax)
    
    plt.show()
"""
获得points中距离sample_points中每个点最近的k个点的索引
"""
def knn(points,sample_points, k):
    tree = KDTree(points)
    _, indices = tree.query(x=sample_points, k=k)  # k+1 because the point itself is included
    return indices  # include the point itself
"""
用Kd树来加速FPS,寻找全局最远点
"""
def fps_with_kdtree(points, num_points,start_points=None):
    N, _ = points.shape
    selected_points = np.zeros((num_points, 3))
    if  start_points is None:
        selected_points[0] = points[np.random.randint(0, N)]
    else:
        selected_points[0] = start_points
    distances = np.full(N, np.inf)
    tree = KDTree(selected_points[:1, :])
    
    for i in range(1, num_points):
        distances = np.minimum(distances, tree.query(points)[0]**2)
        selected_points[i] = points[np.argmax(distances)]
        tree = KDTree(selected_points[:i+1, :])
    
    return selected_points

def get_normal(pointcloud,num_points,group_size=100):
    '''
    为输入点云的每个点计算法向量
    '''
    child_group_num = 3
    child_group_size = group_size/(child_group_num-1)
    # 对输入点云进行FPS采样和KNN搜索，实现分组
    sample_points = fps_with_kdtree(pointcloud, num_points)
    knn_index = knn(pointcloud,sample_points, group_size)
    # print(knn_index.shape)
    group_points = pointcloud[knn_index]
    # print(group_points.shape)
    # 对每个分组计算法向量
    child_group_means = np.zeros((group_points.shape[0],child_group_num,3))
    for i in range(group_points.shape[0]):
       mean=np.mean(group_points[i], axis=0)
       sample_points=fps_with_kdtree(points=group_points[i], num_points=child_group_num,start_points=mean)
       child_knn_index = knn(group_points[i],sample_points, child_group_size)
       child_group_points = group_points[i][child_knn_index]
       for j in range(child_group_points.shape[0]):
            child_group_means[i][j]=np.mean(child_group_points[j], axis=0)
    return group_points,child_group_means #返回的是分组后的点和每个分组的特征点，特征点格式是（num_points,child_group_num,3）

def compute_normal(group_means):
    '''
    为每个分组的特征点计算法向量中点和方向
    '''
    normals = np.zeros((group_means.shape[0], 2,3))
    for i in range(group_means.shape[0]):
       centriod = group_means[i][0]
       direction = (group_means[i][1]-group_means[i][2])/np.linalg.norm(group_means[i][1] - group_means[i][2])
       normals[i][0] = centriod
       normals[i][1] = direction
    return normals

def compute_distance(points1, points2):
    """
    计算两组点之间的距离
    return distance [N, M]
    """
    points1 = np.expand_dims(points1, axis=1)
    points2 = np.expand_dims(points2, axis=0)
    return np.sqrt(np.sum((points1-points2)**2, axis=-1))

def compute_angles(points1, points2):
    """
    计算两组点之间的夹角
    return angles [N, M]
    """
    points1 = np.expand_dims(points1, axis=1)
    points2 = np.expand_dims(points2, axis=0)
    # 计算点积
    dot_product = np.sum(points1 * points2, axis=-1)
    # 计算范数
    norm1 = np.linalg.norm(points1, axis=-1)
    norm2 = np.linalg.norm(points2, axis=-1)
    # 计算夹角
    cos_angle = np.abs(dot_product / (norm1 * norm2))
    angles = np.arccos(np.clip(cos_angle, -1, 1))  # 防止由于浮点误差导致的值超出[-1, 1]的范围

    return angles  # 将弧度转换为度
def get_connections_loss(points,m=0.9):
    """
    从点云中找到连接关系，并计算连接损失。
    points [N,2,3] N<=30
    """
    points_location = points[:, 0,:]
    points_direction = points[:, 1,:]
    point_remain=np.zeros((points.shape[0]-1,2,3))
    i=0
    # 初始化未连接集和连接集
    connect_loss=0 #连接损失
    connect_state=np.zeros(len(points)) #连接状态
    unconnected_set = list(range(len(points)))# 未连接集的索引
    connected_set = [unconnected_set.pop(np.random.randint(0, len(unconnected_set)))]# 连接集的索引
    endpoints = [connected_set[0], connected_set[0]]# 用于存储连接集的端点的索引
    
    while unconnected_set:
        unconnected_points_location=points_location[unconnected_set]
        end_points_location=points_location[endpoints]
        unconnected_points_direction=points_direction[unconnected_set]
        end_points_direction=points_direction[endpoints]
        # 计算连接损失矩阵 [2，N]
        loss_matrix = m*compute_distance(end_points_location,unconnected_points_location)+(1-m)*compute_angles(end_points_direction,unconnected_points_direction)# 计算连接损失
        # 获取最小元素的索引
        index = np.argmin(loss_matrix)

        # 获取最小元素的行列索引
        row, col = np.unravel_index(index, loss_matrix.shape)
        # 记录连接点
        point_remain[i]=[points_location[unconnected_set[col]],points_location[endpoints[row]]]
        i+=1
        # 更新连接集和未连接集
        connected_set.append(unconnected_set[col])
        connect_state[unconnected_set[col]]+=1
        connect_state[endpoints[row]]+=1
        unconnected_set.pop(col)
        connect_loss+=loss_matrix[row][col]
        endpoints=np.where( connect_state==1 )[0]
    return connect_loss,point_remain




if __name__=="__main__":
    # 读取点云数据
    source_path = 'D:/Documents/SoftwareDoc/isaac_sim/repData0520/pointcloud_0780.npy'
    points=np.load(source_path)
    print(points.shape)
    num_points = 30
    group_points,group_means=get_normal(points,num_points)
    point_normals = compute_normal(group_means)
    loss,point_pair=get_connections_loss(point_normals,m=0.8)
    print(loss)
    # draw_lines_between_points(point_pair)
    # 可视化选择的特征点
    visualize_selected_points( group_points, group_means,point_pairs=point_pair)