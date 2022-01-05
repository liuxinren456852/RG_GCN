import open3d as o3d
import numpy as np
import os


def _m(file):
    fin = open(file)
    a = file.split('.')
    b = a[0] + '_m.' + a[1]
    fout = open(b, 'w')
    lines = fin.readlines()
    for line in lines:
        fout.write(line[2:])
    fout.close()
    fin.close()


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


def tras_to_ply(f_in, f_out):
    a = f_in.split('.')[0].split('\\')[-1]
    output_file = f_out + a + '.ply'
    points = np.loadtxt(f_in)
    b = np.float32(points)
    vertices = b[:, 0:3]
    colors = b[:, 3:6]
    create_output(vertices, colors, output_file)


if __name__ == '__main__':
    file = 'F:\\Toronto3Dxd04可视化\\L004_dgcnn.txt'
    _m(file)
    print('转换txt完成')
    tras_to_ply('F:\\Toronto3Dxd04可视化\\L004_dgcnn_m.txt',
                'F:\\')
    print('转换ply完成')

    # pcd = o3d.io.read_point_cloud("F:\\L002_pred_m.ply")
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=1, front=[0.1, 0.1, 1], lookat=[600, 600, 200],
    #                                   up=[0, 0, 1])
