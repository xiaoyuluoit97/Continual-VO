import pandas as pd
import os
import numpy as np
folder_path = 'log/act_naive_pth/csvfile'  # 文件夹路径，请替换成你的文件夹路径
output_file_path = 'res50_vit_overall/res18_act'
# 使用 os.listdir() 列出文件夹中的所有文件
file_list = os.listdir(folder_path)
overall_forward_files = [file for file in file_list if file.startswith('overall_forward')]
sorted_files = sorted([file for file in overall_forward_files if not file.split('_')[-1].split('.')[0].startswith('-')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

variance_x=[]
variance_yaw=[]
variance_z=[]
mean_x=[]
mean_yaw=[]
mean_z=[]
range_upper_x =[]
range_upper_yaw =[]
range_upper_z =[]
range_lower_x =[]
range_lower_yaw =[]
range_lower_z =[]

for file_name in sorted_files:
    # 在这里可以读取文件
    # 比如读取 CSV 文件
    data = pd.read_csv(os.path.join(folder_path,file_name))
    variance_x.append(np.var(data.iloc[:,3]))
    mean_x.append(np.mean(data.iloc[:,3]))
    range_upper_x.append(np.mean(data.iloc[:,3]) + 1.96 * np.sqrt(np.var(data.iloc[:,3])/len(data.iloc[:,3])))
    range_lower_x.append(np.mean(data.iloc[:, 3]) - 1.96 * np.sqrt(np.var(data.iloc[:, 3])/len(data.iloc[:,3])))

    variance_yaw.append(np.var(data.iloc[:,4]))
    mean_yaw.append(np.mean(data.iloc[:,4]))
    range_upper_yaw.append(np.mean(data.iloc[:,4]) + 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))
    range_lower_yaw.append(np.mean(data.iloc[:,4]) - 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))

    variance_z.append(np.var(data.iloc[:,5]))
    mean_z.append(np.mean(data.iloc[:,5]))
    range_upper_z.append(np.mean(data.iloc[:,5]) + 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))
    range_lower_z.append(np.mean(data.iloc[:,5]) - 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))

df = pd.DataFrame({'variance_x': variance_x, 'variance_yaw': variance_yaw, 'variance_z': variance_z,'mean_x': mean_x,'up_x': range_upper_x,'low_x': range_lower_x,'mean_yaw': mean_yaw,'up_yaw': range_upper_yaw,'low_yaw': range_lower_yaw,'mean_z': mean_z,'up_z': range_upper_z,'low_z': range_lower_z})

output_file = 'overall_forward.csv'
df.to_csv(os.path.join(output_file_path,output_file), index=False)

overall_left_files = [file for file in file_list if file.startswith('overall_left')]

sorted_files = sorted([file for file in overall_left_files if not file.split('_')[-1].split('.')[0].startswith('-')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

variance_x=[]
variance_yaw=[]
variance_z=[]
mean_x=[]
mean_yaw=[]
mean_z=[]
range_upper_x =[]
range_upper_yaw =[]
range_upper_z =[]
range_lower_x =[]
range_lower_yaw =[]
range_lower_z =[]

for file_name in sorted_files:
    # 在这里可以读取文件
    # 比如读取 CSV 文件
    data = pd.read_csv(os.path.join(folder_path,file_name))
    variance_x.append(np.var(data.iloc[:,3]))
    mean_x.append(np.mean(data.iloc[:,3]))
    range_upper_x.append(np.mean(data.iloc[:,3]) + 1.96 * np.sqrt(np.var(data.iloc[:,3])/len(data.iloc[:,3])))
    range_lower_x.append(np.mean(data.iloc[:, 3]) - 1.96 * np.sqrt(np.var(data.iloc[:, 3])/len(data.iloc[:,3])))

    variance_yaw.append(np.var(data.iloc[:,4]))
    mean_yaw.append(np.mean(data.iloc[:,4]))
    range_upper_yaw.append(np.mean(data.iloc[:,4]) + 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))
    range_lower_yaw.append(np.mean(data.iloc[:,4]) - 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))

    variance_z.append(np.var(data.iloc[:,5]))
    mean_z.append(np.mean(data.iloc[:,5]))
    range_upper_z.append(np.mean(data.iloc[:,5]) + 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))
    range_lower_z.append(np.mean(data.iloc[:,5]) - 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))

df = pd.DataFrame({'variance_x': variance_x, 'variance_yaw': variance_yaw, 'variance_z': variance_z,'mean_x': mean_x,'up_x': range_upper_x,'low_x': range_lower_x,'mean_yaw': mean_yaw,'up_yaw': range_upper_yaw,'low_yaw': range_lower_yaw,'mean_z': mean_z,'up_z': range_upper_z,'low_z': range_lower_z})

output_file = 'overall_left.csv'
df.to_csv(os.path.join(output_file_path,output_file), index=False)

df=None
output_file = None
sorted_files = None
overall_right_files = None
overall_right_files = [file for file in file_list if file.startswith('overall_right')]
sorted_files = sorted([file for file in overall_right_files if not file.split('_')[-1].split('.')[0].startswith('-')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

variance_x=[]
variance_yaw=[]
variance_z=[]
mean_x=[]
mean_yaw=[]
mean_z=[]
range_upper_x =[]
range_upper_yaw =[]
range_upper_z =[]
range_lower_x =[]
range_lower_yaw =[]
range_lower_z =[]

for file_name in sorted_files:
    # 在这里可以读取文件
    # 比如读取 CSV 文件
    data = pd.read_csv(os.path.join(folder_path,file_name))
    variance_x.append(np.var(data.iloc[:,3]))
    mean_x.append(np.mean(data.iloc[:,3]))
    range_upper_x.append(np.mean(data.iloc[:,3]) + 1.96 * np.sqrt(np.var(data.iloc[:,3])/len(data.iloc[:,3])))
    range_lower_x.append(np.mean(data.iloc[:, 3]) - 1.96 * np.sqrt(np.var(data.iloc[:, 3])/len(data.iloc[:,3])))

    variance_yaw.append(np.var(data.iloc[:,4]))
    mean_yaw.append(np.mean(data.iloc[:,4]))
    range_upper_yaw.append(np.mean(data.iloc[:,4]) + 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))
    range_lower_yaw.append(np.mean(data.iloc[:,4]) - 1.96 * np.sqrt(np.var(data.iloc[:,4])/len(data.iloc[:,4])))

    variance_z.append(np.var(data.iloc[:,5]))
    mean_z.append(np.mean(data.iloc[:,5]))
    range_upper_z.append(np.mean(data.iloc[:,5]) + 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))
    range_lower_z.append(np.mean(data.iloc[:,5]) - 1.96 * np.sqrt(np.var(data.iloc[:,5])/len(data.iloc[:,5])))

df = pd.DataFrame({'variance_x': variance_x, 'variance_yaw': variance_yaw, 'variance_z': variance_z,'mean_x': mean_x,'up_x': range_upper_x,'low_x': range_lower_x,'mean_yaw': mean_yaw,'up_yaw': range_upper_yaw,'low_yaw': range_lower_yaw,'mean_z': mean_z,'up_z': range_upper_z,'low_z': range_lower_z})

output_file = 'overall_right.csv'
df.to_csv(os.path.join(output_file_path,output_file), index=False)