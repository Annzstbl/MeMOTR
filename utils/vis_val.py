


import sys
import os
import pandas as pd
import glob 
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    # val_root_path = sys.argv[1]
    val_root_path = '/data4/litianhao/hsmot/memotr/spectralemb/06_v2_SpectralDecoderRefine_2gpu'
    fig_path = os.path.join(val_root_path, 'fig')
    os.makedirs(fig_path, exist_ok=True)
    val_folder_list = glob.glob(os.path.join(val_root_path, "epoch*"))
    val_folder_list.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1]))

    epoch_list = [int(val_path.split("/")[-1].split("_")[-1]) for val_path in val_folder_list]
    val_matrix = []

    for val_path in val_folder_list:
        val_file = os.path.join(val_root_path, val_path, 'test', 'eval', 'all_cls_summary.csv')
        val_matrix.append(pd.read_csv(val_file))

    # 每个val_matrix的列坐标是cls+HOTA等组成，
    # 每个行第一列是cls的内容，包括：car bike pedestrian van truck bus tricycle awning-bike cls_comb_cls_av, cls_comb_det_av.
    # 现在需要将每一种类的HOTA DetA AssA MOTA IDF1指标提取出来，以EPOCH为横坐标，构建折线图


    HOTA_matrix = defaultdict(list)
    DetA_matrix = defaultdict(list)
    AssA_matrix = defaultdict(list)
    MOTA_matrix = defaultdict(list)
    IDF1_matrix = defaultdict(list)

    # 读取每一行的第一列
    for val_matrix, epoch in zip(val_matrix, epoch_list):
        for row in val_matrix.iterrows():
            row_data = row[1]
            cls = row_data.cls# 0是序号
            if cls not in ['car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle', 'awning-bike', 'cls_comb_cls_av', 'cls_comb_det_av']:
                continue
            HOTA_matrix[cls].append(row_data.HOTA)
            DetA_matrix[cls].append(row_data.DetA)
            AssA_matrix[cls].append(row_data.AssA)
            MOTA_matrix[cls].append(row_data.MOTA)
            IDF1_matrix[cls].append(row_data.IDF1)
    
    # 把每一种类的所有指标画在一张图,并保存
    for cls in HOTA_matrix.keys():
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_list, HOTA_matrix[cls], label='HOTA')
        #把HOTA的数值写在折线图上
        for i, h in enumerate(HOTA_matrix[cls]):
            plt.text(epoch_list[i], h, f'{h:.2f}', ha='center', va='bottom')
        plt.plot(epoch_list, DetA_matrix[cls], label='DetA')
        plt.plot(epoch_list, AssA_matrix[cls], label='AssA')
        plt.plot(epoch_list, MOTA_matrix[cls], label='MOTA')
        plt.plot(epoch_list, IDF1_matrix[cls], label='IDF1')
        plt.legend(loc='lower right')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title(f'{cls} Metrics')
        # x坐标间隔设置为1
        # y坐标固定从0到100
        plt.xticks(np.arange(0, len(epoch_list), 1))
        plt.yticks(np.arange(0, 100, 10))
        plt.savefig(os.path.join(fig_path, f'{cls}.png'))

