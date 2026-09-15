# ============================================================
# 第1部分：数据加载
# ============================================================
# 这一步的目的：从ODPS数据仓库中读取贷款申请数据
# ODPS = 云数据仓库平台，类似于一个超大的数据库
# ============================================================

from odps.df import DataFrame
import multiprocessing
from datetime import datetime

# 连接ODPS数据源（需要在机器学习平台上配置好数据源ID）
odps = mlp.get_odps_instance(data_id="<your-data-source-id>")


def read_data_from_odps(odpstablename):
    """
    从ODPS读取一张表的数据，返回pandas DataFrame

    参数：
        odpstablename: ODPS中的表名（字符串）

    返回：
        df: pandas的DataFrame，可以像Excel表格一样操作
    """
    # 获取表对象
    table = DataFrame(odps.get_table(odpstablename))

    # 用多进程加速读取（自动检测CPU核数）
    n_process = multiprocessing.cpu_count()
    print('使用CPU核数:', n_process)

    # 记录开始时间
    dt1 = datetime.now()
    print('开始读取:', dt1.strftime("%Y-%m-%d %H:%M:%S"))

    # 执行读取（这步可能要几分钟，取决于数据量）
    df = table.to_pandas(n_process=n_process)

    # 记录结束时间，计算耗时
    dt2 = datetime.now()
    duration = dt2 - dt1
    print(f'读取完成，耗时: {duration.seconds/60:.1f}分钟，数据形状: {df.shape}')
    # df.shape 返回 (行数, 列数)，比如 (50000, 300) 表示5万条数据、300个特征

    return df


# 实际使用：读取一张贷款申请宽表（示例表名，替换为你自己的表名）
df = read_data_from_odps('your_table_name_here')

# 查看前3行数据，确认读取正确
df.head(3)
