#模态画图——降水——站点数据插值为格点数据
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
from metpy.interpolate import inverse_distance_to_grid
from functools import reduce
import maskout
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional
import os
import torch
import matplotlib as mpl

def grid_to_station_interp(grid_data: np.ndarray, time_list, position_csv_path='E:\D1\diffusion\my_models\my_model_data\position.csv', var_name='Precip'):
    """
    grid_data: shape [n, 300, 350]
    time_list: 长度为 n 的时间字符串列表
    position_csv_path: 站点经纬度csv路径
    var_name: 变量名（如'Precip'、'anoma'等），用于输出DataFrame的列名
    返回: DataFrame，包含['Stn_No', 'time', 'Lat', 'Long', var_name]
    """
    if grid_data.shape[0] != len(time_list):
        raise ValueError("grid_data的第0维与time_list长度不一致！")
    # 读取站点信息
    pos_df = pd.read_csv(position_csv_path)
    # 假设列名为 Stn_No, Lat, Long
    lats = pos_df['Lat'].values
    lons = pos_df['Long'].values
    stn_nos = pos_df['Stn_No'].values

    # 构造网格经纬度
    grid_lats = np.linspace(60, 0, 120)
    grid_lons = np.linspace(70, 140, 140)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lons, grid_lats)

    records = []
    for i, t in enumerate(time_list):
        grid = grid_data[i]
        # 插值
        interp_values = griddata(
            (grid_lat2d.flatten(), grid_lon2d.flatten()),
            grid.flatten(),
            (lats, lons),
            method='linear'
        )
        for stn, lat, lon, val in zip(stn_nos, lats, lons, interp_values):
            records.append([stn, t, lat, lon, val])
    df = pd.DataFrame(records, columns=['Stn_No', 'time', 'Lat', 'Long', var_name])
    return df



def gred_month_site_to_net(df:pd.DataFrame, to_xr:bool, gred_var:str='Precip'):
    """
    必须包含站点, 经纬度, 年月, Precip

    :param df: 输入数据必须是csv格式
    :param to_xr: 选择输出格式
    """
    if df.isnull().values.any():
        print('there are NaNs in the dataframe')
    unique_year = df['Year'].unique()
    grid_data_all = []
    for year in unique_year:
        data_in_current_year = df[df.Year == year]
        unique_month = data_in_current_year['Month'].unique()
        for month in unique_month:
            current_data= data_in_current_year[data_in_current_year.Month == month]
            target_lons = np.arange(70, 140, 0.1)
            target_lats = np.arange(60, 0, -0.1)
            target_grid_lon, target_grid_lat = np.meshgrid(target_lons, target_lats)
            grid_data = griddata((current_data['Long']/100, current_data['Lat']/100),current_data[gred_var], (target_grid_lon, target_grid_lat),method='linear')
            assert len(grid_data[np.isnan(grid_data)]) != len(grid_data), 'all data is NaN!!!'
            if to_xr:
                date_str = str(year)+str(int(month)).zfill(2)+'01'
                time = pd.date_range(date_str, periods=1)
                arrayed_data = xr.DataArray(grid_data, coords=[('lat', target_lats), ('lon', target_lons)])
                arrayed_data = arrayed_data.expand_dims(time=time)
                grid_data_all.append(arrayed_data)
                
            else:
                grid_data_all.append(grid_data)
    if to_xr:
        xdgrid_data = xr.concat(grid_data_all, dim='time')
        return xdgrid_data
    else:
        npgrid_data = np.stack(grid_data_all)
        return npgrid_data


def gred_time_site_to_net(df: pd.DataFrame, to_xr: bool, gred_var: str = 'Precip', position_csv_path=r'E:\HydroSynth\utils\position.csv'):
    """
    必须包含站点, 经纬度, time, Precip

    :param df: 输入数据必须是csv格式
    :param to_xr: 选择输出格式
    """
    # 检查经纬度是否完整（有缺失或不存在）
    if ('Long' not in df.columns) or ('Lat' not in df.columns) or df['Long'].isnull().any() or df['Lat'].isnull().any():
        print('检测到输入df的经纬度信息不完整，自动从position_csv_path读取并合并...')
        pos_df = pd.read_csv(position_csv_path)
        # 只保留必要列
        pos_df = pos_df[['Stn_No', 'Lat', 'Long']]
        # 合并
        df = pd.merge(df, pos_df, on='Stn_No', how='left')
    if df.isnull().values.any():
        print('there are NaNs in the dataframe')
    unique_time = df['time'].unique()
    grid_data_all = []
    for time in unique_time:
        current_data = df[df['time'] == time]
        target_lons = np.arange(70, 140, 0.5)
        target_lats = np.arange(60, 0, -0.5)
        target_grid_lon, target_grid_lat = np.meshgrid(target_lons, target_lats)
        grid_data = griddata((current_data['Long'], current_data['Lat']), current_data[gred_var], (target_grid_lon, target_grid_lat), method='linear')
        assert len(grid_data[np.isnan(grid_data)]) != len(grid_data), 'all data is NaN!!!'
        if to_xr:
            arrayed_data = xr.DataArray(grid_data, coords=[('lat', target_lats), ('lon', target_lons)])
            arrayed_data = arrayed_data.expand_dims(time=time)
            grid_data_all.append(arrayed_data)
        else:
            grid_data_all.append(grid_data)
    if to_xr:
        xdgrid_data = xr.concat(grid_data_all, dim='time')
        return xdgrid_data
    else:
        npgrid_data = np.stack(grid_data_all)
        return npgrid_data






def clip_data(size_per_image:int, data_lenth:int) -> list:
    """
    返回一个列表, x表示在第x个像素分割
    """
    coord_list = []
    x = 0
    max_j = data_lenth//size_per_image#百分号才是取余
    while True:
        bigger_all_data = (max_j*size_per_image)-data_lenth
        if bigger_all_data >= size_per_image/2:
            break
        max_j += 1
    over_lap = bigger_all_data // (max_j-1)
    for j in range(max_j-1):
        x = (size_per_image-over_lap)*j
        print(x)
        coord_list.append(x)
    last_end = x+64
    last_x = last_end - (size_per_image-(data_lenth-last_end))
    coord_list.append(last_x)
    return coord_list







def draw_rainfall_map(draw_what:np.ndarray, picture_name:str=None, min:float=None, max:float=None, save_path:str=None, color_mode:str="blue"):
    np.set_printoptions(suppress=True)

    class MidpointNormalize(matplotlib.colors.Normalize): 
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))



    lon = np.linspace(70, 140, np.shape(draw_what)[1])
    lat = np.linspace(60, 0, np.shape(draw_what)[0])
    # # 画下面的彩色条子
    min_of_draw_what = np.nanmin(draw_what)
    max_of_draw_what = np.nanmax(draw_what)
    if min is not None:
        min_of_draw_what = min
    if max is not None:
        max_of_draw_what = max

    # 修正色阶分布逻辑：根据 min/max 的符号关系分别构造 lim，并动态设置色阶中点
    if min_of_draw_what >= 0 and max_of_draw_what > 0:
        # 全正区间
        lim = np.linspace(min_of_draw_what, max_of_draw_what, 7)
        midpoint = (min_of_draw_what + max_of_draw_what) / 2
    elif max_of_draw_what <= 0 and min_of_draw_what < 0:
        # 全负区间
        lim = np.linspace(min_of_draw_what, max_of_draw_what, 7)
        midpoint = (min_of_draw_what + max_of_draw_what) / 2
    else:
        # 跨零区间（原逻辑）
        lim = np.concatenate((np.linspace(min_of_draw_what,0,4),np.delete(np.linspace(0,max_of_draw_what,4), 0)),axis=0)
        midpoint = 0


    # 地图界限
    with open(r"E:\HydroSynth\utils\CN-border-La.gmt") as src:
        context = src.read()
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    fig = plt.figure(figsize=(6,6),facecolor='white')
    myproj = ccrs.PlateCarree(central_longitude=0.0)#ccrs跟绘地图有关#调整图像中心位置
    ax = fig.add_axes([0.10, 0.22, 0.8, 0.75],projection=myproj)#add_axes通过相对位置增加子图 
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', lw=0.5, color='k',
                transform=ccrs.Geodetic())
        
    # Plot gridlines绘制网格线
    #ax.gridlines(linestyle='--')

    # Set figure extent设置图范围
    ax.set_extent([70, 140, 10, 60])#摆正图像
    # ax.set_xticks([70,80,90,100,110,120,130,140],crs=ccrs.PlateCarree())
    # ax.set_yticks([10,20,30,40,50],crs=ccrs.PlateCarree())
    ax.set_xticks(range(70,141,10),crs=ccrs.PlateCarree())
    ax.set_yticks(range(10,61,10),crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()#定义一种坐标刻度样式为维度刻度	
    ax.xaxis.set_major_formatter(lon_formatter)#改变X轴主刻度为经度样式
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.tick_params(axis='both',labelsize=14)#刻度的字号
    # 定义原始色带（蓝色偏大）
    base_colors = ['#E11300', '#FF3100', '#FF9F00', '#FFBF3B', '#FFE877','#B3F0FA', '#95D2FA', '#77B8FA', '#4FA4F5', '#3B95F5']
    if color_mode == "red":
        # 反转色带，使红色表示偏大
        base_colors = base_colors[::-1]
    cmap1 = mpl.colors.ListedColormap(base_colors)
    cmap1.set_over('darkred')
    cmap1.set_under('navy')
    cf=plt.contourf(lon,lat,draw_what,levels=lim,cmap =cmap1,extend='both',zorder=0,transform=ccrs.PlateCarree(),norm=MidpointNormalize(midpoint=midpoint))#画出等高线


    colors =matplotlib.colors.ListedColormap(['black'])
    # plt.title('precipitation anomaly', fontsize=14)
    #plt.legend((),('A1',),loc='upper left',fontsize=20,ncol=2)
    # plt.title('D2',fontsize=60,loc='left',y=0.85,x=0.02)#标题和标题大小
    formatter = ticker.FormatStrFormatter('%.1f')
    cbar=plt.colorbar(cf,cax=fig.add_axes([0.13, 0.20, 0.75, 0.03]),format=formatter, orientation='horizontal',cmap=cmap1)
    cbar.ax.tick_params(labelsize=14)#条子的字号

    clip=maskout.shp2clip(cf,ax, r"E:\HydroSynth\utils\china0.shp")
    sub_ax = fig.add_axes([0.76, 0.29, 0.14, 0.155],projection=myproj)#小南海的位置
    sub_ax.add_feature(cfeature.LAND.with_scale('110m'))
    for line in borders:
        sub_ax.plot(line[0::2], line[1::2], '-', lw=0.5, color='k',
                    transform=ccrs.Geodetic())
    # Set figure extent设置图范围
    sub_ax.set_extent([105, 125, 0, 25]) 
    plt.subplots_adjust()
    if picture_name is not None:
        plt.text(.01, .99, picture_name, ha='left', va='top', transform=ax.transAxes, fontsize=16)
    if save_path is not None:
        plt.savefig(save_path, format='svg')
    plt.show()















def read_nc_to_npy(start: int, end: int, data_path: str = "D:\MODESv21_ecmwf_seas51") -> list:
    start_year = int(str(start)[0:4])
    end_year = int(str(end)[0:4])
    start_month = int(str(start)[4:6])
    end_month = int(str(end)[4:6])

    file_name_list = []

    if start_year == end_year:
        for i in range(start_month, end_month + 1):
            file_name = rf"{data_path}\MODESv21_ecmwf_seas51_{start_year}{str(i).zfill(2)}_monthly_em.nc"
            file_name_list.append(file_name)
    else:
        # 起始年
        for i in range(start_month, 13):
            file_name = rf"{data_path}\MODESv21_ecmwf_seas51_{start_year}{str(i).zfill(2)}_monthly_em.nc"
            file_name_list.append(file_name)
        # 跨年
        for year in range(start_year + 1, end_year):
            for month in range(1, 13):
                file_name = rf"{data_path}\MODESv21_ecmwf_seas51_{year}{str(month).zfill(2)}_monthly_em.nc"
                file_name_list.append(file_name)
        # 结束年
        for i in range(1, end_month + 1):
            file_name = rf"{data_path}\MODESv21_ecmwf_seas51_{end_year}{str(i).zfill(2)}_monthly_em.nc"
            file_name_list.append(file_name)

    print('total month', len(file_name_list))
    return file_name_list




def save_image(*tensors: torch.Tensor, 
               names: Optional[List[str]] = None,
               ncols: int = 2,
               path: Optional[str] = None,
               show: bool = False,
               figsize: tuple = (10, 9),
               filename: str = 'image.svg'):
    num_images = len(tensors)
    
    # 验证命名参数
    if names is None:
        names = [f'Image {i+1}' for i in range(num_images)]
    elif len(names) != num_images:
        raise ValueError("names参数长度必须与输入张量数量一致")

    # 计算行列布局
    if num_images == 1:
        ncols = 1
        figsize = (5, 5)
    nrows = (num_images + ncols - 1) // ncols
    
    # 创建子图并统一坐标轴格式
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.array(axs).flatten()  # 强制转换为一维数组

    # 遍历处理每个张量
    for idx, (tensor, name) in enumerate(zip(tensors, names)):
        ax = axs[idx]
        
        # 处理张量维度
        array = tensor.cpu().detach().numpy()
        if array.ndim > 2:
            array = array[(0,) * (array.ndim - 2)]  # 自动降维
        
        # 绘制图像
        im = ax.imshow(array)
        ax.set_title(name, fontsize=10)
        ax.axis('on')
        
        # 添加格式化颜色条
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(
            ticker.FormatStrFormatter('%.1f'))

    # 隐藏空白子图
    for j in range(num_images, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(pad=2.0)

    # 处理输出
    if show:
        plt.show()
    if path is not None:
        full_path = os.path.join(path, filename)
        plt.savefig(full_path, format="svg", bbox_inches='tight')
    
    plt.close()






def symmetric_max_normalize(x: torch.Tensor, save_dir: str = None) -> torch.Tensor:
    """
    对称最大绝对值归一化（分通道处理，忽略NaN）
    Args:
        x: 输入数据 [batch, channels, h, w]
        save_dir: 参数保存目录（可选）
    Returns:
        归一化后的数据 [batch, channels, h, w]
    """
    # 备份原始数据（避免修改原始Tensor）
    x_nan_handled = x.clone()
    
    # 计算每个通道的有效数据极值（忽略NaN）
    # 1. 替换NaN为极值，确保amin/amax计算时忽略NaN
    x_for_min = torch.where(
        torch.isnan(x_nan_handled),
        torch.tensor(float('inf'), device=x.device),
        x_nan_handled
    )
    channel_mins = torch.amin(x_for_min, dim=(0, 2, 3))  # [C]
    
    x_for_max = torch.where(
        torch.isnan(x_nan_handled),
        torch.tensor(float('-inf'), device=x.device),
        x_nan_handled
    )
    channel_maxs = torch.amax(x_for_max, dim=(0, 2, 3))  # [C]
    
    # 2. 处理全为NaN的通道（将缩放因子设为1）
    all_nan_mask = (channel_mins == float('inf')) & (channel_maxs == float('-inf'))
    valid_max_abs = torch.maximum(torch.abs(channel_mins), channel_maxs)
    max_abs_values = torch.where(
        all_nan_mask,
        torch.ones_like(valid_max_abs),
        valid_max_abs
    )
    
    # 3. 处理缩放因子为0的情况
    max_abs_values = torch.where(
        max_abs_values == 0,
        torch.ones_like(max_abs_values),
        max_abs_values
    )
    
    # 保存参数（如果需要）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        params = {
            "max_abs_values": max_abs_values.cpu(),
            "original_dtype": x.dtype,
            "all_nan_mask": all_nan_mask.cpu()  # 记录全NaN通道
        }
        torch.save(params, os.path.join(save_dir, "normalization_params.pt"))
    
    # 归一化（保留NaN位置不变）
    scales = max_abs_values.view(1, -1, 1, 1)  # [1, C, 1, 1]
    return x_nan_handled / scales.to(x.device)




def denormalize(normalized_x: torch.Tensor, save_dir: str) -> torch.Tensor:
    """
    从归一化数据还原原始数据（自动处理全NaN通道）
    Args:
        normalized_x: 归一化后的数据 [batch, channels, h, w]
        save_dir: 参数保存目录
    Returns:
        原始数据（近似）
    """
    # 加载参数
    params_path = os.path.join(save_dir, "normalization_params.pt")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found in {save_dir}")
    
    params = torch.load(params_path, map_location=normalized_x.device)
    max_abs_values = params["max_abs_values"].to(normalized_x.device)
    all_nan_mask = params["all_nan_mask"].to(normalized_x.device)
    
    # 还原数据（全NaN通道保持归一化值不变）
    scales = max_abs_values.view(1, -1, 1, 1)  # [1, C, 1, 1]
    restored = normalized_x * scales
    
    # 恢复全NaN通道的原始NaN（若需要）
    # 此处假设全NaN通道在归一化后仍为NaN，无需额外操作
    
    return restored.to(params["original_dtype"])





def plot_anomaly_distribution(data, start=-10, stop=10, bins=20, precentage=True):
    """
    绘制异常值分布直方图
    :param data: 输入数据 (result['anoma'])
    :param bins: 区间数量，默认为10
    """
    plt.figure(figsize=(10, 6))
    # 将数据展平为 1 维
    flat_data = data.flatten()
    # 指定 bin 范围为 -10 到 10
    bin_range = np.linspace(start, stop, bins+1)
    n, bins, patches = plt.hist(flat_data, bins=bin_range, edgecolor='black')
    
    # 计算总数
    total = len(flat_data)
    # 在每个柱子上标注百分比
    if precentage:
        for i in range(len(patches)):
            percentage = n[i] / total * 100
            plt.text(patches[i].get_x() + patches[i].get_width() / 2, 
                     patches[i].get_height(), 
                     f'{percentage:.1f}%',  # 保留一位小数
                     ha='center', va='bottom')
    plt.title('Anomaly Value Distribution ({} to {})'.format(start, stop))
    plt.xlabel('Anomaly Value Range')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()





def check_nan_status(tensor):
    # 支持 numpy.ndarray 和 torch.Tensor
    if isinstance(tensor, np.ndarray):
        nan_count = np.isnan(tensor).sum()
        total_elements = tensor.size
        tensor_shape = tensor.shape
    elif isinstance(tensor, torch.Tensor):
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        tensor_shape = tuple(tensor.shape)
    else:
        raise TypeError("只支持输入 torch.Tensor 或 numpy.ndarray")
    nan_rate = nan_count / total_elements * 100
    print(f"张量形状: {tensor_shape}")

    if nan_count == total_elements:
        print("全部是NaN")
    elif nan_count > 0:
        print("有{}个NaN，占总数的{:.2f}%".format(nan_count, nan_rate))
    else:
        print("没有NaN")



def cal_acc(observed:torch.Tensor, predicted:torch.Tensor) -> torch.Tensor:
    assert observed.shape == predicted.shape, "Observed and predicted tensors must have the same shape."
    assert observed.ndim < 4, "Observed tensor must smaller or eq 3 dimensions (time, height, width)."
    if observed.ndim == 2:
        observed = observed.unsqueeze(0)
        predicted = predicted.unsqueeze(0)
    observed_mean = torch.nanmean(observed, dim=(1, 2), keepdim=True)
    predicted_mean = torch.nanmean(predicted, dim=(1, 2), keepdim=True)
    A = observed - observed_mean  # Subtract mean
    B = predicted - predicted_mean  # Subtract mean
    C1 = torch.nansum(A * A, dim=(1, 2))  # No mul, use *
    C2 = torch.nansum(B * B, dim=(1, 2))
    C = torch.sqrt(C1 * C2 + 1e-8)  # Add epsilon
    num = torch.nansum(A * B, dim=(1, 2))
    ACC = num / C
    # Also add optional: clamp to [-1,1] if needed
    return ACC.clamp(-1, 1)