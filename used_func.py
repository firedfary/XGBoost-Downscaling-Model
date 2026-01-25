#模态画图——降水——站点数据插值为格点数据
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
from metpy.interpolate import inverse_distance_to_grid
from functools import reduce
import maskout
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl
import cartopy.feature as cfeature
from netCDF4 import Dataset
from cartopy.util import add_cyclic_point
import matplotlib
def gred_site_to_net(input_data):
    df = pd.read_csv('E:\\D1\\f01\\xg6\\result.csv')
    df = df.drop(['time', 'Precip'], axis=1)
    df = df.drop_duplicates('Stn_No')
    df_mode = input_data.to_dataframe().reset_index()

    # 按照'Stn_No'列来合并数据表
    df_final = pd.merge(df, df_mode, how='left', on='Stn_No')

    # 显示结果
    # print(df_final)
    # 删除包含NaN的行
    df_final = df_final.dropna()

    # 显示结果
    # print(df_final)
    min_long = df_final['Long'].min()/100
    max_long = df_final['Long'].max()/100
    min_lat = df_final['Lat'].min()/100
    max_lat = df_final['Lat'].max()/100
    target_lons = np.arange(70, 140, 1)
    target_lats = np.arange(60, 0, -1)

    # 创建目标经纬度的网格
    target_grid_lon, target_grid_lat = np.meshgrid(target_lons, target_lats)

    # 插值
    grid_z = griddata((df_final['Long']/100, df_final['Lat']/100),df_final['leftPattern'], (target_grid_lon, target_grid_lat),method='linear')
    # print(grid_z)
    # 创建插值后的xarray.DataArray
    gridd_site_data = xr.DataArray(grid_z, coords=[('lat', target_lats), ('lon', target_lons)])

    # 显示结果
    print(gridd_site_data)
    return gridd_site_data


def draw_rainfall_map(draw_what):
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
    min_of_draw_what = np.min(draw_what)
    max_of_draw_what = np.max(draw_what)
    lim = np.concatenate((np.linspace(min_of_draw_what,0,4),np.delete(np.linspace(0,max_of_draw_what,4), 0)),axis=0)


    # 地图界限
    with open('./CN-border-La.gmt') as src:
        context = src.read()
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
    fig = plt.figure(figsize=(16,9),facecolor='white',dpi=100)
    myproj = ccrs.PlateCarree(central_longitude=0.0)#ccrs跟绘地图有关#调整图像中心位置
    ax = fig.add_axes([0.10, 0.15, 0.8, 0.75],projection=myproj)#add_axes通过相对位置增加子图 
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', lw=1, color='k',
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
    plt.tick_params(axis='both',labelsize=24)#刻度的字号
    cmap1 = mpl.colors.ListedColormap(['red', 'tomato', 'orange', 'gold', 'lightcyan', 'skyblue', 'royalblue', 'blue'])
    cmap1.set_over('darkred')
    cmap1.set_under('navy')    #22就是最后一年，2016年
    cf=plt.contourf(lon,lat,draw_what,levels=lim,cmap =cmap1,extend='both',zorder=0,transform=ccrs.PlateCarree(),norm=MidpointNormalize(midpoint=0))#画出等高线


    colors =mpl.colors.ListedColormap(['black'])
    # plt.title('precipitation anomaly', fontsize=14)
    #plt.legend((),('A1',),loc='upper left',fontsize=20,ncol=2)
    # plt.title('D2',fontsize=60,loc='left',y=0.85,x=0.02)#标题和标题大小
    cbar=plt.colorbar(cf,cax=fig.add_axes([0.27, 0.04, 0.45, 0.05]),orientation='horizontal',cmap=cmap1)
    cbar.ax.tick_params(labelsize=22)#条子的字号

    clip=maskout.shp2clip(cf,ax, r'./data/province/china0.shp')
    sub_ax = fig.add_axes([0.662, 0.155, 0.14, 0.155],projection=myproj)#小南海的位置
    sub_ax.add_feature(cfeature.LAND.with_scale('110m'))
    for line in borders:
        sub_ax.plot(line[0::2], line[1::2], '-', lw=1, color='k',
                    transform=ccrs.Geodetic())
    # Set figure extent设置图范围
    sub_ax.set_extent([105, 125, 0, 25]) 
    plt.subplots_adjust()
    plt.show()