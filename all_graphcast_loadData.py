# @title Imports
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

## read /root/data/Sedi_dataset.nc to Sedi_dataset
#  Sedi_dataset = xr.open_dataset("/root/data/Sedi_dataset.nc")



# load example_batch from .csv file
print("load example_batch from .csv file")

# Load the data into a DataFrame
df = pd.read_csv("/root/data/SEDI.csv", encoding='latin1')

# Convert the DataFrame to an xarray Dataset
sedi_ds = xr.Dataset.from_dataframe(df)

# 当 interpreted age 为 nan 时，删去该行
sedi_ds = sedi_ds.dropna(dim='index', subset=['interpreted age'])

# 按照 interpreted age 升序排序，并改变其他变量的顺序
sedi_ds = sedi_ds.sortby('interpreted age', ascending=True)


# Rewrite the lon and lat according the resulation of dataset.
print("Rewrite the lon and lat according the resulation of dataset.")
# define 
resolution_Rewrite_lon_lat = 1    #the resolution of longitiude and latitude

def Rewrite_lon_lat(data, resolution):
    '''
    根据 xarray 数据集中的分辨率 重写 lon 和 lat 
    Rewrite the lon and lat according the resulation of dataset.
    data: the original data
    resolution: the resolution of the data
    '''
    condition_number = int(1/resolution)
    data["site latitude"].data = np.round(data["site latitude"].data * condition_number) / condition_number
    data["site longitude"].data = np.round(data["site longitude"].data * condition_number) / condition_number

    return data


# Rewrite the lon and lat according the dataset of xarray.
# 问题：重写了经纬度的分辨率之后，如何处理新出来的经纬度的重复值？
combined = Rewrite_lon_lat(sedi_ds, resolution_Rewrite_lon_lat)


# 使用 groupby 方法根据 lon、lat 和 time 三个变量对数据集进行分组, 并对分组后的数据集求平均
print("使用 groupby 方法根据 lon、lat 和 time 三个变量对数据集进行分组, 并对分组后的数据集求平均")
# Function to process a part of the dataset
sedimentary_list = []
def groupby_and_average(sedi_ds):
    '''
    # 使用 groupby 方法根据 lon、lat 和 time 三个变量对数据集进行分组, 并对分组后的数据集求平均
    '''
    for site_longitude_value, site_longitude in sedi_ds.groupby("site longitude"):
        for site_latitude_value, site_latitude in site_longitude.groupby("site latitude"):
            for interpreted_age_value, sedi in site_latitude.groupby("interpreted age"):
                #sedimentary_dict = sedi.apply(np.mean).to_dict() 
                sedimentary_list.append(sedi.apply(np.mean))
    
    # Add an identifying dimension to each xr.Dataset of sedimentary_list 
    for i, sedi_ds in enumerate(sedimentary_list):
        sedi_ds = sedi_ds.expand_dims({'sample': [i]})

    # Concatenate the datasets
    combined = xr.concat(sedimentary_list, dim='index')


    return combined, site_longitude_value, site_latitude_value, interpreted_age_value


# Divide the dataset into parts
part_number = 9
dim = 'index'  # replace with your actual dimension
dim_size = sedi_ds.dims[dim]
indices = np.linspace(0, dim_size, part_number+1).astype(int)
parts = [sedi_ds.isel({dim: slice(indices[i], indices[i + 1])}) for i in range(part_number)]

# Create a multiprocessing Pool
pool = mp.Pool(mp.cpu_count())

# Process each part of the dataset in parallel with a progress bar
print('Processing Sedi datasets, replacing duplicates with averages ...')
results = []
with tqdm(total=len(parts)) as pbar:
    for result in pool.imap_unordered(groupby_and_average, parts):
        results.append(result)
        pbar.update(1)

# Close the pool
pool.close()

# To combine multiple xarray.Dataset objects
result_list = [result[0] for result in results]
combined = xr.concat(result_list, dim='index')
# 按照 interpreted age 升序排序，并改变其他变量的顺序
combined = combined.sortby('interpreted age', ascending=True)


# Create the new xr.Dataset
# When copy, notice that deep copy and shallow copy.

# define 
resolution = resolution_Rewrite_lon_lat    #the resolution of longitiude and latitude
batch = 0
datetime_temp = np.random.rand(1, len(list(dict.fromkeys(combined['interpreted age'].data))))   # 这里要根据非重复 age 的长度来定义 xarray 的长度
datetime_temp[0, :] = list(dict.fromkeys(combined['interpreted age'].data))

# Create the dimensions
dims = {
    "lon": int(360/resolution),
    "lat": int(181/resolution),
    "level": 13,
    "time": len(list(dict.fromkeys(combined['interpreted age'].data))),
}

# Create the coordinates
coords_creat = {
    "lon": np.linspace(0, 359, int(dims["lon"] - (1/resolution - 1))),
    "lat": np.linspace(-90, 90, int(dims["lat"] - (1/resolution - 1))),
    "level": np.arange(50, 1000, 75),
    "time": datetime_temp[0, :],
    "datetime": (["batch", "time"], datetime_temp),
}


# Create the new dataset
Sedi_dataset = xr.Dataset(coords = coords_creat)

print("Create the new dataset done.")

# load sedi data into the Sedi_dataset



j=0
dims = Sedi_dataset.dims    # Get the dimensions from Sedi_dataset


# remove duplicate values from  <combined['interpreted age'].data>
combined_age_remo_dupli = list(dict.fromkeys(combined['interpreted age'].data))
# 
combined_batch = Sedi_dataset["batch"].data
Sedi_dataset["batch"]

# Add the variables from the combined dataset to the new dataset
for var in tqdm(combined.data_vars, desc="load sedi data"):
    # Skip the variables that are Coordinates.
    if var == "site latitude" or var == "site longitude" or var == "interpreted age":
        continue

    # def  / 是否可以使用广播？
    # create a nan array with the shape of (1,664,181,360) by numpy
    data = np.nan * np.zeros((1, len(combined_age_remo_dupli), dims["lat"], dims["lon"]))   # (banch, time, lat, lon)
    data = data.astype(np.float16)    # Convert the data type to np.float32   有效，这段代码能少一半内存


    # [非常重要]如何测试这段代码？???????????????????????????????????????????????
    for i in range(len(combined["index"])):  
        # 当 age 重复的时候，使用 i-j 来保持时间不变。
        if combined['interpreted age'].data[i-1] == combined['interpreted age'].data[i]:
            j = j + 1    # 如果 age 重复，j 就加 1，i-j 保持时间不变
            # i 指示 age，j 用来固定重复的 age，下面的代码将经纬度上的数据赋值给指定 age 。
            data[batch, i-j, int(combined["site latitude"].values[i]), 
                                int(combined["site longitude"].values[i])] = combined[var].values[i]
        else:
            # 如果 age 不重复，j 不变，i-j 在之前的基础上继续变化
            # i 指示 age，j 用来固定重复的 age，下面的代码将经纬度上的数据赋值给指定 age 。
            data[batch, i-j, int(combined["site latitude"].values[i]), 
                                int(combined["site longitude"].values[i])] = combined[var].values[i]
    j = 0    # 重置 j 的值

    # Create a new DataArray with the same data but new dimensions
    new_dataarray = xr.DataArray(
        data,
        dims=["batch", "time", "lat", "lon"],
        coords={"batch": Sedi_dataset["batch"], "time": combined_age_remo_dupli, "lat": Sedi_dataset["lat"], "lon": Sedi_dataset["lon"]}
    )
    # Add the new DataArray to the new dataset
    Sedi_dataset[var] = new_dataarray
    del data, new_dataarray


Sedi_dataset.astype(np.float32)    # TypeError: Illegal primitive data type, must be one of dict_keys(['S1', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f4', 'f8']), got float16 (variable 'Ag (ppm)', group '/')

# save the Sedi_dataset
path = "/root/autodl-fs/data/SEDI.nc"
Sedi_dataset.to_netcdf(path)
print(f"Save the Sedi_dataset done.  path = {path}")

