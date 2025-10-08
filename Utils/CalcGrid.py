
from pyproj import Transformer
import math

class GridMapperWebMercator:
    # lon1, lat1, lon2, lat2
    def __init__(self, lon_min, lat_min, lon_max, lat_max, grid_size_m):
        """_summary_
        Web Mercator 投影，就类似平常看到的地图上的显示。保角不保距离。会导致计算的栅格数量明显变多。
        大约为原来的数量的1.3倍。
        初始化地图区域与栅格参数。
        Args:
            lon_min (float): 左下角经度
            lat_min (float): 左下角纬度
            lon_max (float): 右上角经度
            lat_max (float): 右上角纬度
            grid_size_m (int): 栅格间隔(米)

        Raises:
            ValueError: 右上角坐标必须大于左下角坐标
            ValueError: 栅格间隔必须为正数
        """
        if lat_max <= lat_min or lon_max <= lon_min:
            raise ValueError("右上角坐标必须大于左下角坐标")
        if grid_size_m < 0:
            raise ValueError("栅格间隔必须为正数")

        self.lat_min, self.lon_min = lat_min, lon_min
        self.lat_max, self.lon_max = lat_max, lon_max
        self.grid_size_m = grid_size_m

        # 建立经纬度与平面投影坐标的转换器(WGS84 -> UTM)
        self.transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.transformer_to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # 转为米制坐标
        self.x_min, self.y_min = self.transformer_to_utm.transform(lon_min, lat_min)
        self.x_max, self.y_max = self.transformer_to_utm.transform(lon_max, lat_max)

        # 计算栅格行列数
        self.num_cols = math.ceil((self.x_max - self.x_min) / grid_size_m)
        self.num_rows = math.ceil((self.y_max - self.y_min) / grid_size_m)
        self.total_grids = self.num_cols * self.num_rows

    def latlon_to_grid(self, lon, lat):
        """_summary_
        输入经纬度 -> 输出栅格ID(从1开始)
        Args:
            lon (float): 经度。
            lat (float): 纬度。

        Raises:
            ValueError: 输入经纬度超出初始化区域范围

        Returns:
            grid_id (int): 栅格ID
        """
        x, y = self.transformer_to_utm.transform(lon, lat)

        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            raise ValueError("输入经纬度超出初始化区域范围")

        col = int((x - self.x_min) // self.grid_size_m)
        row = int((y - self.y_min) // self.grid_size_m)
        grid_id = row * self.num_cols + col + 1  # 从1开始编号
        return grid_id

    def grid_to_latlon(self, grid_id):
        """_summary_
        输入栅格ID -> 输出栅格中心点经纬度
        Args:
            grid_id (int): 栅格ID。

        Raises:
            ValueError: 输入栅格ID超出范围

        Returns:
            lon_center (float): 中心点的经度。
            lat_center (float): 中心点的纬度。
        """
        if not (1 <= grid_id <= self.total_grids):
            raise ValueError(f"输入栅格ID超出范围(1 ~ {self.total_grids})")

        grid_id -= 1  # 转为0-based
        row = grid_id // self.num_cols
        col = grid_id % self.num_cols

        x_center = self.x_min + (col + 0.5) * self.grid_size_m
        y_center = self.y_min + (row + 0.5) * self.grid_size_m

        lon_center, lat_center = self.transformer_to_geo.transform(x_center, y_center)
        return lon_center, lat_center



class GridMapperUTM:
    """
    使用 UTM 投影实现米级栅格划分。
    相比于椭球法数量仅略微（很少）增加。
    初始化：城市范围经纬度 + 栅格边长（米）
    方法：
        - lonlat_to_grid(lon, lat) -> grid_id
        - grid_to_lonlat(grid_id) -> (lon, lat)
    """

    def __init__(self, min_lon, min_lat, max_lon, max_lat, cell_size_m=1000):
        if max_lon <= min_lon or max_lat <= min_lat:
            raise ValueError("经纬度范围不合法")
        if cell_size_m <= 0:
            raise ValueError("cell_size_m 必须为正")

        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.cell_size_m = cell_size_m

        # 自动计算 UTM 带号：以区域中心经度为准
        center_lon = (min_lon + max_lon) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        is_northern = (min_lat + max_lat) / 2 >= 0  # 北半球为 True
        epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone

        # 创建 Transformer
        self.to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
        self.to_lonlat = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)

        # 转换边界为 UTM 米坐标
        self.x_min, self.y_min = self.to_utm.transform(min_lon, min_lat)
        self.x_max, self.y_max = self.to_utm.transform(max_lon, max_lat)

        # 计算栅格数量
        self.num_x = math.ceil((self.x_max - self.x_min) / cell_size_m)
        self.num_y = math.ceil((self.y_max - self.y_min) / cell_size_m)

    def lonlat_to_grid(self, lon, lat):
        """输入经纬度 -> 输出栅格ID"""
        x, y = self.to_utm.transform(lon, lat)
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            raise ValueError("经纬度超出地图范围")

        gx = int((x - self.x_min) // self.cell_size_m)
        gy = int((y - self.y_min) // self.cell_size_m)
        return gx * self.num_y + gy

    def grid_to_lonlat(self, grid_id):
        """输入栅格ID -> 输出栅格中心经纬度"""
        total_grids = self.num_x * self.num_y
        if not (0 <= grid_id < total_grids):
            raise ValueError(f"grid_id 超出范围 (0 ~ {total_grids-1})")

        gx = grid_id // self.num_y
        gy = grid_id % self.num_y

        # 栅格中心点坐标（米）
        x = self.x_min + (gx + 0.5) * self.cell_size_m
        y = self.y_min + (gy + 0.5) * self.cell_size_m

        # 转回经纬度
        lon, lat = self.to_lonlat.transform(x, y)
        return lon, lat
    

import math

# WGS-84 椭球参数
WGS84_A = 6378137.0                # 赤道半径 (m)
WGS84_F = 1.0 / 298.257223563     # 扁率

def m_per_deg_at_lat(lat_deg, a=WGS84_A, f=WGS84_F):
    """
    返回在给定纬度 (deg) 处：
      (m_per_deg_lat, m_per_deg_lon)
    依据 WGS-84 椭球精确公式。
    """
    phi = math.radians(lat_deg)
    e2 = 2*f - f*f  # e^2
    sinphi = math.sin(phi)
    cosphi = math.cos(phi)

    # 子午圈曲率半径相关 -> 纬度每度米数
    m_per_deg_lat = (math.pi / 180.0) * (a * (1 - e2)) / ((1 - e2 * sinphi * sinphi) ** 1.5)

    # 卯酉圈曲率半径相关 -> 经度每度米数
    m_per_deg_lon = (math.pi / 180.0) * (a * cosphi) / math.sqrt(1 - e2 * sinphi * sinphi)

    return m_per_deg_lat, m_per_deg_lon

class GridMapperEllipsoid:
    """
    使用椭球精确换算 (米 <-> 度)，适合城市级网格：
    初始化：min_lon, min_lat, max_lon, max_lat (deg)，cell_size_m (m)
    提供：lonlat_to_grid_id(lon, lat) / grid_id_to_lonlat(grid_id)
    grid id 从 0 开始
    """
    def __init__(self, min_lon, min_lat, max_lon, max_lat, cell_size_m):
        if max_lon <= min_lon or max_lat <= min_lat:
            raise ValueError("经纬范围不合法")
        if cell_size_m <= 0:
            raise ValueError("cell_size_m 必须为正")

        self.min_lon = float(min_lon)
        self.min_lat = float(min_lat)
        self.max_lon = float(max_lon)
        self.max_lat = float(max_lat)
        self.cell_size_m = float(cell_size_m)

        # 以区域中心纬度为参考计算 m/deg（椭球公式）
        self.center_lat = (self.min_lat + self.max_lat) / 2.0
        mlat, mlon = m_per_deg_at_lat(self.center_lat)
        self.m_per_deg_lat = mlat
        self.m_per_deg_lon = mlon

        # 将米换算为度
        self.cell_deg_lat = self.cell_size_m / self.m_per_deg_lat
        self.cell_deg_lon = self.cell_size_m / self.m_per_deg_lon

        # 计算格子数
        self.num_x = int(math.ceil((self.max_lon - self.min_lon) / self.cell_deg_lon))
        self.num_y = int(math.ceil((self.max_lat - self.min_lat) / self.cell_deg_lat))
        self.total = self.num_x * self.num_y

    def lonlat_to_grid(self, lon, lat):
        lon = float(lon); lat = float(lat)
        if not (self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat):
            raise ValueError("经纬度超出区域范围")

        gx = int((lon - self.min_lon) // self.cell_deg_lon)
        gy = int((lat - self.min_lat) // self.cell_deg_lat)

        # 边界修正
        if gx >= self.num_x:
            gx = self.num_x - 1
        if gy >= self.num_y:
            gy = self.num_y - 1

        return gx * self.num_y + gy

    def grid_to_lonlat(self, grid_id):
        if not (0 <= grid_id < self.total):
            raise ValueError(f"grid_id 超出范围 (0 ~ {self.total-1})")
        gx = grid_id // self.num_y
        gy = grid_id % self.num_y

        lon_center = self.min_lon + (gx + 0.5) * self.cell_deg_lon
        lat_center = self.min_lat + (gy + 0.5) * self.cell_deg_lat
        return lon_center, lat_center

    def info(self):
        return {
            "center_lat": self.center_lat,
            "m_per_deg_lat": self.m_per_deg_lat,
            "m_per_deg_lon": self.m_per_deg_lon,
            "cell_deg_lat": self.cell_deg_lat,
            "cell_deg_lon": self.cell_deg_lon,
            "num_x": self.num_x,
            "num_y": self.num_y,
            "total": self.total
        }