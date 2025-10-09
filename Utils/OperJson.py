import json
from typing import Any

class JSONConfig:
    """_summary_
    对在parameters.json中存储的全局变量进行操作。
    """
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self):
        """加载 JSON 文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self)->None:
        """保存数据到 JSON 文件"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key:str, default=None) -> Any:
        """获取 JSON 变量"""
        return self.data.get(key, default)

    def set(self, key:str, value:Any)->None:
        """设置 JSON 变量并保存"""
        self.data[key] = value
        self.save()

    def delete(self, key:str)->None:
        """删除 JSON 变量并保存"""
        if key in self.data:
            del self.data[key]
            self.save()