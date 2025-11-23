import json  # 导入 JSON 解析库
from pathlib import Path  # 导入路径处理工具
from typing import List, Optional, Sequence  # 导入类型注解

import av  # 导入 PyAV 视频解码库
import torch  # 导入 PyTorch
from torch.utils.data import Dataset  # 导入数据集基类
import torchvision.transforms as transforms  # 导入图像变换模块


class SomethingSomethingV2Dataset(Dataset):  # 定义数据集类
    """Dataset wrapper for 20bn-something-something-v2."""  # 类文档字符串

    def __init__(  # 构造函数
        self,  # 实例本身
        data_root: str,  # 数据根目录
        split: str = "train",  # 数据集划分
        resolution: int = 96,  # 输出图像分辨率
        context_frames: int = 20,  # 上下文帧数
        label_dir: Optional[str] = None,  # 标签目录
        video_dirs: Optional[Sequence[str]] = None,  # 视频目录列表
        is_training: bool = True,  # 是否训练模式
        color_aug: bool = False,  # 是否颜色增强
        max_samples: Optional[int] = None,  # 最多加载样本数
        label_limit: Optional[int] = None,  # 仅读取标签文件的前 n 条
    ) -> None:
        super().__init__()  # 调用父类构造
        if context_frames < 1:  # 确保上下文帧数合法
            raise ValueError("context_frames must be >= 1")  # 抛出异常

        self.data_root = Path(data_root)  # 保存根目录
        self.split = split  # 保存划分名
        self.resolution = resolution  # 保存分辨率
        self.context_frames = context_frames  # 保存上下文帧数
        self.required_frames = context_frames + 1  # includes target frame  # 计算需要的总帧数
        self.is_training = is_training  # 保存训练模式标志
        self.color_aug = color_aug  # 保存颜色增强标志

        self.label_dir = Path(label_dir) if label_dir else self._default_label_dir()  # 确定标签目录
        self.video_dirs = (  # 确定视频目录
            [Path(p) for p in video_dirs] if video_dirs else self._discover_video_dirs()  # 使用显式参数或自动发现
        )
        if not self.video_dirs:  # 若未找到视频目录
            raise FileNotFoundError(  # 抛出异常
                "No video directories found. Pass `video_dirs` explicitly or "  # 错误提示
                "ensure folders named 20bn-something-something-v2* exist under data_root."
            )

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 构造归一化
        transform_ops = []  # 初始化变换列表
        if self.is_training and self.color_aug:  # 若训练且启用颜色增强
            transform_ops.append(  # 添加颜色抖动
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                )
            )
        transform_ops.extend(  # 追加基础变换
            [
                transforms.Resize((self.resolution, self.resolution)),  # 调整大小
                transforms.ToTensor(),  # 转为张量
                normalize,  # 归一化
            ]
        )
        self.transform = transforms.Compose(transform_ops)  # 合成变换

        label_path = self._resolve_label_file(split)  # 计算标签文件路径

        with open(label_path, "r", encoding="utf-8") as file:  # 打开标签文件
            entries = json.load(file)  # 读取 JSON
        if label_limit is not None:  # 若设置了标签截断
            entries = entries[:label_limit]  # 只保留前 label_limit 条

        self.samples = []  # 存放样本列表
        missing = 0  # 统计缺失视频
        for entry in entries:  # 遍历标签
            video_path = self._locate_video(entry["id"])  # 查找视频
            if video_path is None:  # 未找到
                missing += 1  # 累加缺失
                continue  # 跳过
            self.samples.append({"meta": entry, "video_path": video_path})  # 保存样本
            if max_samples and len(self.samples) >= max_samples:  # 达到上限
                break  # 停止加载

        if not self.samples:  # 若没有样本
            raise RuntimeError("No valid samples found for the requested split.")  # 抛出异常

        if missing:  # 若存在缺失
            print(  # 打印提示
                f"[SomethingSomethingV2Dataset] Skipped {missing} items without available videos."
            )

        progress_ratio = min(self.context_frames / self.required_frames, 0.999)  # 计算进度比例
        self.progress_value = int(progress_ratio * 10)  # 量化为整数

        print("=" * 20)  # 打印分隔线
        print(f"{len(self.samples)} samples found for split='{split}'.")  # 打印统计

    def _default_label_dir(self) -> Path:  # 默认标签目录
        candidate = self.data_root / "labels"  # 假定目录
        if candidate.exists():  # 存在则返回
            return candidate  # 返回目录
        raise FileNotFoundError(  # 不存在则报错
            "Cannot locate labels directory automatically. Provide `label_dir`."
        )

    def _resolve_label_file(self, split: str) -> Path:
        """Locate the JSON file for a given split, including subset folders."""
        filename = split if split.endswith(".json") else f"{split}.json"
        candidates = [self.label_dir / filename]
        subset_roots = [
            path for path in self.label_dir.iterdir() if path.is_dir()
        ]
        candidates.extend(subdir / filename for subdir in subset_roots)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Label file not found for split '{split}'. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    def _discover_video_dirs(self) -> List[Path]:  # 自动发现视频目录
        prefix = "20bn-something-something-v2"  # 目录前缀
        if self.data_root.name.startswith(prefix) and self.data_root.is_dir():  # 根目录即视频目录
            return [self.data_root]  # 返回列表
        dirs = sorted(  # 过滤子目录
            p
            for p in self.data_root.iterdir()
            if p.is_dir() and p.name.startswith(prefix)
        )
        return dirs  # 返回集合

    def _locate_video(self, video_id: str) -> Optional[Path]:  # 定位视频文件
        video_id = str(video_id)  # 转为字符串
        filename = f"{video_id}.webm"  # 拼接文件名
        for directory in self.video_dirs:  # 遍历目录
            candidate = directory / filename  # 拼接路径
            if candidate.exists():  # 若存在
                return candidate  # 返回路径
        return None  # 未找到返回空

    def __len__(self) -> int:  # 返回数据集长度
        return len(self.samples)  # 样本数量

    def __getitem__(self, index: int) -> dict:  # 按索引取样本
        sample = self.samples[index]  # 获取样本字典
        entry = sample["meta"]  # 标签元信息
        video_path = sample["video_path"]  # 视频路径

        frames = self._load_video_frames(video_path)  # 加载帧序列
        context = torch.stack(frames[: self.context_frames], dim=0)  # 堆叠前20帧
        target = frames[self.context_frames]  # 第21帧作为目标
        averaged_context = context.mean(dim=0)  # 前20帧求平均作为IP2P输入

        text_prompt = entry.get("label") or entry.get("template", "")  # 标签文本
        video_id = str(entry.get("id"))  # 视频 ID

        example = {  # 构建输出
            "video_id": video_id,  # 视频编号
            "input_text": [text_prompt],  # 文本列表
            "context_pixel_values": context,  # 上下文张量
            "target_pixel_values": target,  # 目标张量
            "progress": self.progress_value,  # 进度标签
            "original_pixel_values": averaged_context,  # 供 IP2P 使用的输入帧
            "edited_pixel_values": target,  # 供 IP2P 使用的目标帧
        }
        return example  # 返回样本

    def _load_video_frames(self, video_path: Path) -> List[torch.Tensor]:  # 加载帧
        frames: List[torch.Tensor] = []  # 存储帧列表
        with av.open(video_path) as container:  # 打开视频
            stream = container.streams.video[0]  # 获取视频流
            for frame in container.decode(stream):  # 逐帧解码
                image = frame.to_image().convert("RGB")  # 转为 PIL
                tensor = self.transform(image)  # 应用预处理
                frames.append(tensor)  # 保存张量
                if len(frames) >= self.required_frames:  # 足够帧数
                    break  # 停止解码

        if not frames:  # 若没有帧
            raise RuntimeError(f"Failed to decode frames from {video_path}")  # 抛出异常

        last_frame = frames[-1]  # 取最后一帧
        while len(frames) < self.required_frames:  # 若不足
            frames.append(last_frame.clone())  # 重复最后一帧
        return frames  # 返回帧序列


if __name__ == "__main__":  # 脚本入口
    data_root = "/Users/jiatong/Desktop/常用文件/研一上/MDS5122_DL/Final_Project/"  # contains labels/ and 20bn-something-something-v2*  # 示例根目录
    if Path(data_root).exists():  # 若路径存在
        dataset = SomethingSomethingV2Dataset(  # 构建数据集
            data_root=data_root,  # 根目录
            split="train_cover_object",  # 训练集
            resolution=96,  # 输出尺寸
            context_frames=20,  # 上下文帧数
            is_training=True,  # 训练模式
            color_aug=False,  # 不启用颜色增强
            max_samples=10,  # 只取10个样本
        )
        sample = dataset[0]  # 读取一个样本
        print("Context shape:", sample["context_pixel_values"].shape)  # 输出上下文形状
        print("Target shape:", sample["target_pixel_values"].shape)  # 输出目标形状
        print("Text:", sample["input_text"][0])  # 输出文本
