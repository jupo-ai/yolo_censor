from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from PIL import Image, ImageSequence
from collections import defaultdict
from typing import Union, Literal
import questionary
import torch
import toml
import cv2
import math
import numpy as np
import argparse
import imageio


# ===============================================
# コンフィグ
# ===============================================
@dataclass
class GeneralSetting:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch: int = 1
    mode: int = 0 # 0: mosaic, 1: black fill, 2: image
    mosaic_size: int = 0 # 0: auto, 1~: specific size
    mask_image: str = ""
    
    def __post_init__(self):
        if self.device == "cuda":
            if not torch.cuda.is_available():
                print("cuda使用不可のためcpuで処理します")
                self.device = "cpu"
        
        if (self.mode == 2) and (not self.mask_image):
            print("mask_imageが設定されていません。モザイクモードに変更します")
            self.mode = 0
            

@dataclass
class ModelInfoSetting:
    path: str = ""
    label: list[str] = field(default_factory=list)
    threshold: float = 0.5
    mask: Literal["segm", "bbox"] = "bbox"
    mask_offset: int = 4
    
    model: YOLO = field(init=False)
    
    def __post_init__(self):
        if not isinstance(self.label, list):
            self.label = [self.label]
        
        if self.path:
            self.model = YOLO(self.path, verbose=False)

@dataclass
class RootConfig:
    general: GeneralSetting
    models: list[ModelInfoSetting]


def load_config():
    config = {}
    with open("config.toml", mode="r", encoding="utf-8") as file:
        config = toml.load(file)
    
    general = GeneralSetting(**config.get("general", {}))
    model_list = config.get("model", [])
    models = []
    for info in model_list:
        models.append(ModelInfoSetting(**info))
    
    return RootConfig(general=general, models=models)



# ===============================================
# ファイルの仕分け
# 画像: png, jpg, jpeg, webp
# 動画: mp4, webp
# ===============================================
def classification_pathes(files: list[Path]):
    images = []
    videos = []
    for file in files:
        if file.suffix == ".webp":
            if webp_is_animated(file):
                videos.append(file)
            else:
                images.append(file)
        
        elif file.suffix in [".png", ".jpg", ".jpeg"]:
            images.append(file)
        
        elif file.suffix in [".mp4"]:
            videos.append(file)
    
    return images, videos


def webp_is_animated(file: Path):
    try:
        with Image.open(file) as img:
            return img.is_animated
    except Exception as e:
        print(f"Error: {e}")
        return False


# ===============================================
# バッチ作成
# ===============================================
def create_batches(files: list[Path], size):
    batches = []
    batch = []
    for file in files:
        batch.append(file)
        if len(batch) == size:
            batches.append(batch)
            batch = []
    
    if batch:
        batches.append(batch)
    
    return batches


# ===============================================
# 検閲処理
# ===============================================
def apply_mosaic(image: np.ndarray, region: np.ndarray, config: RootConfig) -> np.ndarray:
    """
    マスク領域にモザイク処理を適用する。
    モザイクサイズはオートの場合pixivより max(4, long/100) とする
    """
    height, width = image.shape[:2]
    if config.general.mosaic_size == 0:
        mosaic_size = math.ceil(max(4, max(height, width) / 100))
    else:
        mosaic_size = config.general.mosaic_size
    masked_image = image.copy()
    mask_indices = region > 0
    
    # モザイク化
    small_image = cv2.resize(
        masked_image, 
        (width // mosaic_size, height // mosaic_size), 
        interpolation=cv2.INTER_NEAREST
    )
    mosaic_image = cv2.resize(
        small_image, 
        (width, height), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # マスク領域にモザイクを適用
    masked_image[mask_indices] = mosaic_image[mask_indices]
    return masked_image


def apply_black_fill(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """
    マスク領域を黒で塗りつぶす
    """
    masked_image = image.copy()
    mask_indices = region > 0
    masked_image[mask_indices] = [0, 0, 0]
    return masked_image


def trim_transparent(image: np.ndarray):
    image_data = np.array(image)

    alpha = image_data[:, :, 3]
    non_transparent = alpha > 0

    if np.any(non_transparent):
        rows = np.any(non_transparent, axis=1)
        cols = np.any(non_transparent, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return image.crop((x_min, y_min, x_max+1, y_max+1))
    else:
        return image


def apply_mask_image(image: np.ndarray, region: np.ndarray, config: RootConfig) -> np.ndarray:
    mask_image_path = config.general.mask_image
    mask_image = Image.open(mask_image_path).convert("RGBA")

    # region の bbox を取得
    y_ids, x_ids = np.where(region > 0)
    if len(x_ids) == 0 or len(y_ids) == 0:
        return image
    
    x_min, x_max = x_ids.min(), x_ids.max()
    y_min, y_max = y_ids.min(), y_ids.max()
    region_center_x = (x_min + x_max) // 2
    region_center_y = (y_min + y_max) // 2
    region_width = x_max - x_min
    region_height = y_max - y_min
    
    mask_image = trim_transparent(mask_image) # mask_imageの透明部分を最小限になるようトリム
    resize_scale = region_width / mask_image.size[0]
    resize_width = int(mask_image.size[0] * resize_scale)
    resize_height = int(mask_image.size[1] * resize_scale)
    mask_resized = mask_image.resize((resize_width, resize_height), Image.LANCZOS)

    # マスク画像の貼り付け位置を計算(中心を合わせる)
    mask_w, mask_h = mask_resized.size
    paste_x = region_center_x - mask_w // 2
    paste_y = region_center_y - mask_h // 2
    
    # 画像にマスクを合成
    image_pil = Image.fromarray(image)
    image_pil.paste(mask_resized, (paste_x, paste_y), mask_resized)

    image = np.array(image_pil)
    return image

    


# ===============================================
# メイン処理
# ===============================================
# 領域推論
def predict_regions(input: Union[list[Path], np.ndarray], config: RootConfig):
    """
    各モデルで推論した領域を取得する
    
    Args:
        input: 入力データ (画像パスリスト | 単一画像np.ndarray)
        models: モデル情報のリスト (YOLOインスタンス, label, threshold)
        config: 設定オブジェクト
    
    Returns:
        regions: 検出された領域のリスト(np.ndarray) | dict(パスとnp.ndarrayリストのマップ)
    """
    should_mapping = isinstance(input, list)
    regions = defaultdict(list) if should_mapping else []
    if should_mapping: # 何も検出しなかった時ように明示的に初期化
        for p in input:
            regions[str(p)] = []

    # 複数モデルで検出
    for model_info in config.models:
        results = model_info.model.predict(input, save=False, verbose=False, device=config.general.device)

        for result in results:
            boxes = result.boxes.xyxy
            masks = result.masks
            clses = result.boxes.cls
            confs = result.boxes.conf
            
            for i, (box, cls, conf) in enumerate(zip(boxes, clses, confs)):
                class_name = model_info.model.names[int(cls)]
                if class_name in model_info.label and conf >= model_info.threshold:
                    # 領域の取得方法
                    if config.general.mode == 2: # if mode is 'mask image' then, mask type is bbox
                        model_info.mask = "bbox"
                                        
                    if masks is not None and i < len(masks) and model_info.mask == "segm":
                        contour = masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
                        region = np.zeros(result.orig_shape, dtype=np.uint8)
                        cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
                    else:
                        x_min, y_min, x_max, y_max = box.int().tolist()
                        mask = np.zeros(result.orig_shape, dtype=np.uint8)
                        mask[y_min:y_max, x_min:x_max] = 1
                        region = mask
                    
                    # 領域をoffset分だけ拡張
                    offset = int(model_info.mask_offset)
                    if offset > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * offset + 1, 2 * offset + 1))
                        region = cv2.dilate(region, kernel, iterations=1)
                else:
                    region = None

                if should_mapping:
                    regions[result.path].append(region)
                else:
                    regions.append(region)
    
    return regions


# 画像
def process_images(images: list[Path], config: RootConfig):
    print(f"画像処理を始めます: file数 {len(images)}")
    
    # バッチ作成
    batches = create_batches(images, config.general.batch)
    
    bar = tqdm(total=len(images))
    
    for batch in batches:
        regions_map = predict_regions(batch, config)
        
        # 後処理
        for image_path, regions in regions_map.items():
            bar.set_description(image_path)

            pil_image = Image.open(image_path).convert("RGBA")
            image = np.array(pil_image)

            should_save = False
            for region in regions:
                if region is not None:
                    should_save = True
                    
                    if config.general.mode == 0:
                        image = apply_mosaic(image, region, config)
                    elif config.general.mode == 1:
                        image = apply_black_fill(image, region)
                    elif config.general.mode == 2:
                        image = apply_mask_image(image, region, config)
            
            if should_save:
                save_image(image, image_path)

            bar.update(1)
    
    bar.close()


def save_image(image: np.ndarray, orig_path: str):
    output_dir = Path(orig_path).parent / "censored"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(orig_path).name
    
    pil_output = Image.fromarray(image)
    
    pil_output.save(str(output_path))



# 動画
def process_videos(videos: list[Path], config: RootConfig):
    print(f"動画処理を始めます: file数 {len(videos)}")

    for file in videos:
        # フレームジェネレータとその他情報を取得
        frames, count, fps = get_video_frames(file)
        if frames is None:
            return
        
        bar = tqdm(total=count, desc=str(file))
        processed_frames = []

        should_save = False
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            regions = predict_regions(frame_bgr, config)
            
            for region in regions:
                if region is not None:
                    should_save = True
                    
                    if config.general.mode == 0:
                        frame_bgr = apply_mosaic(frame_bgr, region)
                    elif config.general.mode == 1:
                        frame_bgr = apply_black_fill(frame_bgr, region)

            processed_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_frames.append(processed_frame)
            bar.update(1)

        if should_save or True:
            save_video(processed_frames, file, fps, config)
        bar.close()
        

def save_video(frames: list[np.ndarray], orig_path: Path, fps: float, config: RootConfig):
    output_dir = Path(orig_path).parent / "censored"
    output_dir.mkdir(parents=True, exist_ok=True)
    ouput_path = output_dir / Path(orig_path).name
    
    # webp
    if orig_path.suffix.lower() == ".webp":
        pil_frames = [Image.fromarray(frame) for frame in frames]
        duration = int(1000 / fps)
        imageio.mimwrite(ouput_path, pil_frames, format="webp", duration=duration, loop=0)
    pass


def get_video_frames(file: Path):
    """
    動画からフレーム画像をジェネレータとして取得する
    """
    if file.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(file))
        if not cap.isOpened():
            print(f"動画を開けませんでした: {file}")
            return None, None, None
        
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        def frame_generator():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
        
        return frame_generator(), count, fps
    
    elif file.suffix.lower() == ".webp":
        img = Image.open(file)
        if not img.is_animated:
            print(f"アニメーションでないwebp: {file}")
            return None, None, None
        
        count = img.n_frames
        fps = 1000 / img.info.get("duration", 100) # durationはms単位、FPSに変換
        
        def frame_generator():
            for frame in ImageSequence.Iterator(img):
                yield np.array(frame.convert("RGB"))
        
        return frame_generator(), count, fps
    
    else:
        print(f"サポートされていない形式: {file}")
        return None, None, None
                

def main(args):
    config = load_config()
    
    # 対象フォルダ / ファイル
    target_path = args.target
    if not target_path:
        target_path = questionary.text("対象フォルダ / ファイル").ask().strip('"').strip()
    if not target_path:
        return
    target_path = Path(target_path)
    if not target_path.exists():
        print("対象フォルダ/ファイルが存在しません")
        return
    
    files = []
    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = list(target_path.glob("*.*"))
    
    # ファイルを仕分ける
    images, videos = classification_pathes(files)
    
    
    # 画像を処理
    if images:
        process_images(images, config)
    
    # 動画を処理
    if videos:
        pass
        # process_videos(videos, config)
        



    
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    args, _ = parser.parse_known_args()
    main(args)
