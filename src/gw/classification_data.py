"""Validated classification manifest datasets."""

from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np


def load_manifest(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as h:rows=list(csv.DictReader(h))
    required={"sample_id","class_id","split","data_path"}
    if not rows or not required.issubset(rows[0]):raise ValueError(f"manifest requires {sorted(required)}")
    ids=[r["sample_id"] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate sample_id")
    unknown=sorted({r["split"] for r in rows}-{ "train","val","test"})
    if unknown:raise ValueError(f"unknown splits {unknown}")
    return rows


def manifest_sha256(path:Path)->str:
    """Hash semantic split content without binding a run to server mount paths."""
    rows=load_manifest(path);digest=hashlib.sha256()
    for row in sorted(rows,key=lambda value:value["sample_id"]):
        stable={key:value for key,value in row.items() if not key.endswith("_path") and key!="path"}
        digest.update(json.dumps(stable,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8"));digest.update(b"\n")
    return digest.hexdigest()


def _normalize_array(arr:np.ndarray)->np.ndarray:
    arr=np.asarray(arr,dtype=np.float32);finite=np.isfinite(arr)
    if not finite.any():return np.zeros_like(arr,dtype=np.float32)
    lo=float(np.percentile(arr[finite],1));hi=float(np.percentile(arr[finite],99))
    return np.zeros_like(arr,dtype=np.float32) if hi<=lo else np.clip((arr-lo)/(hi-lo),0,1).astype(np.float32)


def _resize_array(arr:np.ndarray,image_size:int|None)->np.ndarray:
    """Match the historical runner: quantize to uint8, then PIL bilinear resize."""
    if image_size is None or arr.shape==(image_size,image_size):return np.asarray(arr,dtype=np.float32)
    from PIL import Image
    image=Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8),mode="L")
    return np.asarray(image.resize((image_size,image_size),Image.Resampling.BILINEAR),dtype=np.float32)/255.0


def _eurosat_sar_vv_vh(arr:np.ndarray,image_size:int|None)->np.ndarray:
    """Reproduce FG-MAE's public VV/VH scaling while retaining both channels."""
    if arr.ndim!=3 or arr.shape[-1]!=2:
        raise ValueError(f"EuroSAT-SAR expects HWC with two bands, got {arr.shape}")
    means=(-12.59,-20.26);stds=(5.26,5.91);channels=[]
    for index,(mean,std) in enumerate(zip(means,stds)):
        channel=np.asarray(arr[...,index],dtype=np.float32).copy();finite=np.isfinite(channel)
        if not finite.any():channel=np.zeros_like(channel,dtype=np.float32)
        else:
            lo=float(np.quantile(channel[finite],0.01));hi=float(np.quantile(channel[finite],0.99))
            channel=np.nan_to_num(channel,nan=lo,posinf=hi,neginf=lo);channel=np.clip(channel,lo,hi)
            channel=np.clip((channel-(mean-2*std))/(4*std),0,1).astype(np.float32)
        channels.append(_resize_array(channel,image_size))
    return np.stack(channels,axis=0)


def balanced_sample_weights(rows:list[dict[str,str]])->list[float]:
    counts:dict[int,int]={}
    labels=[int(row["class_id"]) for row in rows]
    for label in labels:counts[label]=counts.get(label,0)+1
    return [1.0/counts[label] for label in labels]


class ManifestDataset:
    def __init__(self,rows:list[dict[str,str]],split:str,limit:int|None=None,seed:int=0,image_size:int|None=None,data_root:Path|None=None):
        import torch
        self.torch=torch;self.image_size=image_size;self.data_root=data_root;items=[r for r in rows if r["split"]==split]
        if limit is not None and len(items)>limit:
            rng=np.random.default_rng(seed+sum(map(ord,split)));idx=np.sort(rng.choice(len(items),limit,replace=False));items=[items[int(i)] for i in idx]
        self.rows=items
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];path=Path(row["data_path"]);kind=row.get("data_type","")
        if not path.is_absolute() and self.data_root is not None:path=self.data_root/path
        channel_mode=row.get("channel_mode","gray")
        if channel_mode=="eurosat_sar_vv_vh":
            if path.suffix.lower() not in {".tif",".tiff"}:raise ValueError(f"EuroSAT-SAR requires TIFF: {path}")
            try:import tifffile
            except ImportError as exc:raise RuntimeError("EuroSAT-SAR TIFF input requires tifffile and imagecodecs") from exc
            arr=_eurosat_sar_vv_vh(tifffile.imread(path),self.image_size)
            tensor=self.torch.from_numpy(arr).float()
            return tensor,self.torch.tensor(int(row["class_id"]),dtype=self.torch.long)
        if kind in {"image_path","geotiff_path"} or path.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}:
            from PIL import Image
            with Image.open(path) as image:arr=np.asarray(image.convert("L"),dtype=np.float32)/255.0
        elif kind in {"npy_float32",""} or path.suffix.lower()==".npy":arr=_normalize_array(np.load(path))
        else:raise ValueError(f"unsupported data_type {kind!r}: {path}")
        arr=np.asarray(arr,dtype=np.float32)
        if arr.ndim==3:arr=arr.mean(axis=2)
        if arr.ndim!=2:raise ValueError(f"expected 2D array: {row['data_path']}")
        arr=_resize_array(arr,self.image_size);tensor=self.torch.from_numpy(arr[None]).float()
        return tensor,self.torch.tensor(int(row["class_id"]),dtype=self.torch.long)


NpyManifestDataset=ManifestDataset
