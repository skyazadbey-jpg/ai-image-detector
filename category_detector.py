"""Conservative ImageNet object hints; never an AI or fraud score."""
import io,json,hashlib,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABEL_SHA='91a3a6afaeb2a047fc854406fad4a8513ba0719421d50ce8786816954ecf72d2'
MODEL_SHA='70fc6f84dfd48becf4f409a6372af34f8028d1d2a83d1f38d6762a9550f79bb2'
CAR={'beach wagon','cab','convertible','jeep','limousine','minivan','Model T','pickup','racer','sports car','ambulance','minibus'}
PROPERTY={'bathtub','boathouse','bookcase','castle','china cabinet','church','desk','dining table','four-poster','greenhouse','library','mobile home','monastery','mosque','palace','patio','restaurant','studio couch','wardrobe','washbasin','medicine chest','chiffonier','entertainment center','fire screen','window shade','window screen','tub','shower curtain','barn'}
PROPERTY.update({'stove','microwave','dishwasher','washer','refrigerator','plate rack'})
_session=None
_lock=threading.Lock()

def model_path():
    target=ROOT/'category_model.onnx'
    def digest(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    if target.is_file() and digest(target)==MODEL_SHA:return target
    manifest=json.loads((ROOT/'category_model_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('sha256')!=MODEL_SHA:raise RuntimeError('Invalid category model manifest')
    temporary=ROOT/'category_model.assembling'
    try:
        with temporary.open('wb') as output:
            for i,part in enumerate(manifest['parts']):
                name=f'category_model.part{i:02d}'
                if part['name']!=name:raise RuntimeError('Invalid category part name')
                source=ROOT/name
                if source.stat().st_size!=part['bytes'] or digest(source)!=part['sha256']:raise RuntimeError('Invalid category model part')
                with source.open('rb') as f:
                    while chunk:=f.read(1024*1024):output.write(chunk)
        if temporary.stat().st_size!=manifest['bytes'] or digest(temporary)!=MODEL_SHA:raise RuntimeError('Invalid category model')
        temporary.replace(target)
        return target
    finally:temporary.unlink(missing_ok=True)

def classify(contents,mode):
    global _session
    if mode not in ('car','realestate'):return {'status':'not_checked','detected':'unknown'}
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    with _lock:
        if _session is None:
            path=model_path()
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            if session.get_inputs()[0].shape!=[1,3,224,224] or session.get_outputs()[0].shape!=[1,1000]:raise RuntimeError('Invalid category model shape')
            _session=session
        if not contents or len(contents)>20*1024*1024:raise ValueError('Invalid image size')
        with Image.open(io.BytesIO(contents)) as image:
            w,h=image.size
            if w*h>20_000_000:raise ValueError('Image too large')
            size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
            if max(size)>8192:raise ValueError('Unsupported aspect ratio')
            im=image.convert('RGB').resize(size,Image.Resampling.BILINEAR)
            left=round((size[0]-224)/2);top=round((size[1]-224)/2)
            values=np.asarray(im.crop((left,top,left+224,top+224)),dtype=np.float32)/255
        values=(values-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        logits=_session.run(['logits'],{'pixel_values':values.transpose(2,0,1)[None].copy()})[0][0]
        if logits.shape!=(1000,) or not np.isfinite(logits).all():raise RuntimeError('Invalid category response')
        scores=np.exp(logits-logits.max());scores/=scores.sum()
        label_bytes=(ROOT/'category_labels.json').read_bytes()
        if hashlib.sha256(label_bytes).hexdigest()!=LABEL_SHA:raise RuntimeError('Invalid category label mapping')
        labels=json.loads(label_bytes)
        car=float(sum(scores[i] for i,s in enumerate(labels) if s in CAR))
        property_hint=float(sum(scores[i] for i,s in enumerate(labels) if s in PROPERTY))
        animal=float(scores[:398].sum())
        detected='car' if car>=.65 else ('property_related' if property_hint>=.65 else ('other' if animal>=.85 else 'unknown'))
        status='uncertain'
        if (mode=='car' and detected in ('property_related','other')) or (mode=='realestate' and detected in ('car','other')):status='mismatch'
        elif (mode=='car' and detected=='car') or (mode=='realestate' and detected=='property_related'):status='compatible'
        return {'status':status,'detected':detected,'method':'local_object_hint','car_score':round(car,4),'property_related_score':round(property_hint,4),'not_a_category_guarantee':True}
