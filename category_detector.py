"""Two-view object evidence gate; unknown inputs require General Scan. Not an AI/fraud score."""
import io,json,hashlib,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABEL_SHA='91a3a6afaeb2a047fc854406fad4a8513ba0719421d50ce8786816954ecf72d2'
MODEL_SHA='70fc6f84dfd48becf4f409a6372af34f8028d1d2a83d1f38d6762a9550f79bb2'
CAR={'beach wagon','cab','convertible','jeep','limousine','minivan','Model T','pickup','racer','sports car','ambulance','minibus'}
PROPERTY={'bathtub','boathouse','bookcase','castle','china cabinet','church','desk','dining table','four-poster','greenhouse','library','mobile home','monastery','mosque','palace','patio','restaurant','studio couch','wardrobe','washbasin','medicine chest','chiffonier','entertainment center','fire screen','window shade','window screen','tub','shower curtain','barn'}
PROPERTY.update({'stove','microwave','dishwasher','washer','refrigerator','plate rack'})
# Non-residential buildings and buses do not qualify for the specialist sections.
PROPERTY.difference_update({'church','monastery','mosque','palace','castle','restaurant','library','greenhouse','barn','boathouse'})
CAR.difference_update({'ambulance','minibus'})
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

# Parts/textiles are supporting evidence, never sufficient on their own.
CAR_SUPPORT={'car wheel','grille','seat belt'}
PROPERTY_SUPPORT={'quilt','pillow','home theater','sliding door','shoji','table lamp','toilet seat','radiator','rocking chair','cradle','crib','mosquito net'}
# Furniture/fixtures can appear in a room or in a close-up; this is a hint, not scene verification.
PROPERTY.update({'home theater','sliding door','shoji','toilet seat','crib','cradle','rocking chair'})
POLICY_VERSION='two_view_evidence_v3'
_labels=None

def decide_evidence(scores, labels):
    import numpy as np
    car=float(sum(scores[i] for i,s in enumerate(labels) if s in CAR))
    prop=float(sum(scores[i] for i,s in enumerate(labels) if s in PROPERTY))
    car_support=float(sum(scores[i] for i,s in enumerate(labels) if s in CAR_SUPPORT))
    property_support=float(sum(scores[i] for i,s in enumerate(labels) if s in PROPERTY_SUPPORT and s not in PROPERTY))
    animal=float(scores[:398].sum())
    top=int(np.argmax(scores));label=labels[top];top_score=float(scores[top])
    car_ok=(car>=.5 or (car>=.2 and label in CAR and top_score>=.15) or (car>=.15 and car+car_support>=.65)) and car>=3*prop
    property_ok=(prop>=.5 or (prop>=.05 and prop+property_support>=.65)) and prop>=3*car
    detected='car' if car_ok else ('property_related' if property_ok else ('other' if animal>=.85 else 'unknown'))
    return {'detected':detected,'car_score':round(car,4),'property_related_score':round(prop,4)}

def combine_evidence(views, mode):
    signals={v['detected'] for v in views if v['detected']!='unknown'}
    detected=next(iter(signals)) if len(signals)==1 else 'unknown'
    # Weak evidence needs agreement from both views. One view is enough only
    # at the former .65 group confidence; contradictory views always abstain.
    if detected in {'car','property_related'} and sum(v['detected']==detected for v in views)<2:
        key='car_score' if detected=='car' else 'property_related_score'
        if not any(v['detected']==detected and v[key]>=.65 for v in views):detected='unknown'
    expected='car' if mode=='car' else 'property_related'
    status='compatible' if detected==expected else ('mismatch' if detected in {'car','property_related','other'} else 'uncertain')
    return {'status':status,'detected':detected,'method':'local_object_hint_two_views','policy_version':POLICY_VERSION,
            'car_score':max(v['car_score'] for v in views),'property_related_score':max(v['property_related_score'] for v in views),
            'views':views,'not_a_category_guarantee':True}

def classify(contents,mode):
    global _session,_labels
    if mode not in ('car','realestate'):return {'status':'not_checked','detected':'unknown'}
    import numpy as np
    import onnxruntime as ort
    from PIL import Image,ImageOps
    with _lock:
        if _session is None:
            path=model_path()
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            if session.get_inputs()[0].shape!=[1,3,224,224] or session.get_outputs()[0].shape!=[1,1000]:raise RuntimeError('Invalid category model shape')
            _session=session
        if _labels is None:
            label_bytes=(ROOT/'category_labels.json').read_bytes()
            if hashlib.sha256(label_bytes).hexdigest()!=LABEL_SHA:raise RuntimeError('Invalid category label mapping')
            _labels=json.loads(label_bytes)
        if not contents or len(contents)>20*1024*1024:raise ValueError('Invalid image size')
        with Image.open(io.BytesIO(contents)) as image:
            w,h=image.size
            if not w or not h or w*h>20_000_000:raise ValueError('Invalid image dimensions')
            size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
            if max(size)>8192:raise ValueError('Unsupported aspect ratio')
            source=image.convert('RGB');im=source.resize(size,Image.Resampling.BILINEAR)
            left=round((size[0]-224)/2);top=round((size[1]-224)/2)
            # The second view preserves the whole photo without changing its aspect ratio.
            images=[('center',im.crop((left,top,left+224,top+224))),('whole',ImageOps.pad(source,(224,224),method=Image.Resampling.BILINEAR,color=(124,116,104)))]
        views=[]
        for name,image in images:
            values=np.asarray(image,dtype=np.float32)/255
            values=(values-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
            logits=_session.run(['logits'],{'pixel_values':values.transpose(2,0,1)[None].copy()})[0][0]
            if logits.shape!=(1000,) or not np.isfinite(logits).all():raise RuntimeError('Invalid category response')
            scores=np.exp(logits-logits.max());scores/=scores.sum()
            views.append(dict(view=name,**decide_evidence(scores,_labels)))
        return combine_evidence(views,mode)
