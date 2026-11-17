import json
from pathlib import Path
import shutil
import numpy as np

def write_json(path,value):
    def default(v):
        if isinstance(v,(np.integer,np.floating)): return v.item()
        raise TypeError(str(type(v)))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=default,allow_nan=False),encoding='utf-8')

def export_web(results,bundle):
    public=Path('website/public/research')
    public.mkdir(parents=True,exist_ok=True)
    (public/'experiment.json').write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False),encoding='utf-8')
    for name in ['test_predictions.csv','model_comparison.csv','feature_inventory.json','experiment_metadata.json']:
        shutil.copyfile(Path(results)/name,public/name)
    shutil.copytree(Path(results)/'figures',public/'figures',dirs_exist_ok=True)
