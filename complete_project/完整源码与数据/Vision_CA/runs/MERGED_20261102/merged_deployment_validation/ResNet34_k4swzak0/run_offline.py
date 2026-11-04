import importlib.util, socket, sys, urllib.request
from pathlib import Path
import torch
def deny(*args, **kwargs):
    raise RuntimeError("独立检查禁止网络和预训练下载")
socket.create_connection=deny
socket.socket.connect=deny
urllib.request.urlopen=deny
torch.hub.load_state_dict_from_url=deny
torch.hub.download_url_to_file=deny
entry=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("standalone_delivery",entry)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv=[str(entry),*sys.argv[2:]]
raise SystemExit(module.main())
