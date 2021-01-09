import ezai_util
import navsim
from ezai_util import DictObj

conf = DictObj().load_from_json_file("navsim_conf.json")

trainer = navsim.Trainer(run_id='navsim_demo',run_resume=False,conf=conf)

trainer.train()

trainer.env_close()
trainer.files_close()