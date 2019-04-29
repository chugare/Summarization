from DataPipe import DataPipe
from Meta import  Meta
import Main
meta  = Meta( ReadNum= 800000 ).get_meta()
dp = DataPipe(**meta)
# meta = getmeta(**meta)
dp.write_TFRecord(meta,6)
meta = Meta().get_meta()
Main.run_train_task(**meta)