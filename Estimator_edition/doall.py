
import sys
sys.path.append('/home/user/zsm/Summarization')
from  Estimator_edition.ABS import beamsearch as ABSbs
from  Estimator_edition.RNNs2s import beamsearch as s2sbs
from  Estimator_edition.AttentionIsAllYourNeed import beamsearch as tfmbs


date = "209"
size = 100
beam = 5

fmtstr = "{0}-{1}-{2}-{3}"



# ABSbs(beam,size,fmtstr.format(date,"ABS",size,beam))
# s2sbs(beam,size,fmtstr.format(date,"s2slstm",size,beam))
tfmbs(beam,size,fmtstr.format(date,"tfm",size,beam))