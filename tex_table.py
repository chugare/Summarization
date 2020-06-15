
source = """LSTM	0.128 	0.026 	0.085 	0.109 	14.180 	1.931 
LSTM Srp	0.242 	0.036 	0.142 	0.127 	31.370 	2.334 
LSTM Wrp	0.236 	0.032 	0.136 	0.124 	30.830 	2.344 
LSTM Erp	0.220 	0.033 	0.134 	0.134 	27.040 	2.095 

 """


for line in source.split('\n'):
    nums = line.split('\t')
    if len(nums)<1:
        continue
    print('\t'.join(nums))

# for line in source.split('\n'):
#     nums = line.split('\t')
#     if len(nums)<1:
#         continue
#     print(' & '.join(nums)+'\\\\')