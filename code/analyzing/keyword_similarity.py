# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:37:29 2022

@author: yebin
"""

import pandas as pd

pr1 = ['system', 'method', 'device', 'patient', 'information', 'user', 'sensor', 'computer', 'communication', 'monitoring', 'time', 'image']
pr2 = ['system', 'device', 'method', 'patient', 'information', 'time', 'user', 'processor', 'sensor', 'interface', 'location', 'monitor', 'instruction', 'memory', 'display', 'computer']

pe1 = ['image', 'method', 'system', 'device', 'information', 'processing', 'patient', 'plurality', 'display']
pe2 = ['unit', 'information', 'method', 'system', 'application', 'measurement', 'processing', 'apparatus', 'time', 'device', 'display', 'user', 'health', 'program', 'part', 'request', 'response', 'server', 'computer', 'collection', 'parameter', 'care', 'classification', 'period', 'acquisition', 'communication', 'process']
pe3 = ['sensor', 'method', 'application', 'system', 'patient', 'device', 'apparatus', 'computer', 'monitoring', 'plurality', 'time', 'condition', 'record', 'user', 'unit', 'embodiment', 'information', 'parameter', 'processor', 'control', 'function', 'measurement']
pe4 = ['information', 'system', 'method', 'user', 'device', 'apparatus', 'computer', 'embodiment', 'interface', 'identification', 'plurality', 'processing', 'processor', 'program', 'medium', 'unit']
pe5 = ['sensor', 'device', 'system', 'method', 'event', 'information', 'monitoring', 'patient', 'user', 'condition', 'apparatus', 'example', 'drug', 'communication', 'health', 'plurality']
pe6 = ['medication', 'information', 'system', 'method', 'device', 'patient', 'container', 'prescription', 'computer', 'access', 'identification', 'plurality', 'storage', 'user', 'time', 'process', 'machine', 'monitoring', 'image', 'apparatus']
pe7 = ['event', 'method', 'system', 'patient', 'information', 'device', 'risk', 'time', 'treatment', 'plurality', 'monitoring', 'embodiment', 'detection']
pe8 = ['method', 'patient', 'system', 'blood', 'model', 'event', 'image', 'computer', 'flow', 'measurement', 'vessel', 'device', 'information', 'response', 'treatment', 'plurality', 'interest', 'measure', 'level']
pe9 = ['patient', 'risk', 'method', 'system', 'model', 'health', 'plurality', 'care', 'condition', 'event', 'level', 'management', 'record', 'machine', 'disease', 'monitoring', 'score', 'determine', 'time', 'prediction', 'information', 'computer', 'assessment']
pe10 = ['risk', 'treatment', 'method', 'patient', 'system', 'assessment', 'blood', 'device', 'level', 'monitoring', 'term', 'condition', 'invention', 'assistance', 'control', 'platform', 'recommendation', 'technique', 'time', 'decision', 'probability', 'result', 'embodiment', 'marker', 'sample', 'use', 'combination', 'diagnosis']

union11 = list(set().union(pr1, pe1))
inter11 = list(set(pr1).intersection(pe1)) 
sim11 = len(inter11)/len(union11)
union21 = list(set().union(pr2, pe1))
inter21 = list(set(pr2).intersection(pe1)) 
sim21 = len(inter21)/len(union21)
print("유망1과 주변1 유사도: {}, 유망2와 주변1 유사도: {}".format(sim11, sim21))

union12 = list(set().union(pr1, pe2))
inter12 = list(set(pr1).intersection(pe2)) 
sim12 = len(inter12)/len(union12)
union22 = list(set().union(pr2, pe2))
inter22 = list(set(pr2).intersection(pe2)) 
sim22 = len(inter22)/len(union22)
print("유망1과 주변2 유사도: {}, 유망2와 주변2 유사도: {}".format(sim12, sim22))

union13 = list(set().union(pr1, pe3))
inter13 = list(set(pr1).intersection(pe3)) 
sim13 = len(inter13)/len(union13)
union23 = list(set().union(pr2, pe3))
inter23 = list(set(pr2).intersection(pe3)) 
sim23 = len(inter23)/len(union23)
print("유망1과 주변3 유사도: {}, 유망2와 주변3 유사도: {}".format(sim13, sim23))

union14 = list(set().union(pr1, pe4))
inter14 = list(set(pr1).intersection(pe4)) 
sim14 = len(inter14)/len(union14)
union24 = list(set().union(pr2, pe4))
inter24 = list(set(pr2).intersection(pe4)) 
sim24 = len(inter24)/len(union24)
print("유망1과 주변4 유사도: {}, 유망2와 주변4 유사도: {}".format(sim14, sim24))

union15 = list(set().union(pr1, pe5))
inter15 = list(set(pr1).intersection(pe5)) 
sim15 = len(inter15)/len(union15)
union25 = list(set().union(pr2, pe5))
inter25 = list(set(pr2).intersection(pe5)) 
sim25 = len(inter25)/len(union25)
print("유망1과 주변5 유사도: {}, 유망2와 주변5 유사도: {}".format(sim15, sim25))

union16 = list(set().union(pr1, pe6))
inter16 = list(set(pr1).intersection(pe6)) 
sim16 = len(inter16)/len(union16)
union26 = list(set().union(pr2, pe6))
inter26 = list(set(pr2).intersection(pe6)) 
sim26 = len(inter26)/len(union26)
print("유망1과 주변6 유사도: {}, 유망2와 주변6 유사도: {}".format(sim16, sim26))

union17 = list(set().union(pr1, pe7))
inter17 = list(set(pr1).intersection(pe7)) 
sim17 = len(inter17)/len(union17)
union27 = list(set().union(pr2, pe7))
inter27 = list(set(pr2).intersection(pe7)) 
sim27 = len(inter27)/len(union27)
print("유망1과 주변7 유사도: {}, 유망2와 주변7 유사도: {}".format(sim17, sim27))

union18 = list(set().union(pr1, pe8))
inter18 = list(set(pr1).intersection(pe8)) 
sim18 = len(inter18)/len(union18)
union28 = list(set().union(pr2, pe8))
inter28 = list(set(pr2).intersection(pe8)) 
sim28 = len(inter28)/len(union28)
print("유망1과 주변8 유사도: {}, 유망2와 주변8 유사도: {}".format(sim18, sim28))

union19 = list(set().union(pr1, pe9))
inter19 = list(set(pr1).intersection(pe9)) 
sim19 = len(inter19)/len(union19)
union29 = list(set().union(pr2, pe9))
inter29 = list(set(pr2).intersection(pe9)) 
sim29 = len(inter29)/len(union29)
print("유망1과 주변9 유사도: {}, 유망2와 주변9 유사도: {}".format(sim19, sim29))

union110 = list(set().union(pr1, pe10))
inter110 = list(set(pr1).intersection(pe10)) 
sim110 = len(inter110)/len(union110)
union210 = list(set().union(pr2, pe10))
inter210 = list(set(pr2).intersection(pe10)) 
sim210 = len(inter210)/len(union210)
print("유망1과 주변10 유사도: {}, 유망2와 주변10 유사도: {}".format(sim110, sim210))


#%%
vacant = pd.read_csv('C:/Users/yebin/OneDrive/바탕 화면/졸업연구/3차_2205_2/데이터/textvector/vacant_text.csv')

va1_t = vacant['text'][0] + ' ' + vacant['text'][2]
va2_t = vacant['text'][1]
va3_t = vacant['text'][3] + ' ' + vacant['text'][4]
va4_t = vacant['text'][5]
va5_t = vacant['text'][6] + ' ' + vacant['text'][7] + ' ' + vacant['text'][8] + ' ' + vacant['text'][9] + ' ' + vacant['text'][11] + ' ' + vacant['text'][12] + ' ' + vacant['text'][13] + ' ' + vacant['text'][15] + ' ' + vacant['text'][16] + ' ' + vacant['text'][17] + ' ' + vacant['text'][18]
va6_t = vacant['text'][10] + ' ' + vacant['text'][14]
va7_t = vacant['text'][19] + ' ' + vacant['text'][22] + ' ' + vacant['text'][23] + ' ' + vacant['text'][24] + ' ' + vacant['text'][28] + ' ' + vacant['text'][29] + ' ' + vacant['text'][31]
va8_t = vacant['text'][20] + ' ' + vacant['text'][21] + ' ' + vacant['text'][25] + ' ' + vacant['text'][26] + ' ' + vacant['text'][27] + ' ' + vacant['text'][30]
va9_t = vacant['text'][32] + ' ' + vacant['text'][33]
va10_t = vacant['text'][34]

va1 = va1_t.split(' ')
va1 = set(va1)
va1 = list(va1) 

va2 = va2_t.split(' ')
va2 = set(va2)
va2 = list(va2) 

va2 = va2_t.split(' ')
va2 = set(va2)
va2 = list(va2) 

va3 = va3_t.split(' ')
va3 = set(va3)
va3 = list(va3) 

va4 = va4_t.split(' ')
va4 = set(va4)
va4 = list(va4) 

va5 = va5_t.split(' ')
va5 = set(va5)
va5 = list(va5) 

va6 = va6_t.split(' ')
va6 = set(va6)
va6 = list(va6) 

va7 = va7_t.split(' ')
va7 = set(va7)
va7 = list(va7) 

va8 = va8_t.split(' ')
va8 = set(va8)
va8 = list(va8) 

va9 = va9_t.split(' ')
va9 = set(va9)
va9 = list(va9) 

va10 = va10_t.split(' ')
va10 = set(va10)
va10 = list(va10) 

va1.sort()
va2.sort()
va3.sort()
va4.sort()
va5.sort()
va6.sort()
va7.sort()
va8.sort()
va9.sort()
va10.sort()

list(zip(va1,va2))

complement = list(set(va5) - set(va3))
#%%1
un_v_11 = list(set().union(pr1, va1))
in_v_11 = list(set(pr1).intersection(va1)) 
sim_v_11 = len(in_v_11)/len(un_v_11)
un_v_21 = list(set().union(pr2, va1))
in_v_21 = list(set(pr2).intersection(va1)) 
sim_v_21 = len(in_v_21)/len(un_v_21)
un_v_31 = list(set().union(pe1, va1))
in_v_31 = list(set(pe1).intersection(va1)) 
sim_v_31 = len(in_v_31)/len(un_v_31)

print("유망1과 공백1 유사도: {}, 유망2와 공백1 유사도: {}, 주변1과 공백1 유사도: {}".format(sim_v_11, sim_v_21, sim_v_31))

#%%2
un_v_12 = list(set().union(pr1, va2))
in_v_12 = list(set(pr1).intersection(va2)) 
sim_v_12 = len(in_v_12)/len(un_v_12)
un_v_22 = list(set().union(pr2, va2))
in_v_22 = list(set(pr2).intersection(va2)) 
sim_v_22 = len(in_v_22)/len(un_v_22)
un_v_32 = list(set().union(pe2, va2))
in_v_32 = list(set(pe2).intersection(va2)) 
sim_v_32 = len(in_v_32)/len(un_v_32)

print("유망1과 공백2 유사도: {}, 유망2와 공백2 유사도: {}, 주변2과 공백2 유사도: {}".format(sim_v_12, sim_v_22, sim_v_32))
#%%3
un_v_13 = list(set().union(pr1, va3))
in_v_13 = list(set(pr1).intersection(va3)) 
sim_v_13 = len(in_v_13)/len(un_v_13)
un_v_23 = list(set().union(pr2, va3))
in_v_23 = list(set(pr2).intersection(va3)) 
sim_v_23 = len(in_v_23)/len(un_v_23)
un_v_33 = list(set().union(pe3, va3))
in_v_33 = list(set(pe3).intersection(va3)) 
sim_v_33 = len(in_v_33)/len(un_v_33)

print("유망1과 공백3 유사도: {}, 유망2와 공백3 유사도: {}, 주변3과 공백3 유사도: {}".format(sim_v_13, sim_v_23, sim_v_33))
#%%4
un_v_14 = list(set().union(pr1, va4))
in_v_14 = list(set(pr1).intersection(va4)) 
sim_v_14 = len(in_v_14)/len(un_v_14)
un_v_24 = list(set().union(pr2, va4))
in_v_24 = list(set(pr2).intersection(va4)) 
sim_v_24 = len(in_v_24)/len(un_v_24)
un_v_34 = list(set().union(pe4, va4))
in_v_34 = list(set(pe4).intersection(va4)) 
sim_v_34 = len(in_v_34)/len(un_v_34)

print("유망1과 공백4 유사도: {}, 유망2와 공백4 유사도: {}, 주변4과 공백4 유사도: {}".format(sim_v_14, sim_v_24, sim_v_34))

#%%5
un_v_15 = list(set().union(pr1, va5))
in_v_15 = list(set(pr1).intersection(va5)) 
sim_v_15 = len(in_v_15)/len(un_v_15)
un_v_25 = list(set().union(pr2, va5))
in_v_25 = list(set(pr2).intersection(va5)) 
sim_v_25 = len(in_v_25)/len(un_v_25)
un_v_35 = list(set().union(pe5, va5))
in_v_35 = list(set(pe5).intersection(va5)) 
sim_v_35 = len(in_v_35)/len(un_v_35)

print("유망1과 공백5 유사도: {}, 유망2와 공백5 유사도: {}, 주변5과 공백5 유사도: {}".format(sim_v_15, sim_v_25, sim_v_35))
#%%6
un_v_16 = list(set().union(pr1, va6))
in_v_16 = list(set(pr1).intersection(va6)) 
sim_v_16 = len(in_v_16)/len(un_v_16)
un_v_26 = list(set().union(pr2, va6))
in_v_26 = list(set(pr2).intersection(va6)) 
sim_v_26 = len(in_v_26)/len(un_v_26)
un_v_36 = list(set().union(pe6, va6))
in_v_36 = list(set(pe6).intersection(va6)) 
sim_v_36 = len(in_v_36)/len(un_v_36)

print("유망1과 공백6 유사도: {}, 유망2와 공백6 유사도: {}, 주변6과 공백6 유사도: {}".format(sim_v_16, sim_v_26, sim_v_36))

#%%7
un_v_17 = list(set().union(pr1, va7))
in_v_17 = list(set(pr1).intersection(va7)) 
sim_v_17 = len(in_v_17)/len(un_v_17)
un_v_27 = list(set().union(pr2, va7))
in_v_27 = list(set(pr2).intersection(va7)) 
sim_v_27 = len(in_v_27)/len(un_v_27)
un_v_37 = list(set().union(pe7, va7))
in_v_37 = list(set(pe7).intersection(va7)) 
sim_v_37 = len(in_v_37)/len(un_v_37)

print("유망1과 공백7 유사도: {}, 유망2와 공백7 유사도: {}, 주변7과 공백7 유사도: {}".format(sim_v_17, sim_v_27, sim_v_37))
#%%8
un_v_18 = list(set().union(pr1, va8))
in_v_18 = list(set(pr1).intersection(va8)) 
sim_v_18 = len(in_v_18)/len(un_v_18)
un_v_28 = list(set().union(pr2, va8))
in_v_28 = list(set(pr2).intersection(va8)) 
sim_v_28 = len(in_v_28)/len(un_v_28)
un_v_38 = list(set().union(pe8, va8))
in_v_38 = list(set(pe8).intersection(va8)) 
sim_v_38 = len(in_v_38)/len(un_v_38)

print("유망1과 공백8 유사도: {}, 유망2와 공백8 유사도: {}, 주변8과 공백8 유사도: {}".format(sim_v_18, sim_v_28, sim_v_38))

#%%9
un_v_19 = list(set().union(pr1, va9))
in_v_19 = list(set(pr1).intersection(va9)) 
sim_v_19 = len(in_v_19)/len(un_v_19)
un_v_29 = list(set().union(pr2, va9))
in_v_29 = list(set(pr2).intersection(va9)) 
sim_v_29 = len(in_v_29)/len(un_v_29)
un_v_39 = list(set().union(pe9, va9))
in_v_39 = list(set(pe9).intersection(va9)) 
sim_v_39 = len(in_v_39)/len(un_v_39)

print("유망1과 공백9 유사도: {}, 유망2와 공백9 유사도: {}, 주변9과 공백9 유사도: {}".format(sim_v_19, sim_v_29, sim_v_39))

#%%10
un_v_110 = list(set().union(pr1, va10))
in_v_110 = list(set(pr1).intersection(va10)) 
sim_v_110 = len(in_v_110)/len(un_v_110)
un_v_210 = list(set().union(pr2, va10))
in_v_210 = list(set(pr2).intersection(va10)) 
sim_v_210 = len(in_v_210)/len(un_v_210)
un_v_310 = list(set().union(pe10, va10))
in_v_310 = list(set(pe10).intersection(va10)) 
sim_v_310 = len(in_v_310)/len(un_v_310)

print("유망1과 공백10 유사도: {}, 유망2와 공백10 유사도: {}, 주변1과 공백10 유사도: {}".format(sim_v_110, sim_v_210, sim_v_310))
