import json
import codecs
import uni2tex as uni
import numpy as np
import unicodedata

mapping = json.load(open('raga_name_mapping.json','r'))
lines = open('carnatic_ragas_uuids.txt', 'r').readlines()
temp = []
for l in lines:
    temp.append(l.strip())
temp = np.array(temp)
temp = np.unique(temp)
import codecs

fid = codecs.open('carnatic_ragas_names.txt', 'w', encoding='utf-8')
for l in temp:
    fid.write("%s\n"%mapping[l])
fid.close()

fid = codecs.open('carnatic_ragas_glossary.txt', 'w', encoding='utf-8')
for l in temp:
    fid.write("\\newglossaryentry{%s}\n{\n name={%s},\ndescription={},\n sort=%s,\ntype=ragaCMD\n}\n\n"%(unicodedata.normalize('NFD', mapping[l]).encode('ascii', 'ignore').lower().replace(' ', '_'), uni.uni2tex(mapping[l]),unicodedata.normalize('NFD', mapping[l]).encode('ascii', 'ignore').lower().replace(' ', '_')))
fid.close()


lines = open('hindustani_ragas_uuids.txt', 'r').readlines()
temp = []
for l in lines:
    temp.append(l.strip())
temp = np.array(temp)
temp = np.unique(temp)

fid = codecs.open('hindustani_ragas_names.txt', 'w', encoding='utf-8')
for l in temp:
    fid.write("%s\n"%mapping[l])
fid.close()

fid = codecs.open('hindustani_ragas_glossary.txt', 'w', encoding='utf-8')
for l in temp:
    fid.write("\\newglossaryentry{%s}\n{\n name={%s},\ndescription={},\n sort=%s,\ntype=ragaHMD\n}\n\n"%(unicodedata.normalize('NFD', mapping[l]).encode('ascii', 'ignore').lower().replace(' ', '_'), uni.uni2tex(mapping[l]),unicodedata.normalize('NFD', mapping[l]).encode('ascii', 'ignore').lower().replace(' ', '_')))
fid.close()


