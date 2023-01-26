import csv

f=open('/home/hwiric/Internship/LightGCN_MovieLens/u1.base', 'r')
rdr=csv.reader(f)

# 유저-아이템 딕셔너리 생성
d={}
for l in rdr:
    s=''.join(l).split('\t')
    if int(s[2]) < 3:
        continue
    s=s[:2]
    if s[0] in d:
        d[s[0]].append(int(s[1]))
    else:
        d[s[0]] = []

print(d)

f.close()