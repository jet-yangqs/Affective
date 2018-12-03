import json
import os
right = 0
classes = 5
with open("confusion.json") as f:
     d = json.load(f)
print(sum([d[key] for key in d.keys()]))
for cnt in range(classes):
     right += int(d["%s,%s"%(cnt,cnt)])
print("right=%s"%right)
