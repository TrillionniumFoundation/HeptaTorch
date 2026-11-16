"""Restore exact published native inputs for the observer CI; this is not a build."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, tarfile
P = {
 "355": ("validation/gradient-objects-root/root-cold-qualification-01.tar.xz",
         "07d32eb46016ee8d222475de68117d84eeded0e014d488f7ce1936fbcafb0898", "native/",
         "1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5",
         "tensor-gradient-object-session", "be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b"),
 "428": ("validation/gradient-adam-distributions-root/root-integrated-cold-05.tar.xz",
         "458fab50d7eb4c9685fd400053dd6746225ccae348e75c758d1e0cbd6d8fb6fe", "producer/native/",
         "1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae",
         "tensor-gradient-adam-distributions-session", "6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe"),
}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--profile",choices=P,required=True)
 p.add_argument("--repo",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--report",type=Path,required=True)
 a=p.parse_args()
 repo,output=a.repo.resolve(),a.output.resolve()
 archive,digest,prefix,source_sha,image,image_sha=P[a.profile]
 path=repo/archive
 assert sha(path.read_bytes())==digest, "archive pin"
 output.mkdir(parents=True,exist_ok=False)
 rows=[];seen=set()
 with tarfile.open(path,"r:xz") as t:
  for m in t.getmembers():
   if not m.name.startswith(prefix): continue
   rel=PurePosixPath(m.name[len(prefix):])
   assert not rel.is_absolute() and ".." not in rel.parts, "unsafe archive path"
   if m.isdir(): continue
   assert m.isfile() and str(rel) not in seen and not (m.mode&0o7000), "nonregular or duplicate member"
   seen.add(str(rel))
   target=output/str(rel)
   target.parent.mkdir(parents=True,exist_ok=True)
   raw=t.extractfile(m).read()
   assert len(raw)==m.size
   with target.open("xb") as f:f.write(raw)
   target.chmod(m.mode)
   rows.append({"path":str(rel),"bytes":len(raw),"sha256":sha(raw),"mode":m.mode})
 assert sha((output/"runtime.hepta").read_bytes())==source_sha
 assert sha((output/image).read_bytes())==image_sha
 report={"schema":"heptatorch.observer-native-restoration.v1",
         "status":"EXACT_PUBLISHED_BYTES_RESTORED_NOT_NEW_BUILD","profile":a.profile,
         "archive":archive,"archive_sha256":digest,"members":rows,"native_rebuilt":False,
         "source_sha256":source_sha,"native_sha256":image_sha,"image":str(output/image)}
 a.report.parent.mkdir(parents=True,exist_ok=True)
 a.report.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"profile":a.profile,"members":len(rows)}))
if __name__=="__main__":main()
