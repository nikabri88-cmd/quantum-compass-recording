"""Builds notebooks/MAG-S3c_colab.ipynb (author's harness, not part of the study). The code, the structure file,
requirements.txt and the reference calibration go into the notebook as a base64 tar.gz together with their SHA-256,
so the notebook needs no upload; its first cell refuses to go on when a file differs. The archive is written
deterministically (sorted members, zero times and owners), so the same files always give the same notebook.
The markers of cells 3 and 4 hold the identity of the embedded file set (the SHA-256 of its list of hashes), and a
marker from another build counts as missing. Cell 5 starts the confirmatory run only after the blind check of cell 4
has passed on this build and after the answer «запускай» typed into it; it writes confirmation.json, which
scripts/pipeline.sh requires before it computes anything at the rates of the hypotheses.
The notebook is rebuilt from the frozen files at the freeze, and its own SHA-256 is posted with theirs.
Usage: python3 -I scripts/make_colab_notebook.py CODE_DIR REF_CALIBRATION_DIR OUT.ipynb"""
import base64
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile

CODE, REF, OUT = sys.argv[1:4]
FILES = ["mag_s3c.py", "analyze_s3c.py", "test_s3c.py", "mag_s3b.py", "mag_s3.py", "analyze.py", "mag_s2.py",
         "physics.json", "requirements.txt", "data/6PU0.pdb", "scripts/pipeline.sh", "scripts/env_check.py"]
REF_FILES = ["calibration.json"]


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# the reference calibration must come from the code that goes into the notebook
for f in REF_FILES:
    with open(os.path.join(REF, f)) as fh:
        own = json.load(fh)["metadata"]["own"]
    for code_file in ("mag_s3c.py",):
        if own.get(code_file) != sha(os.path.join(CODE, code_file)):
            sys.exit(f"reference calibration {f} was written by another {code_file}")

members = [(f, os.path.join(CODE, f)) for f in FILES] + [(f"reference_calibration/{f}", os.path.join(REF, f))
                                                         for f in REF_FILES]
members.sort()
hashes = {name: sha(path) for name, path in members}
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
    with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, path in members:
            with open(path, "rb") as fh:
                data = fh.read()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid, info.uname, info.gname = len(data), 0, 0, 0, "", ""
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
payload = base64.b64encode(buf.getvalue()).decode()
lines = [payload[i:i + 100] for i in range(0, len(payload), 100)]
code_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

md0 = """# MAG-S3c в Google Colab

Среда выполнения: **TPU v6e-1** (44 ядра, OpenBLAS SkylakeX), как прогоны MAG-S3 и MAG-S3b. Если её нет, подойдёт
v5e с 24 ядрами; сверка окружения в ячейке 3 запишет разницу в последних знаках.

1. Ячейки 1 и 2 готовят среду: файлы со сверкой SHA-256, пакеты закреплённых версий, Google Диск.
2. Ячейка 3 гонит тесты, повторяет калибровку C1–C6 и сравнивает её число за числом с эталонной калибровкой.
3. Ячейка 4 гонит слепую проверку конвейера на подменённых скоростях и удаляет выход непросмотренным.
4. Ячейка 5 – подтверждающий прогон. Она ждёт ответа «запускай», который вводится вручную после публикации SHA-256,
   и без него не начинается."""

cell1 = f'''# 1. Файлы MAG-S3c и их SHA-256
import base64, hashlib, io, json, os, subprocess, sys, tarfile, time
ROOT, WORK, DRIVE = "/content/mag_s3c", "/content/work", "/content/drive/MyDrive/MAG-S3c"
SHA256 = {json.dumps(hashes, indent=1)}
CODE_ID = hashlib.sha256(json.dumps(SHA256, sort_keys=True).encode()).hexdigest()
assert CODE_ID == "{code_id}"
PAYLOAD = """
{chr(10).join(lines)}
"""
os.makedirs(ROOT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
    tar.extractall(ROOT, filter="data")


def bad_files():
    return [name for name, h in SHA256.items()
            if hashlib.sha256(open(os.path.join(ROOT, name), "rb").read()).hexdigest() != h]


def marker_ok(path):
    """A marker of cells 3 and 4 counts only when it was written for this file set."""
    return os.path.exists(path) and json.load(open(path)).get("code_id") == CODE_ID


def write_marker(path):
    with open(path, "w") as fh:
        json.dump(dict(code_id=CODE_ID, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), fh)


bad = bad_files()
print("все SHA-256 совпадают с вшитыми" if not bad else f"РАСХОЖДЕНИЕ: {{bad}}", "; сборка", CODE_ID[:16])
assert not bad
mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
print(f"ядер: {{os.cpu_count()}}, память: {{mem:.0f}} ГБ, Python {{sys.version.split()[0]}}")


def sh(cmd, cwd=ROOT):
    env = dict(os.environ, WORKERS=str(os.cpu_count()))
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    for line in p.stdout:
        print(line, end="")
    return p.wait()'''

cell2 = '''# 2. Пакеты закреплённых версий и Google Диск
rc = sh("set -o pipefail; pip install -q -r requirements.txt 2>&1 | tail -3")
assert rc == 0, "pip install failed"
sh("python3 -I -c 'import sys, numpy, scipy, radicalpy; print(sys.version.split()[0], numpy.__version__, "
   "scipy.__version__, radicalpy.__version__)'")
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE, exist_ok=True)
print("Диск подключён:", DRIVE)'''

cell3 = '''# 3. Тесты, повтор калибровки C1-C6 и сверка окружения с эталоном (несколько минут на 44 ядрах)
t0 = time.time()
cal = f"{WORK}/calibration"
sh(f"rm -rf {cal}")
os.makedirs(cal, exist_ok=True)
rc = sh(f"set -o pipefail; python3 -I test_s3c.py --fast 2>&1 | tee {cal}/tests.log")
rc = rc or sh(f"python3 -I mag_s3c.py calibrate --out {cal} --workers {os.cpu_count()}")
rc = rc or sh(f"python3 -I scripts/env_check.py reference_calibration {cal} | tee {cal}/env_check.txt; "
              "exit ${PIPESTATUS[0]}")
if rc == 0:
    write_marker(f"{cal}/CELL3_OK")
sh(f"rm -f {cal}/C1_yields.npz; rm -rf {DRIVE}/calibration && cp -r {cal} {DRIVE}/calibration")
print(f"\\nкод выхода {rc}, {(time.time() - t0) / 60:.1f} мин; калибровка и сверка окружения: {DRIVE}/calibration")'''

cell4 = '''# 4. Слепая проверка конвейера на подменённых скоростях; печатает только технические факты и удаляет выход
t0 = time.time()
if not marker_ok(f"{WORK}/calibration/CELL3_OK"):
    sh(f"rm -rf {WORK}/calibration && cp -r {DRIVE}/calibration {WORK}/calibration")
assert marker_ok(f"{WORK}/calibration/CELL3_OK"), "ячейка 3 не прошла на этой сборке: тесты, калибровка или сверка"
sh(f"rm -f {DRIVE}/BLIND_OK")
rc = sh(f"set -o pipefail; bash scripts/pipeline.sh blind {WORK} 1 2>&1 | tee -a {DRIVE}/blind_check.log")
if rc == 0:
    write_marker(f"{DRIVE}/BLIND_OK")
print(f"\\nкод выхода {rc} (3 – пауза по времени, запусти ячейку снова), {(time.time() - t0) / 60:.1f} мин; "
      f"журнал: {DRIVE}/blind_check.log")'''

md5 = """Ячейка 5 – подтверждающий прогон. Сначала она сверяет файлы с вшитыми SHA-256 и отметку слепой проверки этой
сборки, потом спрашивает подтверждение: ответ «запускай» вводится вручную после заморозки и публикации SHA-256 в
Telegram, а любой другой ответ останавливает ячейку до начала счёта. Время ответа записывается в
`MAG-S3c/confirmation.txt` на Диске. Прогон идёт частями по полчаса и после каждой части кладёт архив
`MAG-S3c/confirmatory_s3c.tar.gz` на Диск; после обрыва сессии достаточно выполнить ячейки 1 и 2 и снова запустить
эту ячейку с тем же ответом."""

cell5 = '''# 5. Подтверждающий прогон MAG-S3c, только после ответа «запускай»
assert not bad_files(), "файлы отличаются от вшитых SHA-256"
assert marker_ok(f"{DRIVE}/BLIND_OK"), "слепая проверка ячейки 4 не прошла на этой сборке или не запускалась"
answer = input("SHA-256 опубликованы? Для старта введи «запускай»: ").strip().strip("«»\\"'")
if answer != "запускай":
    raise SystemExit("подтверждения нет, прогон не начат")
t0 = time.time()
arch = f"{DRIVE}/confirmatory_s3c.tar.gz"
if os.path.exists(arch) and not os.path.exists(f"{WORK}/confirmatory_s3c"):
    sh(f"tar -xzf {arch} -C {WORK}")
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(f"{WORK}/confirmation.json", "w") as fh:
    json.dump(dict(answer=answer, time_utc=stamp, code_id=CODE_ID, sha256=SHA256), fh, ensure_ascii=False, indent=1)
with open(f"{DRIVE}/confirmation.txt", "a") as fh:
    fh.write(f"{stamp} confirmation «{answer}», build {CODE_ID}\\n")
print("подтверждение принято", stamp)
while True:
    rc = sh(f"set -o pipefail; bash scripts/pipeline.sh confirmatory {WORK} 0.5 2>&1 | tee -a {WORK}/confirmatory.log")
    sh(f"tar -czf {WORK}/part.tar.gz -C {WORK} confirmatory_s3c confirmation.json && cp {WORK}/part.tar.gz {arch}.tmp && "
       f"mv {arch}.tmp {arch}")
    if rc != 3:
        break
sh(f"cp {WORK}/confirmatory.log {DRIVE}/; sha256sum {arch}")
print(f"\\nкод выхода {rc}, {(time.time() - t0) / 60:.1f} мин; архив: {arch}")'''


def code(src):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=[ln + "\n" for ln in src.split("\n")[:-1]] + [src.split("\n")[-1]])


def md(src):
    return dict(cell_type="markdown", metadata={}, source=[ln + "\n" for ln in src.split("\n")[:-1]] + [src.split("\n")[-1]])


nb = dict(nbformat=4, nbformat_minor=5, cells=[md(md0), code(cell1), code(cell2), code(cell3), code(cell4), md(md5),
                                                code(cell5)],
          metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                        language_info=dict(name="python"), colab=dict(provenance=[]), accelerator="TPU"))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)
print(OUT, sha(OUT), f"{os.path.getsize(OUT) / 1024:.0f} KB, {len(hashes)} files, build {code_id[:16]}")
