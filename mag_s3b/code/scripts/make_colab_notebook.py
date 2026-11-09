"""Builds notebooks/MAG-S3b_colab.ipynb (author's harness, not part of the study). The code, requirements.txt and the
reference calibration go into the notebook as a base64 tar.gz together with their SHA-256, so the notebook needs no
upload; its first cell refuses to go on when a file differs. The notebook is rebuilt from the frozen files at the
freeze, and its own SHA-256 is posted with theirs.
Usage: python3 -I scripts/make_colab_notebook.py CODE_DIR REF_CALIBRATION_DIR OUT.ipynb"""
import base64
import hashlib
import io
import json
import os
import sys
import tarfile

CODE, REF, OUT = sys.argv[1:4]
FILES = ["mag_s3b.py", "estimate.py", "analyze_s3b.py", "test_s3b.py", "mag_s3.py", "analyze.py", "mag_s2.py",
         "physics.json", "mag_s2_bird1.json", "requirements.txt", "scripts/pipeline.sh", "scripts/env_check.py",
         "scripts/mutate_estimate.py"]
REF_FILES = ["C1.json", "C2_C4.json", "C3.json", "C5.json"]


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# the reference calibration must come from the code that goes into the notebook
for f in REF_FILES:
    with open(os.path.join(REF, f)) as fh:
        own = json.load(fh)["metadata"]["own"]
    for code_file in ("mag_s3b.py", "estimate.py"):
        if own.get(code_file) != sha(os.path.join(CODE, code_file)):
            sys.exit(f"reference calibration {f} was written by another {code_file}")

buf, hashes = io.BytesIO(), {}
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for f in FILES:
        tar.add(os.path.join(CODE, f), arcname=f)
        hashes[f] = sha(os.path.join(CODE, f))
    for f in REF_FILES:
        tar.add(os.path.join(REF, f), arcname=f"reference_calibration/{f}")
        hashes[f"reference_calibration/{f}"] = sha(os.path.join(REF, f))
payload = base64.b64encode(buf.getvalue()).decode()
lines = [payload[i:i + 100] for i in range(0, len(payload), 100)]

md0 = """# MAG-S3b в Google Colab

Среда выполнения: **TPU v6e-1** (44 ядра, OpenBLAS SkylakeX), как подтверждающий прогон MAG-S3. Если её нет, подойдёт
v5e с 24 ядрами; сверка окружения в ячейке 3 запишет разницу в последних знаках.

1. Ячейки 1 и 2 готовят среду: файлы со сверкой SHA-256, пакеты закреплённых версий, Google Диск.
2. Ячейка 3 повторяет калибровку C1–C5 и сравнивает её число за числом с эталонной калибровкой.
3. Ячейка 4 гонит слепую проверку конвейера на птицах 9401–9402 и удаляет выход непросмотренным.
4. Ячейка 5 – подтверждающий прогон, только после команды «запускай»."""

cell1 = f'''# 1. Файлы MAG-S3b и их SHA-256
import base64, hashlib, io, json, os, subprocess, sys, tarfile, time
ROOT, WORK, DRIVE = "/content/mag_s3b", "/content/work", "/content/drive/MyDrive/MAG-S3b"
SHA256 = {json.dumps(hashes, indent=1)}
PAYLOAD = """
{chr(10).join(lines)}
"""
os.makedirs(ROOT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
    tar.extractall(ROOT, filter="data")
bad = [name for name, h in SHA256.items()
       if hashlib.sha256(open(os.path.join(ROOT, name), "rb").read()).hexdigest() != h]
print("все SHA-256 совпадают с вшитыми" if not bad else f"РАСХОЖДЕНИЕ: {{bad}}")
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

cell3 = '''# 3. Тесты, повтор калибровки C1-C5 и сверка окружения с эталоном (несколько минут на 44 ядрах)
t0 = time.time()
cal = f"{WORK}/calibration"
os.makedirs(cal, exist_ok=True)
rc = sh("python3 -I test_s3b.py --fast")
rc = rc or sh(f"python3 -I mag_s3b.py calibrate --out {cal} --workers {os.cpu_count()}")
rc = rc or sh(f"python3 -I scripts/env_check.py reference_calibration {cal} | tee {cal}/env_check.txt; "
              "exit ${PIPESTATUS[0]}")
if rc == 0:
    open(f"{cal}/CELL3_OK", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
sh(f"rm -rf {DRIVE}/calibration && cp -r {cal} {DRIVE}/calibration")
print(f"\\nкод выхода {rc}, {(time.time() - t0) / 60:.1f} мин; калибровка и сверка окружения: {DRIVE}/calibration")'''

cell4 = '''# 4. Слепая проверка конвейера на птицах 9401-9402; печатает только технические факты и удаляет выход
t0 = time.time()
if not os.path.exists(f"{WORK}/calibration/C1.json"):
    sh(f"cp -r {DRIVE}/calibration {WORK}/calibration")
assert os.path.exists(f"{WORK}/calibration/CELL3_OK"), "ячейка 3 не прошла: тесты, калибровка или сверка окружения"
rc = sh(f"set -o pipefail; bash scripts/pipeline.sh blind {WORK} {WORK}/calibration 1 2>&1 | tee {DRIVE}/blind_check.log")
print(f"\\nкод выхода {rc} (3 – пауза по времени, запусти ячейку снова), {(time.time() - t0) / 60:.1f} мин; "
      f"журнал: {DRIVE}/blind_check.log")'''

md5 = """Ячейка 5 запускается только после команды «запускай» и публикации SHA-256 в Telegram. Она считает частями по
полчаса и после каждой части кладёт архив `MAG-S3b/confirmatory_s3b.tar.gz` на Диск; после обрыва сессии достаточно
выполнить ячейки 1 и 2 и снова запустить её."""

cell5 = '''# 5. Подтверждающий прогон MAG-S3b
t0 = time.time()
arch = f"{DRIVE}/confirmatory_s3b.tar.gz"
if not os.path.exists(f"{WORK}/calibration/C1.json"):
    sh(f"cp -r {DRIVE}/calibration {WORK}/calibration")
assert os.path.exists(f"{WORK}/calibration/CELL3_OK"), "ячейка 3 не прошла: тесты, калибровка или сверка окружения"
if os.path.exists(arch) and not os.path.exists(f"{WORK}/confirmatory_s3b"):
    sh(f"tar -xzf {arch} -C {WORK}")
while True:
    rc = sh(f"set -o pipefail; bash scripts/pipeline.sh confirmatory {WORK} {WORK}/calibration 0.5 2>&1 "
            f"| tee -a {WORK}/confirmatory.log")
    sh(f"tar -czf {WORK}/part.tar.gz -C {WORK} confirmatory_s3b && cp {WORK}/part.tar.gz {arch}.tmp && "
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
print(OUT, sha(OUT), f"{os.path.getsize(OUT) / 1024:.0f} KB, {len(hashes)} files")
