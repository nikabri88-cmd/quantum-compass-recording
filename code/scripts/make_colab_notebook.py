#!/usr/bin/env python3
"""Build MAG-S3_colab.ipynb: a Colab notebook that carries the frozen MAG-S3 files (tar.gz, base64) with their SHA-256,
installs the pinned packages and runs the environment check, the blind pipeline check and the confirmatory run on
every processor of the runtime. Author's harness.

  python3 -I scripts/make_colab_notebook.py OUT.ipynb
"""
import base64
import hashlib
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("mag_s3.py", "analyze.py", "test_analyze.py", "mag_s2.py", "physics.json", "mag_s2_bird1.json",
         "requirements.txt", "scripts/pipeline.sh", "scripts/env_check.py")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def payload():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for name in FILES:
            info = tar.gettarinfo(os.path.join(HERE, name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(os.path.join(HERE, name), "rb") as fh:
                tar.addfile(info, fh)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [b64[i:i + 100] for i in range(0, len(b64), 100)]


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(keepends=True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=text.strip("\n").splitlines(keepends=True))


HELPERS = r'''
import os, subprocess, sys, time

ROOT, WORK, DRIVE = "/content/mag_s3", "/content/work", "/content/drive/MyDrive/MAG-S3"


def sh(cmd, cwd=ROOT, env=None):
    """Run a shell command, stream its output, return the exit code."""
    e = dict(os.environ, WORKERS=str(os.cpu_count()))
    e.update(env or {})
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    return p.wait()
'''


def build(out):
    frozen = {name: sha(os.path.join(HERE, name)) for name in FILES}
    lines = payload()
    cells = [
        md('''
# MAG-S3 в Google Colab

Блокнот несёт замороженные файлы MAG-S3 (протокол v0.4) и запускает их на всех процессорных ядрах среды.
Среда: «Среда выполнения → Сменить среду выполнения → TPU v5e-1». Сам TPU расчёт не использует, ему нужны 24 ядра
этой машины.

Порядок:
1. Ячейки 1 и 2 – подготовка: файлы со сверкой SHA-256, пакеты закреплённых версий, Google Диск.
2. Ячейка 3 – проверка окружения на калибровочных птицах 1001–1010 и 2001–3000 (около 15 минут).
   Пришли вывод или файл `MAG-S3/env_check.json` с Диска.
3. Ячейка 4 – слепая проверка конвейера на птицах 9301–9302 (около 15 минут). Она печатает только технические
   факты и удаляет свой выход.
4. Ячейка 5 – подтверждающий прогон. Запускать **только после заморозки и команды «запускай»**.

Всё, что нужно сохранить, пишется в папку `MAG-S3` на Google Диске.
'''),
        code(HELPERS + r'''
# 1. Файлы MAG-S3 и их SHA-256
import base64, hashlib, io, json, tarfile

FROZEN = ''' + json.dumps(frozen, indent=1) + r'''

PAYLOAD = """
''' + "\n".join(lines) + r'''
"""

os.makedirs(ROOT, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
    tar.extractall(ROOT, filter="data")
bad = []
for name, want in FROZEN.items():
    got = hashlib.sha256(open(os.path.join(ROOT, name), "rb").read()).hexdigest()
    print(f"{got}  {name}")
    if got != want:
        bad.append(name)
print("\nвсе SHA-256 совпадают с замороженными" if not bad else f"\nРАСХОЖДЕНИЕ: {bad}")
mem = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0]) / 1048576
print(f"ядер: {os.cpu_count()}, память: {mem:.0f} ГБ, Python {sys.version.split()[0]}")
'''),
        code(r'''
# 2. Пакеты закреплённых версий и Google Диск
rc = sh("pip install -q -r requirements.txt 2>&1 | tail -3")
sh("python3 -I -c 'import numpy, scipy, ripser, radicalpy, threadpoolctl; "
   "print(\"numpy\", numpy.__version__, \"scipy\", scipy.__version__)'")
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print("Диск подключён:", DRIVE)
'''),
        code(r'''
# 3. Проверка окружения: smoke, физика, калибровка C3-C7 и контрольные числа на калибровочных птицах
t0 = time.time()
c = f"{WORK}/check"
rc = sh(f"python3 -I mag_s3.py smoke --out {c}")
rc = rc or sh(f"python3 -I mag_s3.py physics --out {c} > {c}/physics.log 2>&1; r=$?; tail -2 {c}/physics.log; exit $r")
rc = rc or sh(f"python3 -I mag_s3.py calibrate --out {c} --workers {os.cpu_count()}")
rc = rc or sh(f"python3 -I scripts/env_check.py {c} > {c}/env_check.out 2>&1; r=$?; tail -40 {c}/env_check.out; exit $r")
if not rc:
    sh(f"cp {c}/env_check.json {c}/calibration.json {DRIVE}/")
print(f"\nкод выхода {rc}, {(time.time() - t0) / 60:.1f} мин; файлы env_check.json и calibration.json в {DRIVE}")
'''),
        code(r'''
# 4. Слепая проверка конвейера на птицах 9301-9302 (выход удаляется, вердикты не печатаются)
t0 = time.time()
rc = sh(f"set -o pipefail; bash scripts/pipeline.sh blind {WORK} 12 2>&1 | tee {DRIVE}/blind_check.log")
print(f"\nкод выхода {rc}, {(time.time() - t0) / 60:.1f} мин; журнал: {DRIVE}/blind_check.log")
'''),
        md('''
## Подтверждающий прогон

Запускать **только после заморозки протокола и кода и команды «запускай»**.
Ячейка считает частями по полчаса и после каждой части кладёт архив `MAG-S3/confirmatory_v04.tar.gz` на Диск.
Если сессия оборвётся, запусти ячейки 1, 2 и 5 снова: счёт продолжится из архива.
'''),
        code(r'''
# 5. Подтверждающий прогон
import shutil
t0 = time.time()
arch = f"{DRIVE}/confirmatory_v04.tar.gz"
if os.path.exists(arch) and not os.path.exists(f"{WORK}/confirmatory_v04"):
    sh(f"tar -xzf {arch} -C {WORK}")
    print("продолжаю из архива на Диске")
while True:
    rc = sh(f"set -o pipefail; bash scripts/pipeline.sh confirmatory {WORK} 0.5 2>&1 | tee -a {WORK}/confirmatory.log")
    sh(f"tar -czf {WORK}/part.tar.gz -C {WORK} confirmatory_v04 && cp {WORK}/part.tar.gz {arch}.tmp && "
       f"mv {arch}.tmp {arch}")
    print(f"архив обновлён, {(time.time() - t0) / 60:.0f} мин с начала")
    if rc != 3:
        break
sh(f"cp {WORK}/confirmatory.log {DRIVE}/; sha256sum {arch}")
print(f"\nкод выхода {rc}; итог в {WORK}/confirmatory_v04/final.log")
sh(f"cat {WORK}/confirmatory_v04/final.log")
'''),
    ]
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
              metadata=dict(colab=dict(name="MAG-S3_colab.ipynb", provenance=[]),
                            kernelspec=dict(name="python3", display_name="Python 3"),
                            language_info=dict(name="python"), accelerator="TPU"))
    with open(out, "w") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    return frozen


if __name__ == "__main__":
    for name, h in build(sys.argv[1]).items():
        print(h, name)
