# MAG-S3c: notes on the confirmatory run

- 10 November 2026, 19:40 UTC: freeze (commit 964800c). The SHA-256 of the 16 frozen files were posted at
  https://t.me/attractordm/79, and the link came at 19:42 UTC (commit 5c3eb15).
- 19:42:43 UTC: «запускай» typed into cell 5 of the frozen notebook (build f994d9b191aa9dbb). The run started at
  19:42:44 UTC on TPU v6e-1 with 44 workers, in parts of 0.5 h as cell 5 sets them.
- 20:12:45 UTC: the pair step paused at the end of the first part as designed. The archive went to Google Drive at
  20:12:46 UTC, and the second part started.
- About 20:30 UTC the runtime was disconnected and deleted on purpose. At a pause the pair step stops every item still
  being computed, and the next part starts that item from zero. The 15 slowest items take about 50 minutes each on
  2000 directions on the Colab host: the direct cells P-same and P-LH of W_C and W_D and the three references at
  dephasing 1e8 s^-1. On the machine of the author they take 0.84-0.94 s per direction, and the host is about 1.7 times
  slower per process. Parts of 0.5 h could therefore never finish them; every other item takes at most about 25
  minutes. The blind check (500 directions, calls of 1 h) and the smoke runs never paused, so this path had not run.
  The work of the second part was lost, and nothing of the run was read.
- Deviation, recorded before the continuation: the run goes on from the archive of the first part with
  notebooks/MAG-S3c_colab_continuation.ipynb. Its cells 1-4 are those of the frozen notebook, and its cell 5a differs
  from cell 5 in two places only: parts of 1.5 h instead of 0.5 h, and a copy of the log of the run to Google Drive
  after each part. The frozen files do not change: cell 1 checks them against the SHA-256 embedded in the notebook,
  and pipeline.sh and mag_s3c.py check them again before every step. Cell 5a asks for «запускай» again and appends a
  line to confirmation.txt on Google Drive.
