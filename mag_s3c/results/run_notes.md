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
- 11 November 2026, a new runtime of the same kind (TPU v6e-1, 44 processors). Cells 1-3 ran there. Cell 3 is not
  needed for the run and repeated the 30 fast tests and the comparison of the environment with the reference
  calibration of round 3: 192 identical values, 31 equal within the tolerances, none different and none missing
  (Python 3.13.18, OpenBLAS SkylakeX); its marker was written at 16:54:07 UTC.
- 16:54:15 UTC: cell 4 was started on this runtime by mistake and stopped in the notebook, while its process went on
  in the background; at 16:56:08 UTC a second cell 4 started beside it. The first action of cell 4 removes the marker
  of the passed blind check from Google Drive, so cell 5a refused to start. Both processes were stopped in the
  terminal, their output was deleted unread ("check output deleted (exit 1)"), and the folder of the check was removed
  by hand. No marker was written by hand.
- 17:07:39 UTC: a clean third blind check of build f994d9b on this runtime. Pair 17:33:22, composite pair 17:42:00,
  checks 17:44:23, curves 17:48:11, receptor 17:53:46, analysis 17:53:47 UTC; all eleven invariants held, I11 stood
  at 2.5287541332023138e-15 and 2.4932082761588624e-15 of the odd part of the molecule (5.194759962849077e-12 in the
  normalization of the first check), bit for bit the values of 10 November on the first runtime. The output was
  deleted unread, and the marker of the passed check was written at 17:53:47 UTC.
- 18:08:50 UTC: «запускай» typed into cell 5a (second line of confirmation.txt). The pipeline started at 18:08:51 UTC
  with 44 workers from the archive of the first part of 10 November.
- 19:39:03 UTC: the pair step paused at the limit of 1.5 h; the archive and the log went to Google Drive, and the next
  part started at 19:39:07 UTC.
- Steps of the run, all with exit code 0: pair 19:46:10, composite pair 20:32:53, checks 20:35:53, curves 20:39:31,
  receptor 20:58:59, analysis 20:59:01 UTC; 574 cells, 54 references, 24 cells of the composite pair. The run took
  30 minutes on 10 November and 2 h 50 min on 11 November. The verdicts go to the log of the analysis step inside the
  archive; the notebook showed only the exit codes of the steps.
- 20:59:20 UTC: the archive of the run, 48,804,217 bytes, on Google Drive. Nobody read the output in Colab; the
  notebook showed the exit codes of the steps and the counts of the files.
- 21:01–21:12 UTC: the archive was split into parts on Google Drive (first parts of 7 MB, too large for the
  connector, then 25 parts of 2 MB) and joined on the calibration machine; SHA-256
  99e6288ba6d583181cb2e18fbf8df5bd00dbd1fce5c8568fbe68dda6f9c8446a, gzip check passed.
- About 21:14–21:20 UTC: checks of provenance, which printed the technical facts and the verdicts of the
  expectations, and the reanalysis with the frozen analyze_s3c.py: every value of results.json came out identical
  (checks/reanalysis_of_the_archive.txt). Then the verdicts of the hypotheses were read, and the sealed file with
  the line of the trial run was opened; its SHA-256 equals the one in FREEZE.txt.
