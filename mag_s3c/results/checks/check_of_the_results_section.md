# MAG-S3c: independent check of the results section and the README results

## What I checked and how

- **Texts.** I checked `protocol/MAG-S3c_protocol_v0.4_with_results_ru.md`, from «## Результаты подтверждающего прогона» to «## Источники» (called RU below). In `README.md` I checked "Results of the confirmatory run", "What it means", "Deviations", "Literature the protocol missed" and "Contents" (called README below). Both files are unchanged since the draft commit 4fdd4e2 (file times 21:39 UTC; later commits touch only results/exploratory), and the working tree is clean.
- **Ground truth.** I used `results.json` and `maps_axes.npz` from the extracted archive, the run folder (`receptor.json`, `maps_E5.npz`, cells, refs, composite), the four tables, `confirmation.json`, the logs, FREEZE.txt and the git history. I read everything with `python3 -I` from my own scripts (not published), and I imported nothing from the extract folder.
- **Field definitions.** I took the definitions from `analyze_s3c.py`: `eps_mol` is the odd half-range divided by the even range (`mag_s3b.parity_parts`); `E6.F_amp_over_F` is F_amp/F at p = 1, S = 1; E5 quantiles use `nanquantile`, so axes beyond the scale are dropped; `E4.eps_x` uses the ratio transfer (`R_REN·d_x/zonal even range`).
- **Rounding.** I rounded every value to the precision the text uses. RU uses decimal commas.
- **Integrity checks (all hold).**
  - The text above the results heading and the «Источники» section are byte-identical to the frozen `MAG-S3c_protocol_v0.4_ru.md`. Its SHA-256 is 20353b73…, as in FREEZE.txt.
  - The archive has 48 804 217 bytes and SHA-256 99e6288b…. It holds 1149 files in 8 folders.
  - `confirmation.json` holds «запускай» at 2026-11-11T18:08:50Z. Its 14 SHA-256 values equal FREEZE.txt, and the blind-check marker is f994d9b… at 17:53:47Z.
  - All 574 cells, 54 references and 24 composite cells carry mag_s3c.py 2ccbb82e…. The metadata of pair, composite, checks, curves, receptor and analyze hold 13 code hashes equal to FREEZE.txt, one BLAS thread, and Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1 and RadicalPy 1.0.9.
  - The reanalysis reproduces `results.json` with 0 differences in 174 753 values outside metadata. `maps_axes.npz` has identical arrays, and `cells.csv` is identical.
  - The sealed file has SHA-256 3ea921e0…. Run notes confirm the deviation was committed in 8208c9a at 15:48:21 UTC on 11 Nov, before the continuation.

---

## A. Real errors

### A1. Numbers that do not match the data

**A1.1. E6 medians (RU, «Разведочные карты», E6).**
- Text: «Отношение F\_amp/F формы K при m = 1 имеет медиану 1,5 в S2, 0,10 в W\_C и 0,05 в W\_D».
- Problem: S2 is correct, but the W\_C and W\_D values are not reproducible.
- Evidence: the median of `maps.E6[cell].F_amp_over_F` over form K cells with m = 1 at Cry4a rates:

| Geometry | Cells | Median of F_amp/F |
| --- | --- | --- |
| S2 | 40 | 1.497 |
| W\_C | 64 | 0.0712 |
| W\_D | 64 | 0.0437 |

- Other subsets I tried:
  - Main grid only gives W\_C 0.070 and W\_D 0.045.
  - Adding the E12 cells gives W\_C 0.058.
  - Old levels only give W\_C 0.103, but then W\_D is 0.041 and S2 is 1.93.
  - No single set gives 1,5, 0,10 and 0,05 together.
- Fix: «медиану 1,5 в S2, 0,07 в W\_C и 0,04 в W\_D». The conclusion that the amplitude transfer decides in almost all cells stays true: |F\_amp| < |F| in 397 of 398 W\_C cells and in 320 of 320 W\_D cells.

**A1.2. E8, upper end of the H1 ratios.**
- Text: «(отношения 0,007–2,5)».
- Evidence: the maximum is `maps.E8.H1["std1e+07"].W_D`, label `W_D_r_std1e+07_F` = 2.44994. Rounded to the precision used, that is 2,4; 2,5 is a double rounding.
- Fix: «0,007–2,4».

**A1.3. E8, S2 at 10⁷ s⁻¹.**
- Text: «а в S2 при 10⁷ с⁻¹ сильнее в 64–3400 раз».
- Evidence: the minimum is `maps.E8.H1["std1e+07"].S2`, label `S2_x_std1e+07_F` = 63.461. The maximum is `S2_x_rf1e+07_eps_mol` = 3428.
- Fix: «в 63–3400 раз».

**A1.4. E8, H4 at 10⁸ s⁻¹.**
- Text: «при 10⁸ с⁻¹ – 0,10–0,58».
- Evidence: in `maps.E8.H4["std1e+08"]` the maximum is `W_D_z_chi0.05_std1e+08` = 0.574748. The minimum is `W_D_r_chi0.1_std1e+08` = 0.104967.
- Fix: «0,10–0,57».

**A1.5. E10 at 3·10⁷ s⁻¹ in W\_C.**
- Text: «при 3·10⁷ с⁻¹ он равен 0,048 в W\_C».
- Evidence: the sentence is about the coherence contribution s\_C·F\_amp. `maps.E10.W_C_r_std.levels[k=3e7].F_amp_coherence` = 0.0474896, which rounds to 0,047. The value 0,048 is F\_amp without s\_C (`F_amp_K` = 0.04775).
- Fix: «0,047». W\_D 0,097 is correct (0.097183).

**A1.6. Even range of form K in S2 (RU, H3 paragraph).**
- Text: «у формы K он равен 3,1–4,2·10⁻³ в S2».
- Evidence: over S2 form K cells without relaxation, the maximum is `maps.E1[S2_z_none0_K_chi0.1_m0.1_cry4a].even_range` = 0.00414995. That rounds to 4,1·10⁻³, so 4,2 is a double rounding. If only m = 1 is meant, the range is 3,1–3,9·10⁻³.
- Fix: «3,1–4,1·10⁻³». The W\_C value 3,0–3,5·10⁻⁴ and the W\_D value 6,1–7,8·10⁻⁴ are correct.

**A1.7. E5 median k½ in W\_C (RU, «Скорость k½»).**
- Text: «медиана равна 5,9–6,0·10⁷ с⁻¹ в W\_C».
- Problem: the stored result differs.
  - `maps.E5.k_half_W_C_chi0.05_std.k_half_quantiles[2]` = 5.849·10⁷.
  - `maps.E5.k_half_W_C_chi0.1_std.k_half_quantiles[2]` = 5.915·10⁷.
  - These are «5,8–5,9·10⁷». They exclude, through `nanquantile`, the 80 and 135 axes whose k½ lies above 10⁸.
- The text's values come from a different calculation. If those axes are counted as larger values (from `maps_axes.npz`, code 3), the medians are 5.879·10⁷ and 5.987·10⁷, which gives 5,9–6,0. That calculation was done after the run and is not marked as such.
- Fix: give the results.json value, or state the method and mark it as computed after the run.
- W\_D is correct with either method: 3.47·10⁷ from results.json, 3.54–3.55·10⁷ with out-of-scale axes counted.

### A2. Statements stronger than the data or contradicting them

**A2.1. Form E against form K and the polarisation forms (RU, E11).**
- Text: «Нечётный диполь формы E повёрнут к диполю формы K на 102–170°, то есть смотрит почти против него, как у форм поляризации в W\_C.»
- Problem (a): at the low end the dipole is far from opposite. Model 2 gives 102° at r̂ (`maps.E11.W_C_r_none0_theta0.1_M2.d_odd_angle_to_K_deg` = 101.96), 109–113° at r̂ under relaxation and 126–132° at x.
- Problem (b): the polarisation forms in W\_C point roughly against K only on the CISS axis z. These angles are computed from `maps.E1[...].d_odd`:

| CISS axis | Angle of P-same, P-LH and P-0 to K | Angle of form E to the P forms |
| --- | --- | --- |
| z | 124–135° | 5–19° |
| x | 33–43° | 96–140° |
| r̂ | 34–55° | 70–127° |

- So "как у форм поляризации" is wrong on x and r̂.
- The same claim appears in «Что это значит» («её нечётный диполь смотрит против диполя формы K») and in «Письма» («форма E с диполем против формы K»).
- Fix: «повёрнут к диполю формы K на 102–170° (в среднем около 140°); при оси z он почти совпадает по направлению с диполями форм поляризации, а при осях x и r̂ расходится с ними на 70–140°».

**A2.2. Dominance over P-0 at D = 0 (RU, E3).**
- Text: «Перевес формы K над формами поляризации и контролем P-0 при D = 0 возвращается (подробности в пояснении к H1)».
- Problem: for P-0 by |F| the claim fails. In `maps.E3[...].ratios_noD.K_over_P0_F`:
  - W\_D on axis x: 0.613, 0.498 and 0.573, so P-0 is stronger than K.
  - W\_C on x: 8.6–9.3; W\_C on r̂: 5.9–7.3; W\_D on r̂: 9.3–9.8. All are below 10.
  - By ε\_mol (`K_over_P0_eps`) the ratio is 7.5–780, and it is below 10 at z in W\_C for two of the three levels (7.50 and 8.33).
- The H1 paragraph gives no P-0 numbers at D = 0, so the cross-reference does not support the claim.
- Fix: «Перевес формы K над формами поляризации при D = 0 возвращается; над контролем P-0 по ε\_mol он составляет 7,5–780 раз, а по |F| – 0,5–3900 раз, и при оси x в W\_D P-0 сильнее K».

**A2.3. "Only small Δχ" at p = 0.5 (RU, «Форма K»).**
- Text: «При осях x и r̂ и p = 0,5 исключено всё без релаксации и при дефазировке любой скорости, а допустимыми становятся только малые Δχ при случайных полях 10⁷–3·10⁷ с⁻¹.»
- Problem: at 3·10⁷ s⁻¹ of random fields, R1 admits large Δχ.
  - `maps.E2.regions["W_D_x_rf3e+07_cry4a"]`, p = 0.5: Δχ = 0,05π for all S, and Δχ = 0,1π, the largest in the grid, for S = 1 and 0,8.
  - `["W_C_r_rf3e+07_cry4a"]`: Δχ = 0,05π for S = 1 and 0,8.
  - At 10⁷ s⁻¹ only 0,005π and 0,01π are admitted, which matches the text.
- Fix: «при случайных полях 10⁷ с⁻¹ допустимы Δχ = 0,005π и 0,01π, а при 3·10⁷ с⁻¹ – и 0,05–0,1π».

**A2.4. p = 0.5, axis z, without a stated level (RU, «Форма K»).**
- Text: «При p = 0,5 при оси z допустимы Δχ = 0,005π (в W\_D только при части S) и в W\_C при части S ещё 0,01π.»
- Problem: this follows the sentence about dephasing at 3·10⁷ and 10⁸, but the statement holds only without relaxation and at weak levels. Elsewhere:
  - Dephasing 10⁸ admits Δχ = 0,05π in W\_C at z for S = 1, 0,8 and 0,6 (`regions["W_C_z_std1e+08_cry4a"]`).
  - Random fields 10⁷ admit 0,05π in W\_C and 0,01π for all S in W\_D.
  - Random fields 3·10⁷ admit 0,05π and 0,1π in both pairs.
- Fix: add «без релаксации» and name these exceptions.

**A2.5. k½ "coincides with" the measured dephasing.**
- RU «Что это значит»: «k½ дефазировки, 3,3–6,2·10⁷ с⁻¹ и выше 10⁸ с⁻¹ при оси z в W\_D, совпадает с измеренной … дефазировкой 3–6·10⁷ с⁻¹».
- The same claim appears in the RU figure title «Нечётный диполь падает вдвое внутри измеренной дефазировки», in the README heading "k½ falls inside the measured dephasing", and in the README alt text "it falls to half inside the dephasing measured in Cry4a".
- Evidence from `hypotheses.H4.k_half`:
  - In W\_C and W\_D, 8 of 12 combinations lie inside 3–6·10⁷ and 4 lie above: W\_C r̂ at 6.2067 and 6.2141·10⁷, and W\_D z above 10⁸.
  - Over all 16 combinations, 10 lie inside and 6 above.
  - In the figure itself, 2 of the 3 plotted curves cross 0,5 to the right of the shaded band: S2 z at 7.0·10⁷ and W\_C r̂ at 6.2·10⁷.
- Fix: «лежит внутри измеренной дефазировки или выше неё (в 10 сочетаниях из 16 внутри, в 6 выше)», and a matching figure title.

**A2.6. "5–10 times above threshold at this dephasing" (RU, «Что это значит», Релаксация).**
- Text: «При этой дефазировке вклад когерентности остаётся в 5–10 раз выше порога дипольного теста (E10)».
- Problem: «этой» refers to the measured 3–6·10⁷. The factor 5–10 holds only at the grid level 3·10⁷ (0.0475/0.01 = 4,7 and 0.0972/0.01 = 9,7). At 6·10⁷, interpolating linearly in lg k between the 3·10⁷ and 10⁸ levels as for k½, s\_C·F\_amp is about 0,030 in W\_C and 0,052 in W\_D, which is about 3 and 5 times the threshold.
- The README already says "At 3·10⁷ s⁻¹".
- Fix: «При 3·10⁷ с⁻¹ вклад когерентности остаётся в 5–10 раз выше порога».

**A2.7. "Reversed" dominance in the real pairs.**
- RU «Гипотезы»: «Перевес формы когерентности над формами поляризации подтвердился только в S2, а в реальных парах он обратился (H1).» README: "The dipolar coupling turns H1 around."
- Problem: the reversal holds by ε\_mol, where all 30 + 30 ratios are 0.025–0.32. By |F|, form K stays stronger than the larger polarisation form in 15 of 30 F comparisons (`hypotheses.H1.W_C/W_D.comparisons`):

| Comparison | K/P ratio by \|F\| |
| --- | --- |
| W\_C, axis x | 11.27, 6.47, 1.84, 9.57, 4.32 |
| W\_D, axis x | 1.71–2.45 |
| W\_D, axis r̂ | 1.91–8.24 |

- Fix: «по ε\_mol он обратился во всех сравнениях, а по |F| форма K сильнее форм поляризации в 15 сравнениях из 30, но кроме одного меньше чем в 10 раз».

**A2.8. Dependence on J stated as a result.**
- RU «Физика CISS»: «Перевес форм поляризации поэтому зависит от J пары, которая в Cry4a не измерена.» README: "so the size of the reversal depends on the J of the pair."
- Problem: J = 0 throughout the run, so no computation shows a dependence on J. This is an inference from Efimova & Hore (2008).
- Fix: «может зависеть от J» / "may depend on J", marked as an inference from the literature.

### A3. Statement that goes beyond the frozen protocol

**A3.1. "The objection gets a numerical answer."**
- RU «Скорость k½» and «Что это значит»: «Нечётный диполь модели переживает её примерно наполовину, и возражение о быстром затухании когерентности в белке получает ответ числом». README "What it means": "which answers with a number the objection…".
- Protocol rule («Интерпретация», paragraph «k½ и измеренная дефазировка»):
  - The reading "переживает … больше чем наполовину, и возражение … получает ответ числом" is reserved for k½ > 6·10⁷.
  - k½ < 3·10⁷ means the mechanism needs protection.
  - The middle class 3–6·10⁷ has no pre-registered reading.
- Result: 10 of 16 combinations, and 8 of 12 in W\_C and W\_D, fall in the middle class. Only 6 of 16, and 4 of 12, fall in the class the protocol ties to this reading.
- Fix: describe the result neutrally («внутри измеренного диапазона диполь падает примерно вдвое; благоприятное прочтение протокола (k½ > 6·10⁷) выполнено в 6 сочетаниях из 16, неблагоприятное (k½ < 3·10⁷) – ни в одном»), or mark the stronger reading as the author's interpretation.

### A4. Wording error

**A4.1. "Upper border of the class" (RU, «Скорость k½»).**
- Text: «У W\_C с осью r̂ k½ = 6,2·10⁷ с⁻¹ лежит у верхней границы класса».
- Problem: its class is «выше 6·10⁷», and 6,2·10⁷ lies just above the lower border of that class, which is the upper end of the measured range. `hypotheses.H4.k_half["W_C_r_chi0.1_std"]` gives k\_half = 6.2141·10⁷ with class "above 6e7".
- Fix: «лежит чуть выше границы классов 6·10⁷ с⁻¹».

### A5. README "Contents" lists files that do not exist yet

**A5.1.**
- "SHA256SUMS.txt | SHA-256 of every file in this folder": there is no such file in `mag_s3c/` (mag_s3b has one).
- "results/checks/ … the check of the results section": there is no such file in `results/checks/`.
- `results/exploratory/` (explore\_after\_run.py, commits 88b9073 and f276873) is not listed.
- Fix: add the files before release, or edit the table.

---

## B. Minor suggestions

1. **Scope of "5136".** RU «Перевод в физику» says «Из 5136 сочетаний основной сетки». The 5136 combinations include the 8 E10 cells (96 combinations). The main grid alone gives 5040: R1 2413, R2 2772, R3 3541, dipole 1378, antipodal 410. The numbers 2445, 2804, 3573, 1442 and 457 are correct for the main grid plus E10. Write «основной сетки и ветки E10» or «геометрий S2, W\_C и W\_D».
2. **K/P-0 measure.** In RU H1, «(отношение K к P-0 равно 0,046–0,61)» is by ε\_mol. By |F| the ratio is 0,035–21: `maps.E1[K chi0.05 m1].K_over_P0_F`, with K > P-0 on axis x in both pairs and on r̂ in W\_D. Add «по ε\_mol».
3. **Local-estimate table.** The values are the smaller (tightest) bound over χ\_b = 0,05π and 0,1π; all entries reproduce in that reading. At χ\_b = 0,1π alone, W\_D R2 and R3 round to 0,0012 and 0,0044. State «наименьшая по χ\_b».
4. **E5 share below 3·10⁷ in W\_D.** `class_shares` gives 27,4% at χ\_b = 0,05π and 27,9% at 0,1π, so write «27–28%». W\_C at 7,6–8,3% → «8%» is fine.
5. **E14 median.** «По всем ячейкам W\_C и W\_D медиана этой доли равна 0,08%». 0,078% holds for all 844 crystal maps, including the 24 composite-pair cells. The 820 W\_C/W\_D cells alone give 0,075%, and the 718 cells without the D = 0 and Grüning variants give 0,088%. Name the set.
6. **E14 dipole norm.** «её норма около 10⁻²⁰»: `maps.E14.crystal[*].dipole_norm` ranges from 0 to 8,1·10⁻²⁰, with median 4·10⁻²². Write «не больше 10⁻¹⁹».
7. **E14 "only slightly higher".** «лишь немного выше»: in W\_D the maximum 0,23 is 27% above 0,18.
8. **Missing "without relaxation".** «85–95% осей CISS и 83–96% осей рецептора» (RU «Что это значит», README "Exploratory maps") hold without relaxation. At 10⁸ s⁻¹ of dephasing they are 45–68% (E5) and 51–93% (E14), and at random fields 3·10⁷ s⁻¹ they are 0%. Add «без релаксации».
9. **E4 in «Живой опыт».** «при закреплённом азимуте нечётная часть доходит до нейрона и при p = 0 (E4)» rests on the ratio transfer, which the E4 bullet states correctly. Scaled to the amplitude transfer (ε\_x·F\_amp/F, my estimate; the run did not compute it), the four W\_C cells with CISS axis z give 0,006–0,007, below 0,01. The other 20 cells give 0,013–0,25. Add «по переносу отношения».
10. **Slowest cells.** «около 50 минут» (RU deviation; README "about 50 minutes"): the archive `seconds` give 54–62 minutes for these 15 items. Two more references, `ref_W_C_std3e+07_cr2_1e2` and `cr2_1e3`, took 34 and 44 minutes, which is also more than half an hour. That contradicts the run\_notes line "every other item takes at most about 25 minutes".
11. **"Worst axis of the grid".** «худшей из осей сетки» should say «из трёх осей основной сетки». On the E5 sphere, axes near the zero circle give smaller projections.
12. **README Q1 row.** "up to 15% in 22 of 42 cells" should read "above 1% (up to 15%) in 22 of 42 cells".
13. **README "as large as the even range or larger"** for ε\_mol 0,56–1,49. At 0,56 the odd half-range is about half the even range; the RU «сравнимую» is accurate.
14. **README "margin of 5–18".** Write "5–18 times".
15. **README mechanism.** "A likely mechanism…" should say that the protocol did not test it, as RU does.
16. **README "The question" (outside the checked sections).** "Two kinds of spin relaxation … run up to 10⁸ s⁻¹": random fields only go to 3·10⁷ s⁻¹.
17. **Terminology.** «аполярная стопка» should be the protocol's term «неполярная стопка» (I7).
18. **Two cell-4 processes.** «Оба процесса проверки остановлены»: the text mentions only one start of cell 4. Run notes show two, at 16:54:15 and 16:56:08 (the drive log confirms). Say «запущена по ошибке дважды».
19. **"At the measured level".** «то есть на уровне измеренной» (README "At the measured dephasing") covers 3·10⁷ and 10⁸. 10⁸ lies above the measured 3–6·10⁷; the protocol's wording is «охватывают измеренные».
20. **Even-range sentence (RU H3).** The ranges are without relaxation; add that. The evidence that D narrows the range is E3: at D = 0 form K has 3,5–3,9·10⁻³ in W\_C and 3,4–3,7·10⁻³ in W\_D. Cite it, because comparing S2 with the real pairs mixes geometry and D.
21. **RU Q1 paragraph.** «узор несёт более тонкую угловую структуру» is an explanation the run did not test. «их ячейки раскладываются точно» should be «с невязкой не выше 0,14%».
22. **m = 0.1 ratios not given.** The protocol (H1, E1) promises K/P ratios at m = 0,1 without a verdict. The text does not give them, and `cells.csv` has no K\_over\_P columns; they are only in `results.json maps.E1`.
    - S2: 18–177 by ε\_mol and 13–79 by F.
    - W\_C: 0,003–0,03 and 0,002–1,4.
    - W\_D: 0,003–0,02 and 0,005–0,96.
    - The E1 bullet's «диполем» is partly covered: the CSV has only d\_odd·z and |d\_odd|.
23. **E7 wording.** «F\_amp падает до 0,12–0,25 и 0,01–0,11 по модулю» gives ratios to the RadicalPy values. The absolute F\_amp at z is 0,0005–0,0033. Write «до 0,12–0,25 и 0,01–0,11 своего значения».
24. **xyz amplitude.** «около 1–3%» (RU «Цепочка и кристалл», README "about 1–3%"): the values are 0,87% and 3,4%. Write «0,9–3,4%».
25. **RU «Физика CISS».** «без связи форма когерентности сильнее в 68–4200 раз» holds by ε\_mol. By F, at axis x in W\_D the ratio is only 4,6–5,7. Add «по ε\_mol».
26. **Telegram timing.** «опубликован в Telegram 10 ноября до прогона»: the repository only shows the link commit 5c3eb15 at 19:42:51 UTC, 7 s after the run started (19:42:44). Quote the post's own timestamp to support "до прогона".
27. **Understatement.** «в части сочетаний формы поляризации подходят к форме K ближе чем в 10 раз»: 59 of 60 comparisons fail, and by ε\_mol the polarisation forms exceed K in all 60.
28. **Machine name.** RU «на рабочей машине» and README "on the calibration machine": harmonise.
29. **Precision.** «0,0195» (E10) has three significant figures while neighbouring values have two.
30. **Sealed line.** The sealed line itself starts with "INVALID": the trial run was invalid through I1. One clause would make this clear.

## C. The author's writing rules

- **Em dash.** There is no em dash (U+2014) or other dash variant in either text. RU has 131 en dashes and README 56. Four U+2212 minus signs mark negative numbers (E11, E12), which is correct.
- **«не X – это Y» and negative definitions.** None found.
- **Verbless formula-like lists.**
  - Both figure captions are verbless formula-like lists with «·»: «*results.json подтверждающего прогона, H1 и карта E3 · форма K при χ\_b = 0,05π и m = 1, формы поляризации при χ = 0,1π, без релаксации*» and «*… H4 · форма K при χ\_b = 0,1π и m = 1*». Suggested rewrite: «Рисунок построен по results.json подтверждающего прогона (H4) для формы K при χ\_b = 0,1π и m = 1.»
  - Chains of elliptical fragments («при 3·10⁷ с⁻¹ – у 81% и 91%, при 10⁸ с⁻¹ – у 45% и 68%» in E5; similar chains in E8) are borderline; optional.
- **Mirrored aphoristic endings.** None found in RU or README.
- **Optional affirmative rewrites.**
  - «по-прежнему не сильнее форм поляризации в 10 раз ни в одном сравнении» → «во всех сравнениях отношение 0,007–2,4, ниже порога 10».
  - «В S2 формы поляризации по-прежнему не исключены ни в одном сочетании» → «…допустимы во всех сочетаниях».
  - «дипольной части не имеет» → «дипольная часть равна нулю до 10⁻¹⁹».

## D. Verified correct

These are listed briefly; every value was recomputed.

- **Validity and counts.** valid, grid\_ok, rates\_ok, curves\_full, blas\_single\_thread and counts\_ok are all true. The counts 574, 54 and 24 are correct, and the 24 curves are complete (8 cases × 3). There are 184 comparisons, 8 of them known in advance, and the 13 verdicts are as stated. Q4 is met and Q1–Q3 are not.
- **Invariants.** All eleven values are correct: 3,4·10⁻¹⁴, 3,6·10⁻¹⁴, 6,1·10⁻¹⁶, 1,9·10⁻¹⁴, 1,1·10⁻¹³, 5,4·10⁻¹³, 2,3·10⁻¹⁴, 1,4·10⁻⁷, 1,9·10⁻¹⁴, 244 iterations, 1,0·10⁻¹⁴, 6·10⁻⁹, 1,7·10⁻¹⁴ on 224 cells, 1,1·10⁻¹⁶, 1,2 and 1,5·10⁻¹⁵ on 844 maps, and 3,9·10⁻¹².
- **H1–H5 table and known values.**
  - The table is correct in RU and README alike. Every range endpoint matches `hypotheses.*`.
  - The smallest margins are: H1 S2 13,4; H2 6,57 and 7,76; H3 5,05 and 15,3; H4 1,59, 1,61 and 1,66; H5 1,11, 1,09 and 1,08.
  - The known values are 253, 1541, 460, 218, 1,0006, 1,0012, 1,040 and 1,086.
  - P-same is the larger polarisation form in 60 of 60 comparisons.
- **Verdict rules follow the frozen protocol.**
  - H1 compares K (χ\_b = 0,05π, m = 1) with max(P-same, P-LH) at χ = 0,1π over 40 combinations.
  - H2 uses the median over 360 axes.
  - H3 requires both F and F\_amp, so the stricter decides.
  - In H5 the S2 range excludes the known comparisons, as the protocol requires.
  - No H4/H5 comparison is flagged near the zero circle.
- **Remaining numbers.**
  - k½: all 40 entries of the table, the 10/6 class split and the rf range 2,7–7,9·10⁶ are correct.
  - Q1: 0,14%, 974, 22 of 42, 15%, 2·10⁻⁵ and 160 are correct.
  - Q2: 16 of 144, 8,9% and 8,1%, 20–83%, 10–23%, +103% and 5,6·10⁻⁶ are correct.
  - Q3: 32 of 32 for random fields, 9 of 32 for dephasing, 12%, W\_D and E7 are correct.
  - E2: 2445, 2804, 3573, 1442 and 457 are correct, as are 120 of 120, 53 of 120, 94/104, 55/56, the single p = 1 exception, the z-axis and 3·10⁷/10⁸ statements, the 9% linearity, and all local-estimate values (0,0016π = 0,0051 rad = 0,29°, 0,00052π, 0,0036, 0,0011, 0,013, 0,0043, 0,0014π, 0,00055π, 0,0039π, 0,0023π and 3–9×).
  - All numbers of E3, E4, E5, E7, E9, E11, E12, E13 and E14 are correct except those listed in A and B.
  - Tables: comparisons\_H1\_H5.csv has 184 rows, matching results.json with 0 mismatches. k\_half.csv has 32 rows with 0 mismatches, e2\_regions.csv has 6432 rows, and cells.csv has 1016 rows.
  - The Efimova & Hore (2008) reference (Biophys. J. 94(5): 1565–1574, doi 10.1529/biophysj.107.119362) and its summary are correct: partial cancellation at 2,0 ± 0,2 nm.
- **Deviations and provenance.** The times, the 15 items (12 P-same/P-LH at 10⁸ plus 3 references), the third blind check from 17:07:39 to 17:53:47 with I11 bit-identical, and 11 of 13 trial verdicts are all as stated.

## E. Counts

- **Checked.**
  - The RU section has 604 numeric tokens. About 460 are result values, counts, percentages or times (range endpoints counted separately), which I recomputed from results.json, the npz files, the run files or the tables, or traced to the logs. The rest are protocol parameters, which I checked against the frozen text.
  - The README sections have 226 numeric tokens, about 130 of them distinct result values, mostly repeating RU values. I checked them against the data and against RU.
  - I also checked about 40 qualitative claims ("все", "ни одно", "единственное исключение", which form or axis is larger, signs, directions, classes).
- **Real errors: 18.**
  - 7 places where numbers do not match the data, involving 9 numbers (A1.1–A1.7).
  - 8 statements stronger than or contradicting the data (A2.1–A2.8).
  - 1 interpretation beyond the frozen protocol (A3.1).
  - 1 wording error (A4.1).
  - 1 README contents mismatch (A5.1).
- **Minor suggestions:** 30 (section B), plus the caption style issue under the writing rules (section C).
- **Writing-rule violations:** no em dash and no «не X – это Y». The two figure captions break the rule against verbless formula-like lists.

---

## F. What was changed after this check (11 November 2026, UTC)

All 18 real errors (A1.1–A5.1) were corrected in the results section of the protocol and in README.md, with the
values and wordings given above: the E6 medians (0,07 and 0,04), the E8 ranges (0,007–2,4; 63–3400; 0,10–0,57), E10
in W_C (0,047), the even range of S2 (3,1–4,1·10⁻³), the E5 median of k½ (now from results.json, 5,8–5,9·10⁷ s⁻¹ over
the axes on the scale, with the shares above 10⁸ s⁻¹), the direction of form E (102–170° from form K; close to the
polarisation forms only on the axis z), the control P-0 at D = 0 (by ε_mol and |F| separately), the admissible Δχ
at p = 0,5 with the levels named, k½ described as inside the measured dephasing or above it (figure title, README
heading and alt text included), the margin of 5–10 tied to 3·10⁷ s⁻¹, the reversal of H1 stated by ε_mol with the 15
comparisons by |F| where form K stays stronger, the dependence on J stated as an inference, the readings of k½ given
as the protocol defines them, the border of the classes, and the contents of the README, which now lists the check
file and results/exploratory; SHA256SUMS.txt, which the README also lists, was generated as the last step before the
release.

Of the minor suggestions, 1–25 and 27–30 were applied, among them the ratios of form K at m = 0,1 in E1 (22) and
the clause on the word INVALID of the sealed line (30); for 26 the text first cited FREEZE.txt for the time the link
to the post came. The two figure captions are now full sentences. The section also gained, after this check, the
exploratory recomputations of results/exploratory (the CISS scale of Kobori et al. 2026 and the H3 cells of the trial
run), the dipolar sweep of Gwynn, Adams & Petruccione (2026) as the closest precedent of branch E3, and rewritten
letters: to Exeter as a transfer of the comparison of CISS forms by Smith et al. (2025) from the even anisotropy to
the odd part, and to the group of Kobori in Kobe as a third addressee.

These changes went through a second pass of the same checker on commit eac8486
(check_of_the_results_section_pass2.md). It confirmed the fixes above except two: this section claimed the contents
of the README fixed while SHA256SUMS.txt did not exist yet, and item 26 still claimed the post came before the run on
evidence of minute resolution. In the changed and new text it found five real errors and seventeen minor points;
section G of that file lists what was changed after it.
