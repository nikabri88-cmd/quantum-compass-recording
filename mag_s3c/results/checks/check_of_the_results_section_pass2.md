# MAG-S3c: second pass of the independent check (commit eac8486)

## What I checked and how

- **Scope.** I checked the diff f276873..eac8486:
  - in `mag_s3c`: the RU results section, `README.md`, `results/run_notes.md`, `figures/make_figures.py` and the redrawn `k_half.png`, plus the new `results/checks/check_of_the_results_section.md`;
  - outside `mag_s3c`: `.zenodo.json`, `CITATION.cff` and the root `README.md`.
- **Repository state.** The working tree is clean, and my copies of the four mag_s3c text files equal eac8486 byte for byte. The frozen text above «## Результаты подтверждающего прогона» and the «Источники» section are still byte-identical to the frozen protocol.
- **Ground truth.**
  - From the first pass: `results.json`, `maps_axes.npz` and the run files (cell `seconds`).
  - New in this pass: `results/exploratory/*.json` and logs.
  - Code: `explore_after_run.py`, `mag_s3c.py` (`summarize`, `ref_solutions`, `rebuilt_yields`, `cell_weights`, `rates`, `use_blind_rates`), `mag_s3b.py` (`polarised`, `turned`, `cell_operators`, `harmonic_fit`) and `scripts/pipeline.sh`.
  - I read everything with `python3 -I`.
- **Literature I could read.**
  - Gwynn, Adams & Petruccione (arXiv:2605.22130), full text.
  - Smith, Tallapudi, Denton & Kattnig (arXiv:2505.01519, html).
  - The Kobe University press release.
  - Cantarella et al. (2026) could not be fetched: the URL is not on the allowlist and search did not find it. For it I relied on the summary in the frozen protocol.
- **Copy of my first report.** Sections A–E of the repository copy are identical to my first report; only section F is new.

---

## A. Real errors in the changed or new text

**A1. "Within a quarter" understates the F_amp shortfall on 60 directions (RU, «Пробные вердикты»).**
- Text: «…а F\_amp остаётся в пределах четверти своего значения на 2000 направлениях.»
- Evidence: on 60 directions F\_amp is 74,0–75,7% of its 2000-direction value at the same rates.
  - W\_C, Cry4a rates (h3_cry4a_60.json against results.json): 0.04280/0.05780 = 0.7406; 0.03743/0.05046 = 0.7418; 0.04468/0.06037 = 0.7401.
  - W\_C, substitute rates (h3_blind_60.json against h3_blind_2000.json): 0.7406, 0.7416, 0.7402.
  - W\_D: 0.7569, 0.7516, 0.7555 (Cry4a) and 0.7530, 0.7483, 0.7518 (substitute).
- So W\_C falls 26% short, slightly more than a quarter.
- Fix: «…а F\_amp ниже своего значения на 2000 направлениях на 24–26%». The exploratory README says "within about a quarter", which is acceptable.

**A2. The literature item misdescribes how the protocol used Gwynn et al. (RU and README).**
- Text: «Протокол ссылался на эту работу только по поводу взаимной ориентации тензоров…» / "The protocol cited it only for the relative orientation of the tensors".
- Evidence: the frozen `MAG-S3c_protocol_v0.4_ru.md` cites the work in three places:
  - line 26, «Что уже существует»: the odd part on a model pair without relaxation;
  - line 34, the table: the odd part is robust at non-collinear hyperfine and dipolar axes and suppressed by a second collinear nucleus;
  - line 355, «Ограничения», «Три ядра»: a second collinear nucleus suppresses the antisymmetric part, so the three-nucleus pair may overestimate it.
- Fix: «Протокол ссылался на неё по поводу нечётной части модельной пары, взаимной ориентации тензоров и подавления вторым коллинеарным ядром, а её развёртку D не использовал.» Make the matching change in README.

**A3. The EPR caveat is stronger than its source (RU «Масштаб CISS», README "The scale of CISS").**
- Text: «ЭПР на замороженном образце действительную и мнимую когерентность не различает (Cantarella et al. 2026).» / "since EPR on frozen samples does not tell real from imaginary singlet–triplet coherence".
- Evidence: the only summary of Cantarella et al. in the repository, the frozen protocol's table, says «Действительную и мнимую синглет-триплетные когерентности ЭПР различает только на ориентированных образцах или в первые мгновения». The new text drops the early-time case. Kobori et al. used time-resolved EPR (frozen protocol; the Kobe press release confirms "time-resolved electron paramagnetic resonance").
- Fix: «ЭПР различает действительную и мнимую когерентность только на ориентированных образцах или в первые мгновения после рождения пары (Cantarella et al. 2026)…», and the same in README.

**A4. Section F of `check_of_the_results_section.md` overstates the fixes.**
- Text: "All 18 real errors (A1.1–A5.1) were corrected … and the contents of the README."
- Problem: A5.1 is still open.
  - README "Contents" still lists "SHA256SUMS.txt | SHA-256 of every file in this folder", but `git ls-tree eac8486 mag_s3c` has no such file (`mag_s3b/` and the repository root have theirs).
  - The other two points of A5.1 are fixed: the check file now exists, and results/exploratory is listed.
- F also says the additions "went through a second pass of the same checker" without recording its outcome.
- Fix:
  - Generate `mag_s3c/SHA256SUMS.txt` as the last step before release, or have F say it will be generated then.
  - Add one line to F with the outcome of this pass.

**A5. The new run\_notes line rounds an upper bound down.**
- Text: "every other item of the pair step took at most 23 minutes."
- Evidence: the next slowest item after the 17 named ones is `ref_S2_std3e+07_cry4a` at 23.25 minutes (`seconds` = 1395.2 in the run folder).
- Fix: "at most 24 minutes" or "at most about 23 minutes".
- The other numbers of the line are correct: 54.1–61.6 minutes → 54–62, and 33.8 and 43.6 minutes → 34 and 44.

---

## B. Fixes from the first pass

**All correct, except as noted.** I checked every item A1.1–A4.1 and B1–B25, B27–B30 against the data. The new values and wordings are right:

| Item | New text | Data |
| --- | --- | --- |
| E6 medians | 0,07 and 0,04 | 0.0712 and 0.0437 |
| E8 | 0,007–2,4; 63–3400; 0,10–0,57 | as stated |
| E10 | 0,047 and 0,019 | 0.04749 and 0.019457 |
| S2 even range | 3,1–4,1·10⁻³ | as stated |
| Even range at D = 0 | 3,5–3,9·10⁻³ (W\_C) and 3,4–3,7·10⁻³ (W\_D) | 3.479–3.918·10⁻³ and 3.406–3.697·10⁻³ |
| E5 shares | 8%; 27–28%; above 10⁸ 4–7% and 7% | 80/2004 and 135/2004; 148/2004 and 147/2004 |
| E5 median | 5,8–5,9·10⁷ | 5.849 and 5.915·10⁷ |
| E11 | 5–19° at z; 70–140° at x and r̂ | as stated |
| E3, K/P-0 | 7,5–780 by ε\_mol; 0,5–3900 by \|F\| | as stated |

- **Figure.** The new k½ title «Дефазировка гасит нечётный диполь вдвое при 3–7·10⁷ с⁻¹» matches the three plotted curves: 3.31, 6.21 and 6.96·10⁷. The README alt text matches too.
- **E1 m = 0,1 ratios.** They are correct: S2 18.4–177.4 and 13.4–78.8; W\_C 0.0027–0.032 and 0.0019–1.38; W\_D 0.0031–0.022 and 0.0047–0.96. They include the published 24 (z) and 159 (x).
- **E14.** "на 5% и 27%" is correct: 1.052 and 1.272.
- **E2 at p = 0,5.** The admissible Δχ per level are correct:
  - z: rf 10⁷ gives up to 0,05π in W\_C and up to 0,01π in W\_D; rf 3·10⁷ gives 0,05π and 0,1π in both pairs; std 10⁸ gives 0,05π in W\_C at S = 1, 0,8 and 0,6.
  - x and r̂: rf 10⁷ gives 0,005π and 0,01π; rf 3·10⁷ gives W\_C r̂ 0,05π at S = 1 and 0,8, and W\_D x 0,05π at all S and 0,1π at S = 1 and 0,8.
- **Readings of k½.** The protocol readings (6 of 16, none below 3·10⁷, 10 without a reading) and the class border are correct.

**B26, the timing of the Telegram post, is fixed only in part.**
- RU now says «опубликован в Telegram 10 ноября до прогона (…; ссылка на него пришла в 19:42 UTC, FREEZE.txt)».
- Problem: the cited time has minute resolution, while the same section gives the run start as 19:42:44 UTC. FREEZE.txt itself received the link in commit 5c3eb15 at 19:42:51 UTC, 7 s after the start.
- So the evidence does not show «до прогона», although the claim may well be true. The README "Status" line and `.zenodo.json` make the same claim.
- Quoting the post's own timestamp to the second would settle it. I did not open the Telegram post.

---

## C. New passages checked from scratch

**C1. Gwynn, Adams & Petruccione (2026), RU H1 paragraph and README H1 paragraph: correct.** The paper (full text) matches the description:
- **Model.**
  - The main sweeps use a single nucleus on the donor.
  - Table I sweeps D₀ over [−1, 1] mT in 31 steps.
  - The CISS birth state is Eq. (2), |ψ\_I⟩ = cos(χ/2)|S⟩ + sin(χ/2)|T₀⟩, with real coefficients.
- **Findings.**
  - The antisymmetric component of the singlet yield "dips, under CISS, at D₀ = 0 mT".
  - The authors write "suggesting a competing rather than cooperative relationship between dipolar-induced and CISS-induced symmetry breaking". The RU «называют конкурирующими» is fine; «считают» would mirror "suggesting" more closely.
- **"Polarisation-type state" is right.** In `mag_s3b.py`, `polarised(chi, n, +1)` builds exactly cos(χ/2)|S⟩ + sin(χ/2)|T₀ along n⟩ as the birth state of P-same and P-LH. Smith et al. (2025) state that this Luo–Hore form gives ⟨S₁z − S₂z⟩(0) = −sin χ, which is a spin polarisation. Gwynn's recombination state cos(χ/2)|S⟩ − sin(χ/2)|T₀⟩ equals our P-LH recombination (`polarised(chi, n, −1)`), so their model is our form P-LH with one nucleus and no relaxation. Saying so would make the precedent exact (suggestion D10).
- **E3 sentence.** «Ветка E3 переносит этот вопрос … на обе формы CISS и на релаксацию» is correct: E3 has K and P forms at no relaxation and at 10⁶ s⁻¹.

**C2. The H3 discrepancy of the trial run (RU «Пробные вердикты», README results paragraph).**
- **Numbers.**
  - 66 even and 55 odd real harmonics up to degree 10 (1+5+9+13+17+21 and 3+7+11+15+19) against 30 antipodal pairs is correct, so the fit is underdetermined.
  - `harmonic_fit` uses `np.linalg.lstsq`, which returns the minimum-norm solution, as the exploratory README says.
  - F on 60 directions is 0,7–2,7·10⁻⁴ at Cry4a rates and 1,1–3,7·10⁻⁴ at substitute rates, so «около 10⁻⁴» is fine.
  - On 2000 directions at substitute rates, F\_amp is 0,072–0,084 in W\_C and 0,207–0,248 in W\_D. F is 0,61–0,87 and 3,2–3,8. The margins are 7,2–8,4 and 20,7–24,8, so «7–8» and «21–25» are correct.
  - The exploratory README table matches all 24 values, and "about 40%" is 34–42%.
  - The F\_amp shortfall is in A1.
- **Is it the run's code path?** Yes, for the values; there are two differences from the trial itself.
  - The script solves the references anew with the frozen `ref_solutions` (same `solve_X`).
  - It rebuilds the cells with the frozen `rebuilt_yields(cell_weights(...))`, the path of the run's rebuilt cells, and summarizes with the frozen `summarize`. `check_fit=False` drops only the degree-24 check and does not change F or F\_amp.
  - The grid is `fibonacci_sphere(30)` with antipodes, as `pipeline.sh smoke --grid-half 30`.
  - The rates are `b.BLIND_RATES["cry4a"]` = (6·10⁶, 5·10⁵), as in the trial. The Pool is created after `use_blind_rates()`, so the forked workers use them; the different outputs confirm it.
  - Difference 1: the trial computed the H3 cells directly. Rebuilt and direct cells are equal per direction by the reconstruction identity (I8 to 1,7·10⁻¹⁴ in the run), and the 60 check rows of scale.json reproduce the run to 1,4·10⁻¹¹, so this changes nothing.
  - Difference 2: the trial ran pre-freeze code, which was corrected during that run (protocol, «Раунд 3»). The recomputation shows that the frozen code fails H3 the same way on the trial's grid and rates. It therefore shows the grid is sufficient to explain the failure; it is not a replay of the trial (suggestion D4).

**C3. The CISS scale (RU «Масштаб CISS», README "The scale of CISS", exploratory README): the numbers are correct.**
- **Check rows.** The 60 check rows at 0,1π equal the run to ≤1,4·10⁻¹¹ relative in F, F\_amp, d\_odd·z and the even range.
- **Angles.** sin χ = 0,06 gives 0,01911π and sin²χ = 0,06 gives 0,07877π, so 0,019π and 0,079π are correct. 0,0255π and 0,0913π are also correct.
- **Responses** (form K, axis r̂; the stricter transfer is always F\_amp):

| Reading | W\_C | W\_D |
| --- | --- | --- |
| Amplitude, no relaxation and 10⁶ | 0,0102–0,0122 → «0,010–0,012» | 0,0311–0,0376 → «0,031–0,038» |
| Amplitude, dephasing 3·10⁷ | 0,0097 | 0,0198 → 0,020 |
| Triplet share, no relaxation and 10⁶ | 0,0407–0,0487 → «0,041–0,049» | 0,1236–0,1491 → «0,12–0,15» |

- «2–4 раза» corresponds to 2,0–3,8.
- The exploratory README table matches all 40 values. The sin χ cos χ scaling holds: predicted 0,2038 and 0,8081 against 0,202–0,203 and 0,806–0,808. The P-0 values 0,030–0,042, 0,036–0,052 and 0,005 are correct.
- **Kobori et al. sources.**
  - The numbers (about 8% intermediate, up to 6% late, frog cryptochrome, time-resolved EPR) come from the frozen protocol's table.
  - The Kobe press release gives no numbers and no organism. It confirms time-resolved EPR and the DOI 10.1021/acs.jpclett.6c02747, which `.zenodo.json` now references.
  - The texts stay within these sources except the EPR caveat (A3) and two small departures listed in D (the dropped «около» and «по подгонке модели»).

**C4. Letters and literature.**
- **Exeter.** The Exeter wording is correct. Smith et al. (2025) are at the University of Exeter, use FAD–W\_C of ClCry4a (PDB 6PU0) with dipolar coupling and random fields (1 μs⁻¹), compare the Luo–Hore polarisation form with the Fay coherence form, and measure only the even anisotropy. So «перенос … с чётной анизотропии на нечётную часть» is accurate.
- **Kobe.** The Kobe addressee is consistent with the press release: Prof. Kobori Yasuhiro, Kobe University. The P-0 responses it promises exist in scale.json.
- **Literature lists.** The Gwynn item's title and authors are correct; its description is wrong (A2).

**C5. Metadata outside mag_s3c.**
- `.zenodo.json` is correct:
  - H2 3,3–4,0 (3,29–4,01);
  - H5 at least 0,976;
  - H4 at least 79% (0,7926);
  - k½ 3,3–6,2·10⁷ with "above 1e8 for one axis";
  - random fields to 3e7 and dephasing to 1e8;
  - the Kobori caveat.
- `CITATION.cff` is correct, but it lacks "(above 1e8 for one axis)" (D12).
- In the root README v0.3.0 paragraph only the k½ phrase changed. Contrary to the coordinator's note, it does not state the ranges 3·10⁷ and 10⁸ s⁻¹, and it still says "with relaxation up to the singlet–triplet dephasing measured in Cry4a" (D11).

---

## D. Minor suggestions

1. **Mixed scope (RU «Релаксация»).** The range «3,3–6,2·10⁷ … и выше 10⁸ при оси z в W\_D» is for W\_C and W\_D, but «(в 10 сочетаниях из 16 внутри, в 6 выше)» counts all three geometries. For W\_C and W\_D the split is 8 of 12 inside and 4 above. State both or name the set.
2. **E3 wording.** «Над контролем P-0 он составляет … по |F| – 0,5–3900 раз»: a "перевес" of 0,5 times reads oddly. Suggest «Отношение K к P-0 при D = 0 равно 7,5–780 по ε\_mol и 0,5–3900 по |F|; при оси x в W\_D P-0 сильнее K».
3. **χ\_b wording.** «как наименьшая из двух значений χ\_b» can be read as "for the smaller χ\_b". The table holds the smaller estimate over the two χ\_b values (W\_C at 0,1π, W\_D at 0,05π). Write «как наименьшая из оценок при двух значениях χ\_b».
4. **Trial explanation.** «показывает, что расхождение даёт сетка пробы» (README "traces the difference to the grid"; exploratory README "The trial failed H3 through its grid"). The trial ran pre-freeze code, so write «объясняется сеткой пробы: на замороженном коде те же сетка и скорости дают H3 не выполненной».
5. **README margins.** "pass H3 with a margin of 7–25" should read "7–8 times in W\_C and 21–25 times in W\_D". `.zenodo.json` "with a margin of 5–18" should read "5–18 times".
6. **Kobori numbers.** RU «в 8% для промежуточной пары» should be «около 8%», as in the frozen protocol, README and exploratory README. The frozen source also says «по подгонке модели» and «промежуточного состояния»; keep the model-fit qualifier.
7. **Triplet-share levels.** Name the levels of the triplet-share range: «0,041–0,049 … без релаксации и при 10⁶ с⁻¹». At 3·10⁷ dephasing it is 0,039 in W\_C and 0,078 in W\_D.
8. **S = 1.** «Пара W\_C тогда стоит у порога … даже при полном полярном порядке и полной невзаимности» holds at S = 1. Axial disorder multiplies the response by g(S) up to 1,66, which gives 0,017–0,020 at S = 0,4, as the H3 interpretation of the protocol notes. Add «при S = 1».
9. **8% reading.** The text reports only the 6% reading. For 8% as an amplitude, W\_C is 0,0136–0,0163 and W\_D 0,041–0,050 (scale.json); one clause would cover it.
10. **Exact precedent.** State that Gwynn's birth and recombination states equal those of our form P-LH (C1).
11. **Root README.** "with relaxation up to the singlet–triplet dephasing measured in Cry4a" should become "with singlet–triplet dephasing up to 10⁸ s⁻¹ and random fields up to 3·10⁷ s⁻¹", as in `.zenodo.json` and `CITATION.cff`.
12. **CITATION.cff.** Add "(above 1e8 for one axis)" after 3.3–6.2e7, as `.zenodo.json` does.
13. **Dates.** F's header "(12 November 2026)" and the exploratory README "Computed on 12 November 2026" use a local date. The commits are 88b9073 at 21:54, f276873 at 22:02 and eac8486 at 22:15 UTC on 11 November, and every other time in the folder is UTC.
14. **F is incomplete.** F does not mention the rewritten «Письма» (Exeter with Smith et al. 2025, the new Kobe addressee) or the INVALID sentence. B22 and B30 are covered only by the item numbers.
15. **E1 wording.** «которые протокол давал без вердикта» should be «которые протокол велел давать без вердикта».
16. **INVALID line.** It could add that I1 also flagged cells of the previous build (protocol, «Раунд 3»). The reason given (substitute rates and small grid) is correct.
17. **Unpublished scripts.** The repository copy of my report names the folder of my scripts, which is not published. Say "my own scripts (not published)", or drop the path.

---

## E. Writing rules (changed and new text)

- **Dashes.** There is no em dash (U+2014) or other forbidden dash in the RU section, README, exploratory README, run\_notes, section F, `.zenodo.json`, `CITATION.cff` or the root README. Minus signs (U+2212) are used for −1 and the negative values, which is correct.
- **«не X – это Y».** None found. The only «это не» is the old factual clause «вердиктов гипотез это не касается».
- **Verbless formula-like lists.** None. Both captions are now full sentences. «Ближайший предшественник этого вывода – развёртка D…» is a nominal sentence with a dash copula, which is allowed.
- **Mirrored aphoristic endings.** None. The paired readings («Неблагоприятное … ни в одном, а благоприятное … в 6 из 16») sit mid-paragraph and carry information.

---

## F. Counts

- **Checked.**
  - About 190 numbers in the changed and new RU text, with each range endpoint counted separately.
  - About 90 in the changed README text, mostly repeating RU.
  - About 55 in the exploratory README and its tables.
  - About 35 in run\_notes, section F, `.zenodo.json`, `CITATION.cff` and the root README.
  - In total about 370, plus about 25 qualitative claims (directions, "all", "none", "only", readings) and the 34 fixes A1.1–A4.1 and B1–B30.
- **Real errors: 5** (A1–A5).
  - A1 and A5 are small numeric slips.
  - A2 and A3 are statements stronger than or contrary to their sources.
  - A4 is section F overstating that A5.1 was fixed.
- **Fix only partly done: 1** (B26, the timing of the Telegram post).
- **Minor suggestions: 17** (D1–D17).
- **Writing-rule violations:** none in the changed and new text.

---

## G. What was changed after this check (11 November 2026, UTC)

- **A1.** The RU section now says «F\_amp ниже своего значения на 2000 направлениях на 24–26%»; README and the exploratory README give 74–76% of the value on 2000 directions.
- **A2.** The Gwynn item in RU and README names the three places where the protocol cites the work (the odd part of a model pair, the relative orientation of the tensors, the suppression by a second collinear nucleus) and says that the protocol did not use its sweep of D.
- **A3.** The EPR caveat in RU and README follows the protocol's summary of Cantarella et al. (2026): EPR tells real from imaginary coherence only on oriented samples or in the first moments after the birth of the pair.
- **A4.** `mag_s3c/SHA256SUMS.txt` was generated as the last step before the release, and section F of the first report records the outcome of this pass.
- **A5.** run\_notes now says "at most 24 minutes".
- **B26.** RU, README, `.zenodo.json`, `CITATION.cff` and the release notes now say that the post came after the freeze at 19:40 UTC and no later than 19:42 UTC, the minute the run started. FREEZE.txt keeps its record of 10 November unchanged.
- **D1–D17.** All applied:
  - D1: «Релаксация» gives 8 of 12 inside and 4 above for the real pairs beside 10 and 6 of 16 with S2, and the favourable reading as 6 of 16, 4 of them in the real pairs.
  - D2: E3 gives the ratio of K to P-0 at D = 0 and says that P-0 is stronger by |F| on the axis x of W\_D.
  - D3, D15: the wordings as suggested.
  - D4: RU, README and the exploratory README say that the grid explains the trial and that the recomputation on the frozen code is no replay of the trial.
  - D5: "times" after the margins in README and `.zenodo.json`.
  - D6: «около 8%», the model fit and «промежуточного состояния» as in the frozen protocol, also in README, the exploratory README and the script.
  - D7: the levels of the triplet-share reading, with 0,039 and 0,078 at dephasing 3·10⁷ s⁻¹.
  - D8: p = 1 and S = 1 named; g(S) at S = 0,4 lifts W\_C to 0,017–0,020.
  - D9: 8% read as an amplitude gives 0,014–0,016 in W\_C and 0,041–0,050 in W\_D.
  - D10: the birth and recombination states of Gwynn et al. are those of form P-LH, so their model is P-LH with one nucleus and no relaxation; «считают скорее конкурирующими».
  - D11: the root README gives dephasing up to 10⁸ s⁻¹ and random fields up to 3·10⁷ s⁻¹.
  - D12: `CITATION.cff` adds "(above 1e8 for one axis)".
  - D13: 11 November 2026, UTC, in F, the exploratory README and the docstring of explore_after_run.py.
  - D14: F names items 22 and 30 of the first pass and the rewritten letters.
  - D16: the INVALID sentence adds that I1 flagged cells of the previous build («Раунд 3»).
  - D17: the path of the unpublished scripts is gone from the first report and from item 17 here.
