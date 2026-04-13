# NASA ROSES A.8 WATER25 — Review-of-Reviews Memo

**Reviewer:** Matt Bromley, DRI
**Date:** 2026-04-13
**Scope:** Feedback on your panel review sheets for proposals 0002, 0004, 0040, 0058, 0111, 0035, 0132

**Files I could read (4 of 7):** 0002, 0004, 0040, 0111
**Files locked by OneDrive (cloud-only stubs) at time of review:** reviews 0035, 0058, 0132 and proposal copies 0035, 0132. Re-run me after forcing those to download ("Always keep on this device") and I will fold them into this memo.

---

## What I checked against

Evaluation criteria per the A.8 solicitation are **Relevance**, **Intrinsic Merit**, and **Cost**. Type 1 proposals also require the reviewer to speak to end-user engagement, ARL progression, and a credible path to operations. I cross-referenced each of your field responses against the proposal narrative and looked for four things:

1. **Alignment with proposal** — did you represent claims accurately?
2. **Completeness vs. criteria** — are all required fields substantively addressed?
3. **Strength of reasoning** — are strengths and weaknesses balanced and defensible?
4. **Tone / professionalism** and **AI "voice"** — does the writing sound like you?

---

## Cross-cutting patterns (apply to all reviews)

**Voice & tone — overall positive.** Across the four reviews I could read, the prose reads like a working scientist — first person, concrete, occasional informal phrasing ("I generally like the idea…", "The idea of informing precision irrigation … is exciting"). That is **not** how an AI tends to draft reviews unprompted. No red flags for AI voice.

**A few recurring surface issues to clean up before submission:**

- **Em-dashes rendered as "—-" or "-–" in several places** (e.g., 0002 "9 km to 10–-30 m", "well-developed.—", 0111 "7-–10 day"). These look like copy/paste artifacts. Search each review for `—-`, `-–`, and `–-` and normalize.
- **Double spaces** show up in a few spots ("appears  plausible", "timeline is dominated by the development of the model and validation. "). Not a big deal but worth a quick pass.
- **Minor typos**: "overwatering", "optimizaiton" (0002 Relevance), "Golf course are" (0002 Relevance weakness — should be "Golf courses are"), "mils" should be "miles" (0040 Summary), "tens of thousandsof miles" / "managestens of thousands" (0040 Relevance — missing space), "course resolution" should be "coarse resolution" (0111 Weaknesses).
- **Cost field is under-used in 3 of 4 reviews.** On 0002, 0040, and 0111 you essentially say "I don't have enough budget detail." That's a fair and honest answer, but Cost also covers **management plan and project timeline** per the solicitation — those you *can* evaluate from the narrative. Consider adding a sentence or two on staffing plausibility, milestone realism, or risk management rather than leaving the Cost fields nearly empty.
- **The "weakness" fields are generally lighter than "strength" fields.** Not a problem if the proposal is strong, but on 0002 and 0004 the asymmetry is striking enough that a panel chair may push back. If you genuinely see few weaknesses, saying so explicitly (as you did on 0111: "I do not see any weaknesses in the cost estimate") is better than leaving a short or empty field.

---

## Proposal 0002 — *RemoteWater: Decision Support System for Optimizing Golf Course Water Management* (Secondary)

**Rating given:** radioFamily30 = /3 (Good, per standard NASA rubric)

**Alignment check:** Accurate overall. Your summary correctly captures the SMAP/Sentinel-1 downscaling → FAO-56 dual-Kc → irrigation prescription pipeline, the five pilot sites (AZ, NV, LA, MS, KY), and the 15–25% water savings target (proposal: "sustained 15–25% water savings in routine periods and up to" 30% in drought). ARL 2→5 is consistent with the proposal's stated endpoint.

**Gaps / suggested additions:**

- Your **Intrinsic Merit weakness** on downscaling from 9 km SMAP to 10–30 m is your strongest technical critique. You could strengthen it by naming the specific concern — e.g., that CNN/GAN super-resolution of passive microwave to sub-field scale over irrigated turf has little published validation precedent, and the validation sites span just two climate regions. Right now the critique is more intuition than citation.
- **Cost field is thin.** The proposal is three years, multi-institution, with UAV/field deployment at five sites. Even without line-item access, the Cost fields could note whether the scope-to-staffing ratio looks realistic, whether the phased shadow-to-advisory schedule is achievable in 36 months, and whether the open-source / containerization commitment is adequately resourced.
- **Relevance weakness** ("Golf courses are a relatively minor water user…") is fair but a bit blunt. The proposal does argue the methods transfer to sports fields, parks, and agriculture — you may want to acknowledge that transfer argument and say whether you found it credible, rather than leaving the weakness standing unrebutted.

**Tone:** Professional. Typos noted above. One sentence fragment to fix: "The engagement strategy is well-developed.— The proposal describes…" (stray em-dash after a period).

---

## Proposal 0004 — *Strengthening Water Market Performance with Satellite Remote Sensing* (Secondary)

**Rating given:** (no radioFamily30 value captured — **verify you actually selected a rating** before submitting; only radioFamily0 appears set)

**Alignment check:** Accurate. The "45% overstated" claim, MWD + WA Ecology + Great Salt Lake scope, and ARL 3→7 all match the proposal. Your list of four methods is supported by the narrative (DID with OpenET, ML gradient boosting for counterfactual ET, established hydrologic/crop-coefficient via OTTER/CropSyst/AquaCrop, and traditional regulatory accounting).

**Largest gap: this review is incomplete.** Only four text fields have content (Summary, Relevance strengths, Intrinsic Merit strengths, and the sourceXML metadata). The following required fields are blank:

- Relevance weaknesses
- Intrinsic Merit weaknesses (and continuation)
- Cost strengths and weaknesses
- Overall Comments
- Final rating (radioFamily30)

This is almost certainly a saving/export artifact rather than your actual opinion, but as exported the review would be rejected by the panel chair. **This is the single most important fix to make before submission.**

**Content suggestions for the missing fields (starting points, not drafts):**

- *Intrinsic Merit weaknesses:* ARL 3→7 in three years is aggressive for a tool that spans three very different regulatory regimes (MWD, WA TWRP, Utah GSL); the transition to routine use in each agency requires sustained engagement beyond the 36-month window. The four-method comparison is rigorous but creates coordination and QA/QC load for a team that also has to execute a Maxar habitat analysis.
- *Relevance weaknesses:* The work is highly regionally tailored (three specific agencies/basins); scalability to other water markets (e.g., Colorado-Big Thompson, Australia-style markets) is asserted but not developed.
- *Cost:* Comment on the plausibility of splitting effort across four methods plus the WorldView habitat analysis within a single Type 1 project.

**Tone:** What's there reads fine — possibly slightly more congratulatory than the other three reviews ("scientifically rigorous", "real credibility", "right mix of expertise"). Balance against the weaknesses once you fill them in.

---

## Proposal 0040 — *Monitoring Critical Waterway Infrastructure using Radar Observations of Soil Moisture and Disturbance* (Primary — Truckee Canal)

**Rating given:** radioFamily30 = /5 (Very Good)

**Alignment check:** Accurate. 31-mile length (proposal: "31.2-mile-long"), age (built 1903–1905 per proposal), 80% detection accuracy target, EVS flight cost-sharing via LA-to-Oregon detour — all supported.

**Strengths of your review:**

- This is the most carefully reasoned of the four. Your Intrinsic Merit weaknesses (IoT sensor robustness for validation, small team / PI-and-Co-I concentration, EVS flight dependency, thin operational integration detail) are all specific and defensible.
- Good balance between praise for the sensor-application match (NISAR/UAVSAR + soil moisture + disturbance) and the concrete risks.

**Gaps / suggested additions:**

- **Relevance weakness is under-developed.** You correctly note the proposal is Truckee-Canal-only with no explicit plan to generalize. The solicitation weights "broad and scalable impact" heavily, so this weakness deserves a second sentence: *why* the Truckee-only focus matters given that Reclamation manages "tens of thousands of miles" of canals and the PI invokes that scalability in the Relevance pitch.
- **Cost fields nearly empty.** As with 0002, both Cost strength and weakness essentially say "not enough information." Given that you rated this a 5/Very Good, the panel chair will want at least a sentence on whether the *management plan and schedule* (which you *do* have) look realistic — especially since small-team risk is one of your Intrinsic Merit weaknesses. That's a cost-side observation, not just an intrinsic-merit one.
- **Overall comments is one sentence.** For a Primary review, panel chairs typically expect a couple of sentences tying strengths and weaknesses into your recommendation.

**Tone:** Good. Minor typos: "mils long" → "miles long"; "managestens of thousands" → "manages tens of thousands"; "issues with validation of the soil moisture data.The" missing space before "The"; "for disturbance and soil moisture" is fine but the preceding sentence has a dropped word ("spatial resolution of the downscaled products is in line with detection requirements"). Quick proofread pass recommended — as Primary this will be read carefully.

---

## Proposal 0111 — *Regional Forecasting of ET and Crop Water Demand Using OpenET, GraphCast, and NISAR* (Primary — Ogallala)

**Rating given:** radioFamily30 = /3 (Good)

**Alignment check:** Accurate and technically strong. Your GraphCast critique is supported by the proposal, which explicitly lists only "temperature, pressure, wind" at six-hour intervals — you are right that humidity/VPD and net radiation, which drive PM-equation reference ET, are not among the listed GraphCast outputs. Your SIMS-overestimation-under-deficit-irrigation concern is a legitimate and well-known limitation. 7–10 day forecast horizon, NISAR 100 m soil moisture, and Ogallala/SGP geography all match.

**Strengths of your review:**

- This is your most substantive technical critique of the four. The GraphCast variable-availability point and the SIMS deficit-irrigation bias are exactly the kinds of domain-specific concerns NASA wants a remote-sensing hydrologist on the panel to raise.

**Gaps / suggested additions:**

- **You pre-emptively dismiss your own GraphCast critique** ("This issue may be overcome in the machine learning approach, which will be demonstrated in the validation phase"). That softens a legitimate weakness. Consider leaving the concern sharp and letting the panel weigh whether the ML approach is an adequate response. You can say "the proposal proposes to address this via …, but …" rather than withdrawing the point.
- **Relevance weakness is very brief** — a single sentence noting limited relevance to heterogeneous regions. Given the solicitation's emphasis on broad and scalable impact, this deserves one more sentence (e.g., canopy-cover, crop-rotation, and irrigation-method heterogeneity in the Central Valley, Mississippi Delta, or Northwest will challenge transfer).
- **Cost strength and weakness are one-liners** ("budget is in-line with the scope of work" / "I do not see any weaknesses"). That's fine as a conclusion, but Cost also covers timeline and management plan. You already flag in Intrinsic Merit that "the project timeline is dominated by the development of the model and validation" with too little time for end-user iteration — that observation actually belongs in the Cost/Schedule weakness field as well (or instead), because it is a schedule critique.

**Tone:** Good. Typos: "course resolution" → "coarse resolution"; "7-–10 day" (stray en-dash); "relative humidity (vapor pressure deficit)" — VPD is not the same as RH, it's derived from RH and temperature, so phrasing as "relative humidity (and by extension vapor pressure deficit)" or simply "humidity and VPD" would be more precise.

---

## Rating coherence across reviews

| Proposal | Your rating | Field 30 value | Rough consistency |
|---|---|---|---|
| 0002 (golf) | Good | /3 | Matches tone: exciting concept, aggressive downscaling risk, minor user sector |
| 0004 (water markets) | **MISSING** | (none set) | **Must select** before submit |
| 0040 (canal seepage) | Very Good | /5 | Matches tone: specific weaknesses but strong sensor-application match |
| 0111 (ET forecasting) | Good | /3 | Matches tone: strong concept, real technical concerns |

A Good / Very Good / Good / (missing) distribution looks internally consistent with the written content — 0040 clearly gets the stronger rating, and the two Goods are each Goods for different reasons (0002 for scope, 0111 for technical risk). No calibration concerns from what I can see.

---

## Priority checklist before you submit

1. **Fix the OneDrive download issue** so the locked files open, then re-run this check on 0035, 0058, and 0132.
2. **Complete review 0004** — five fields plus the rating are blank. This is the biggest single risk.
3. **Proofread for em-dash artifacts** (`—-`, `-–`, `–-`) across all reviews.
4. **Fill out Cost fields** with at least a management-plan / schedule sentence on 0002, 0040, and 0111.
5. **Typo sweep**: "mils" → "miles" (0040), "course" → "coarse" (0111), "optimizaiton" (0002), "managestens" / missing spaces (0040).
6. **Relevance-weakness depth** on 0040 and 0111 — add one sentence each on why the geographic scope matters for "broad and scalable impact."

---

## On the AI-voice question

I looked specifically for signs that any of your text was AI-drafted: stock transitional phrases ("Furthermore,", "It is worth noting that,", "In conclusion,"), over-balanced "strengths/weaknesses" symmetry, hollow qualifiers ("potentially significant", "meaningful contribution"), and smooth connective tissue without concrete detail. **I do not see those patterns here.** Your reviews have first-person opinions ("I generally like…", "I appreciate that…", "I am concerned about…"), domain-specific technical specifics (SIMS deficit bias, GraphCast variable set, SMAP 9 km native, FAO-56 dual-Kc, OTTER/CropSyst), and a characteristic honesty about limits ("I am limited to the information provided by the review packet"). That sounds like you, not a model.

The few places that read a little more generic are in 0004 — the strengths language there ("scientifically rigorous", "real credibility", "the right mix of expertise") is a bit more formulaic than your other reviews. Worth a rewrite pass when you complete that one.
