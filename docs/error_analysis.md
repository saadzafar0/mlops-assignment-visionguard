# Task 2.6 — Image-specific error hypotheses

Project: **23l-0829_A2_Ops**. The hypotheses below follow visual inspection of the actual test-error grids. They are possible explanations, not confirmed causes. Image numbers run left-to-right across the top row, then the bottom row. Full paths and scores are in the corresponding error CSVs.

## Transfer model

Grid: `reports/generated/evaluation/transfer_errors.png`.

1. **Misty landscape — AI → Real, p(AI)=0.086.** Photographic lighting, haze and terrain textures may resemble real landscape images; downsampling can hide small generation defects.

2. **Animal with mixed features — AI → Real, p(AI)=0.154.** Realistic fur, shallow depth of field and earth colours may outweigh unusual anatomy in the learned features.

3. **Person against a vivid patterned background — Real → AI, p(AI)=0.834.** The saturated graphic background may dominate the person and resemble AI artwork seen during training.

4. **Paired studio-style portraits — AI → Real, p(AI)=0.183.** Realistic faces and photographic composition may hide unusual body proportions and generated surface details.

5. **Small animal in a white maze — AI → Real, p(AI)=0.222.** The scene resembles an overhead photograph with ordinary lighting; strong artwork cues are absent.

6. **Office portrait — Real → AI, p(AI)=0.758.** Polished portrait lighting and framed artwork in the background may resemble generated/editorial images; the score is only moderately above the threshold.

7. **Colourful illustrated portrait — Real → AI, p(AI)=0.746.** Strong saturation, smooth gradients and decorative detail overlap with AI-art styles, although the dataset labels it real.

8. **Monochrome hooded portrait — Real → AI, p(AI)=0.730.** Smooth tonal shading and a dark, simple background may resemble generated portrait aesthetics; its score is close to the chosen threshold.

9. **Sketch and lake-image composite — AI → Real, p(AI)=0.332.** The publication-like mixed layout and photographic lake panel may dilute evidence from the generated content.

10. **Pop-art mouth collage — AI → Real, p(AI)=0.365.** Simple graphic shapes and print-like texture can resemble human-made pop art; stylisation alone does not establish AI provenance.

## Baseline model

Grid: `reports/generated/evaluation/baseline_errors.png`.

1. **Colourful illustrated portrait — Real → AI, p(AI)=0.678.** The small CNN may associate vivid colours and smooth illustration textures with AI artwork, producing a false accusation.

2. **Monochrome formal portrait — Real → AI, p(AI)=0.669.** The score is barely above the chosen threshold; smooth studio shading may contribute, but the prediction is weak evidence.

3. **Fantasy landscape with purple fields — AI → Real, p(AI)=0.493.** The model gives an almost neutral score; global pooling may miss spatial inconsistencies in the stylised landscape.

4. **Animal with mixed features — AI → Real, p(AI)=0.494.** Limited semantic features may fail to recognise unusual anatomy when local fur and background textures look photographic.

5. **Framed painting with an impossible scene — AI → Real, p(AI)=0.494.** After resizing, the ordinary gallery/frame context may dominate the smaller generated scene.

6. **Colourful fantasy tree and planets — AI → Real, p(AI)=0.500.** The image overlaps with human-made illustration styles present in the real class; the near-0.5 score shows uncertainty.

7. **Four-portrait collage — AI → Real, p(AI)=0.506.** Mixed photographic and illustrated panels resemble article/collage layouts; small panels lose detail at 128 pixels.

8. **Dense surreal illustration — AI → Real, p(AI)=0.524.** Downsampling and global averaging can discard small structural inconsistencies amid very dense textures.

9. **Web gallery screenshot — AI → Real, p(AI)=0.526.** Most of the image is interface whitespace and tiny thumbnails, so layout may dominate the source-image content.

10. **Portrait artwork displayed on a wall — AI → Real, p(AI)=0.533.** The wall/display context resembles ordinary photographs of human-made artwork; the model remains uncertain about provenance.

## Interpretation

The transfer model misses several photograph-like AI images and falsely flags some real portraits and illustrations. This is consistent with learning style and presentation shortcuts, which the EDA suggested were possible. The baseline has many near-neutral scores; its high validation-chosen threshold sends those uncertain images to the real class, improving its false-positive cost while missing many AI images.

These observations do not change the locked test thresholds or trigger retraining. Improvements such as handling gallery screenshots, studying near duplicates, and testing broader sources belong to a future development round with a fresh holdout set.
