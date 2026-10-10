# Iba Lens design

Brand feel: clinical, calm, paper (a lab report, not a dashboard).

Palette
- Paper `#faf8f5`, card `#ffffff`, rule `#e7e2dc`, ink `#1c1917`, muted `#6b6460`
- One accent: crimson `#b4232a` (likely positive, primary action)
- Status only: amber `#b7791f` (unclear), green `#2f7d50` (likely negative)

Type: Inter for text, IBM Plex Mono for numbers and slide IDs.

Rules
- Always light theme; no hero video, no gradients.
- One accent colour. Amber and green appear only as status.
- Every state has a design: empty queue selection, loading, single cell, field.
- The proof sits next to the claim: boxes on the field, a heatmap for each flagged cell, test metrics one tab away.
- The safety line ("a lab scientist confirms every result") is always visible.
- Mobile: the stacked layout must still read top to bottom as queue → report → evidence.
