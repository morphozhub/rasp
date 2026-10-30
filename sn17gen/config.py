"""Committed, replay-safe knobs (part of the admitted source)."""

# Keep each chain's pre-refine module as a panel candidate instead of accepting the last
# render-passing refinement blindly (refines can distort geometry or add unseen features).
KEEP_GEN = True

# Tilt flat objects toward the camera by comparing the judge front view silhouette with the
# photo (renders only); replaces the fixed 15-degree tilt in the final wrapper.
PHOTO_TILT = True

# Refinement prompt guard: keep the module's pose unless the renders clearly contradict the photo.
KEEP_POSE = True

# Refinement looks at the organizer judge's exact four front views (front_left 30/0, front_right
# 330/0, front_below 0/15, front_above 0/-30) and must list discrepancies before the module.
JUDGE_REFINE = True

# Numeric colour/material rules appended to the generation system prompt.
COLOUR_RULES = True

# First generation of every chain on gpt-6-astra (reasoning high) instead of gpt-6.1-sol medium.
ASTRA_GEN = False
