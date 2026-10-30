"""Committed, replay-safe knobs (part of the admitted source)."""

# Keep each chain's pre-refine module as a panel candidate instead of accepting the last
# render-passing refinement blindly (refines can distort geometry or add unseen features).
KEEP_GEN = True

# Tilt flat objects toward the camera by comparing the judge front view silhouette with the
# photo (renders only); replaces the fixed 15-degree tilt in the final wrapper.
PHOTO_TILT = True

# Refinement prompt guard: keep the module's pose unless the renders clearly contradict the photo.
KEEP_POSE = True
