- vignettes: working folder for collage generator
  - combine vignettes, vignettes_new_level_0 and HE_001_rotated
  - tubules in vignettes are excluded
  - artery from HE_001 is excluded 
- patch_annotation:
  - original patch of slide and tubules masks
  - HE_001 from multistain and 11000_16000_0 from FFPE PostRep 001.svs
  - 13600_18000_0 and 20000_14800_0 from FFPE PostRep 001.svs. Full annotation for multiple labels. Attache a labelmap.txt.
- data_archived
  - vignettes: from Dr. Coley
  - vignettes_new: vignettes from FFPE PostRep 001.svs level 1, (11000,16000) on top left, 1028*1028
  - vignettes_new_level_0: same as vignettes_new but on level 0, 4112*4112
  - normal proximal and distal tubular segments_HE_001: vignettes from MultiStain normal proximal and distal tubular segments/HE_001, 3000*3000
  - HE_001_rotated: rotate 90 anti-clock to match the angle of tubules in vignettes_new (updated)
  - FFPE PostRep 001: annotate biopsy on top. take out all glom and arteries. full annotation on two selected patches 13600_18000_0 and 20000_14800_0.