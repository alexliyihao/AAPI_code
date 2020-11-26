- vignettes: working folder for collage generator
  - combine vignettes_new_level_0, HE_001_rotated, FFPE PostRep 001 and multistain and FFPE PostRep 003-006， 008-010
  - train：slide 001,004-006,008,010, all multistain but tubule_HE_001(name as HE_001_...)
  - validation: slide 003, 009, tubule_HE_001
 ```
  ds          label          
train       artery              49
            distal_tubule      126
            glomerulus         157
            glomerulus_gs        6
            proximal_tubule    150
validation  artery              18
            distal_tubule       81
            glomerulus          81
            glomerulus_gs        4
            proximal_tubule     79
 ```           
  - test: slide 002 and 007 (not included here)
  - exclude some incompleted tubules and glom
- patch_annotation:
  - original patch of slide and tubules masks
  - HE_001 from multistain and 11000_16000_0 from FFPE PostRep 001.svs
  - 13600_18000_0 and 20000_14800_0 from FFPE PostRep 001.svs. Full annotation for multiple labels. Attache a labelmap.txt.
- other archived data transfered to data repo


