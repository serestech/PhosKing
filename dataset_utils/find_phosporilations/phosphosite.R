library("tidyverse")

# phosphosite = read_tsv("C:/Users/dani_/Desktop/Phosphorylation_site_dataset.tsv", skip = 3)
KPSDB = read_tsv("C:/Users/dani_/Desktop/Kinase_Substrate_Dataset.tsv", skip = 3)

kinase_counts = KPSDB %>% 
  select(KINASE, KIN_ACC_ID) %>% 
  group_by(KINASE) %>% 
  count() %>% 
  arrange(desc(n)) %>% 
  # left_join(KPSDB %>% select(KINASE, KIN_ACC_ID),
  #           by = "KINASE") %>% 
  unique()

KPSDB %>% 
  select(GENE:KIN_ORGANISM, SUB_ACC_ID, SUB_MOD_RSD) %>% 
  filter(SUB_ACC_ID == 'Q9Y5B0') %>% 
  View()

KPSDB %>% 
  select(GENE:KIN_ORGANISM, SUB_ACC_ID, SUB_MOD_RSD) %>% 
  mutate(aminoacid = str_sub(string = SUB_MOD_RSD, end = 1)) %>% 
  filter(!aminoacid %in% c('S', 'T', 'Y')) %>% 
  View()

KPSDB %>% 
  group_by(KIN_ORGANISM) %>% 
  summarise(n = n())

KPSDB %>% 
  mutate(aminoacid = str_sub(string = SUB_MOD_RSD, end = 1)) %>% 
  group_by(aminoacid) %>% 
  summarise(n = n())
