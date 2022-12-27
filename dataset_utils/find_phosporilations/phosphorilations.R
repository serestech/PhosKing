library("tidyverse")

# Workdir should be the root of the project

metadata = read_tsv("./data/dataset_uniprot/metadata.tsv")
features = read_tsv("./data/dataset_uniprot/features.tsv")

phosphorilations = features %>% 
  select(!feature_type) %>% 
  filter(str_sub(feature_description, end = 4) == "Phos")

phosphorilations_with_kinase = phosphorilations %>% 
  filter(str_detect(feature_description,
                    pattern = ".+ by .+")) %>% 
  separate(col = feature_description,
           into = c("amino_acid", "extra_data"),
           remove = TRUE,
           extra = "merge",
           sep = ";") %>%
  mutate(amino_acid = str_sub(amino_acid, start = 8),
         amino_acid = str_to_title(amino_acid), 
         extra_data = str_remove(extra_data, "^.+by "),
         extra_data = str_remove(extra_data, ";.+$")) %>%
  rename(kinases = extra_data) %>%
  mutate(kinases = str_replace_all(kinases,
                                   pattern = ", | and | or ",
                                   replacement = ",")) %>%   
  mutate(kinases = str_replace_all(kinases,
                                   pattern = "autocatalysis,?",
                                   replacement = "")) %>%
  mutate(kinases = str_replace_all(kinases,
                                   pattern = " ?host ?",
                                   replacement = "")) %>% 
  filter(kinases != "")

write_tsv(phosphorilations_with_kinase, 'data/dataset_uniprot/phosphorilations_with_kinases.tsv')

# interesting_uniprots = phosphorilations_with_kinase %>% 
#   pull(uniprot_accession) %>% 
#   unique()

# metadata %>% 
#   filter(uniprot_accession %in% pull(phosphorilations_with_kinase, uniprot_accession)) %>% 
#   group_by(org_name) %>% 
#   count() %>% 
#   arrange(desc(n))
