library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

neural_os <- read.csv("../neural/onlyshort_evals.csv")
neural_ps <- read.csv("../neural/plusshort_evals.csv")
neural_ps_small <- neural_ps %>% 
  filter(train_set_size == "Small") %>%
  mutate(recall = tp / (tp+fn)) %>%
  select(alphabet_size = alph, 
         tier_size = tier, 
         language_class = class, 
         factor_width = k, 
         threshold = j, 
         index = i, 
         split = test_type,
         accuracy, 
         precision, 
         recall, 
         f1 = fscore,
         brier_score = brier,
         model = network_type
         )

ff_os <- read.csv("../FlexFringe/FlexFringe/models-os/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv")
ff_ps_small <- 
  read.csv("../FlexFringe/FlexFringe/models-ps-small/0.0.1.0.0.0.searchdeep.0.ini/eval_combined.csv") %>%
  mutate(split = str_remove(split, "Test")) %>%
  mutate(model = "FF") %>%
  select(-data_size, -ini, -model_path, -data_path, -last_modified)


both <- rbind(ff_ps_small, neural_ps_small) %>%
  filter(!split %in% c("Train", "Dev")) %>%
  filter(alphabet_size < 64)

both %>%
  ggplot(aes(x = model, y = f1)) + 
  geom_boxplot() + 
  facet_grid(~split)



